"""E13c: train the 20M byte transformer on a domain mix instead of books only.

Why: domains.py showed both tables and models are domain-bound (a books-only model loses to gzip
on Wikipedia), and mixed_table.py showed a mixed table closes the gap at no cost on books.
Same model, schedule and evaluation as train_big.py; each batch row is drawn from a domain with
fixed probabilities:
  books 60%  the 3 GB tier (train_big.big_stream, E10 holdout books excluded)
  wiki  25%  enwik8[0:85 MB]       (domains.py tests on enwik8[90 MB:])
  code  15%  CPython Lib/<subdir>  (domains.py tests on the top-level Lib/*.py)
Validation curve: the same books holdout as E10/train_big, so curves stay comparable.

Usage: .venv/bin/python lab/round2/train_mixed.py NAME STEPS
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "exp_tinymodel"), str(HERE.parent)]

from common import CKPT, CTX, ROOT, VOCAB, build, quantized, summarize, test_docs  # noqa: E402
from mixed_table import code_text  # noqa: E402
from train import lr_at, val_bpc  # noqa: E402
from train_big import BATCH, PEAK, RESULTS, big_stream  # noqa: E402

MIX = (("books", 0.60), ("wiki", 0.25), ("code", 0.15))


def streams():
    books, holdout = big_stream()
    wiki = np.frombuffer((ROOT / "corpus" / "enwik8").read_bytes()[:85_000_000], dtype=np.uint8)
    code = np.frombuffer(code_text().encode("utf-8"), dtype=np.uint8)
    return {"books": books, "wiki": wiki, "code": code}, holdout


def main():
    name, steps = sys.argv[1], int(sys.argv[2])
    torch.manual_seed(0)
    torch.set_num_threads(2)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    data, val = streams()
    names = [n for n, _ in MIX]
    probs = np.array([p for _, p in MIX])
    model = build("20m").to(device)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    other = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": other, "weight_decay": 0.0}],
                            lr=PEAK, betas=(0.9, 0.95))
    rng, val_rng = np.random.default_rng(0), np.random.default_rng(1)
    offsets = np.arange(CTX + 1)
    curve = []
    print(f"{name}: {model.params():,} params, {steps} steps x {BATCH} x {CTX} bytes; mix "
          + ", ".join(f"{n} {p:.0%} of {len(data[n]) / 1e6:.0f} MB" for n, p in MIX), flush=True)
    t0 = time.time()
    for step in range(steps):
        for group in opt.param_groups:
            group["lr"] = lr_at(step, steps, PEAK)
        rows = []
        for dom in rng.choice(len(names), size=BATCH, p=probs):
            stream = data[names[dom]]
            start = rng.integers(0, len(stream) - CTX - 1)
            rows.append(stream[start:start + CTX + 1])
        x = torch.from_numpy(np.stack(rows).astype(np.int64)).to(device)
        loss = F.cross_entropy(model(x[:, :-1]).reshape(-1, VOCAB), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0 or step == steps - 1:
            point = {"step": step, "minutes": round((time.time() - t0) / 60, 1),
                     "train_bpc": round(loss.item() / math.log(2), 4),
                     "val_bpc": round(val_bpc(model, val, device, val_rng), 4)}
            curve.append(point)
            print(json.dumps(point), flush=True)
        if step % 1000 == 999:
            torch.save(model.state_dict(), CKPT / f"round2_{name}.pt")
    torch.save(model.state_dict(), CKPT / f"round2_{name}.pt")
    out = {"name": name, "mix": MIX, "steps": steps, "batch": BATCH, "ctx": CTX, "peak_lr": PEAK,
           "train_minutes": round((time.time() - t0) / 60, 1), "curve": curve}
    for label, m in (("fp32", model), ("int8", quantized(model))):
        out[label] = summarize(m, device, test_docs())
        print(label, json.dumps({k: (round(v["avg_bytes"]), round(v["bpc"], 3)) for k, v in out[label].items()}), flush=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(out, indent=1))
    print(f"done {name}", flush=True)


if __name__ == "__main__":
    main()
