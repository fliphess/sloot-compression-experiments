"""Round 2: train the E10 20M byte transformer on the 3 GB corpus instead of the 90 books.

Same model, context and evaluation as lab/exp_tinymodel (E10); only the training data
differs: the 8,206 books of E8c's 3 GB tier (the 90 v1 training books + leakage-checked
extra books, list in lab/exp_table/results/exp_table_3gb.build.json). Batches are sampled
uniformly from the whole stream, so ~9,000 steps see ~300 MB of mostly unique text
(E10's 20M run saw 82 MB, ~1 epoch of the 90 books).
The validation curve uses the same 1 MB holdout as E10 (the end of the 90-book stream);
the training books that contain it are left out entirely.

Usage: .venv/bin/python lab/round2/train_big.py NAME STEPS
       e.g. .venv/bin/python lab/round2/train_big.py 20m_big 9000
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
sys.path[:0] = [str(HERE.parent / "exp_tinymodel"), str(HERE.parent)]

import common  # noqa: E402
from common import CKPT, CTX, ROOT, VOCAB, build, quantized, summarize, test_docs, train_bytes  # noqa: E402
from steno import split_books  # noqa: E402
from train import HOLDOUT, lr_at, val_bpc  # noqa: E402

BATCH, PEAK = 32, 6e-4
RESULTS = HERE / "results"
BOOK_LIST = ROOT / "lab" / "exp_table" / "results" / "exp_table_3gb.build.json"


def big_stream():
    """All 3 GB-tier books as one uint8 stream, without the book that holds the E10 validation holdout.

    The holdout is the last 1 MB of the 90-book stream (sorted by ID), so it covers the tail of
    the last training book(s). Those whole books are left out, so no holdout byte is trained on.
    """
    holdout = train_bytes()[-HOLDOUT:]
    skip, covered = set(), 0
    for path in reversed(split_books()[0]):
        skip.add(path)
        covered += path.stat().st_size
        if covered >= HOLDOUT:
            break
    small, big = ROOT / "corpus" / "gutenberg", ROOT / "corpus" / "gutenberg_big"
    paths = []
    for book_id in json.loads(BOOK_LIST.read_text())["book_ids"]:
        path = small / f"{book_id}.txt"
        if path not in skip:
            paths.append(path if path.exists() else big / f"{book_id}.txt")
    data = np.empty(sum(p.stat().st_size for p in paths), dtype=np.uint8)
    pos = 0
    for path in paths:
        raw = path.read_bytes()
        data[pos:pos + len(raw)] = np.frombuffer(raw, dtype=np.uint8)
        pos += len(raw)
    return data, holdout


def main():
    name, steps = sys.argv[1], int(sys.argv[2])
    torch.manual_seed(0)
    torch.set_num_threads(2)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    train, val = big_stream()
    model = build(name.split("_")[0]).to(device)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    other = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": other, "weight_decay": 0.0}],
                            lr=PEAK, betas=(0.9, 0.95))
    rng, val_rng = np.random.default_rng(0), np.random.default_rng(1)
    offsets = np.arange(CTX + 1)
    curve = []
    print(f"{name}: {model.params():,} params, {steps} steps x {BATCH} x {CTX} bytes = "
          f"{steps * BATCH * CTX / 1e6:.0f} MB sampled from {len(train) / 1e9:.2f} GB", flush=True)
    t0 = time.time()
    for step in range(steps):
        for group in opt.param_groups:
            group["lr"] = lr_at(step, steps, PEAK)
        starts = rng.integers(0, len(train) - CTX - 1, BATCH)
        x = torch.from_numpy(train[starts[:, None] + offsets].astype(np.int64)).to(device)
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
    train_minutes = (time.time() - t0) / 60
    torch.save(model.state_dict(), CKPT / f"round2_{name}.pt")

    docs = test_docs()
    out = {"name": name, "params": model.params(), "steps": steps, "batch": BATCH, "ctx": CTX, "peak_lr": PEAK,
           "sampled_mb": steps * BATCH * CTX / 1e6, "stream_gb": len(train) / 1e9,
           "train_minutes": round(train_minutes, 1), "torch": torch.__version__, "curve": curve}
    for label, m in (("fp32", model), ("int8", quantized(model))):
        out[label] = summarize(m, device, docs)
        print(label, json.dumps({k: (round(v["avg_bytes"]), round(v["bpc"], 3)) for k, v in out[label].items()}), flush=True)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(out, indent=1))
    print(f"done {name}", flush=True)


if __name__ == "__main__":
    main()
