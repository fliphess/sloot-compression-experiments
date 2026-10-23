"""E10: train one tiny byte-level model on the 90 training books, then evaluate it.

Fixed step counts (not time budgets) keep runs reproducible up to GPU
nondeterminism. The last 1 MB of the training stream is held out for the
validation curve only; the test set is never seen in training.

Usage: .venv/bin/python lab/exp_tinymodel/train.py NAME   (NAME in 1m, 5m, 20m)
"""
import json
import math
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import CKPT, CTX, ROOT, VOCAB, build, quantized, summarize, test_docs, train_bytes

RUNS = {  # name: (steps, batch, peak lr)
    "1m": (8000, 64, 2e-3),
    "5m": (4500, 64, 1e-3),
    "20m": (5000, 32, 6e-4),
}
WARMUP = 300
HOLDOUT = 1_000_000
RESULTS = ROOT / "lab" / "exp_tinymodel" / "results"


def lr_at(step, steps, peak):
    if step < WARMUP:
        return peak * (step + 1) / WARMUP
    progress = (step - WARMUP) / max(1, steps - WARMUP)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def val_bpc(model, val, device, rng):
    model.eval()
    starts = rng.integers(0, len(val) - CTX - 1, 128)
    x = torch.from_numpy(np.stack([val[s:s + CTX + 1] for s in starts]).astype(np.int64)).to(device)
    loss = F.cross_entropy(model(x[:, :-1]).reshape(-1, VOCAB), x[:, 1:].reshape(-1))
    model.train()
    return loss.item() / math.log(2)


def main():
    name = sys.argv[1]
    steps, batch, peak = RUNS[name]
    torch.manual_seed(0)
    torch.set_num_threads(2)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    data = train_bytes()
    train, val = data[:-HOLDOUT], data[-HOLDOUT:]
    gpu_train = torch.from_numpy(train.astype(np.int64)).to(device)
    model = build(name).to(device)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    other = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": other, "weight_decay": 0.0}],
                            lr=peak, betas=(0.9, 0.95))
    rng = np.random.default_rng(0)
    val_rng = np.random.default_rng(1)
    curve = []
    print(f"{name}: {model.params():,} params, {steps} steps x {batch} x {CTX} bytes "
          f"= {steps * batch * CTX / len(train):.2f} epochs of {len(train) / 1e6:.1f} MB", flush=True)
    t0 = time.time()
    for step in range(steps):
        for group in opt.param_groups:
            group["lr"] = lr_at(step, steps, peak)
        starts = torch.from_numpy(rng.integers(0, len(train) - CTX - 1, batch)).to(device)
        idx = starts[:, None] + torch.arange(CTX + 1, device=device)[None, :]
        x = gpu_train[idx]
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
            torch.save(model.state_dict(), CKPT / f"exp_tinymodel_{name}.pt")
    train_minutes = (time.time() - t0) / 60
    torch.save(model.state_dict(), CKPT / f"exp_tinymodel_{name}.pt")

    docs = test_docs()
    out = {"name": name, "params": model.params(), "steps": steps, "batch": batch, "ctx": CTX, "peak_lr": peak,
           "epochs": steps * batch * CTX / len(train), "train_minutes": round(train_minutes, 1),
           "int8_mb": round((model.params() + sum(p.shape[0] for p in decay) * 4) / 1e6, 2),
           "flops_per_byte": model.flops_per_byte(), "torch": torch.__version__, "curve": curve}
    for label, m in (("fp32", model), ("int8", quantized(model))):
        t1 = time.time()
        out[label] = summarize(m, device, docs)
        out[f"{label}_eval_minutes"] = round((time.time() - t1) / 60, 1)
        brief = {k: (round(v["avg_bytes"]), round(v["bpc"], 3)) for k, v in out[label].items()}
        print(label, json.dumps(brief), flush=True)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(out, indent=1))
    print(f"done {name}", flush=True)


if __name__ == "__main__":
    main()
