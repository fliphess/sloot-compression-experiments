"""Trick 2: a match model next to the neural model, for memory beyond its 512-byte context.

Match model (PAQ/cmix style, data-only, deterministic): hash the last MINLEN bytes; on a hit, follow
the earlier occurrence and predict the byte that came after it. Match length L grows while it keeps
being right, and resets on a miss.
Combination (an APM, all counts, integer-implementable): when the match model predicts byte b*, a
learned table indexed by (L bucket, quantised model probability of b*) gives q = P(next == b*):
  hit:  -log2 q
  miss: -log2(1 - q) - log2(p_model(x) / (1 - p_model(b*)))
  no prediction: -log2 p_model(x)
Ideal bits + 7 B container, model = round2 20m_big (3 GB books), fp32 on CPU.

Usage: .venv/bin/python lab/round2/match_model.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "exp_tinymodel"), str(HERE.parent)]

from common import BOS, CKPT, CONTAINER, build, windows  # noqa: E402
from domains import domains  # noqa: E402

MINLEN, MAXLEN = 6, 64
PBINS = 16


def match_predictions(data):
    """For each position t: predicted byte (-1 if none) and match length, from data[:t] only."""
    n = len(data)
    pred = np.full(n, -1, dtype=np.int64)
    length = np.zeros(n, dtype=np.int64)
    table, ptr, cur = {}, 0, 0
    for t in range(n):
        if cur > 0:
            pred[t], length[t] = data[ptr], cur
        # after coding data[t]: extend or drop the match, then index the window ending here
        if cur > 0:
            if data[ptr] == data[t]:
                cur, ptr = min(cur + 1, MAXLEN), ptr + 1
            else:
                cur = 0
        if t + 1 >= MINLEN:
            key = data[t + 1 - MINLEN:t + 1]
            if cur == 0 and key in table:
                ptr = table[key]
                back = 0
                while back < MAXLEN and ptr - 1 - back >= 0 and data[ptr - 1 - back] == data[t - back]:
                    back += 1
                cur = max(back, 1)
            table[key] = t + 1
    return pred, length


@torch.no_grad()
def model_probs(model, docs, preds, batch=32):
    """Per document: p_model(actual byte) and p_model(predicted byte) at every position."""
    items = [(i, w, k) for i, d in enumerate(docs) for k, w in enumerate(windows(d))]
    pa = [np.zeros(len(d)) for d in docs]
    pb = [np.zeros(len(d)) for d in docs]
    for start in range(0, len(items), batch):
        chunk = items[start:start + batch]
        t = max(len(w[1]) for _, w, _ in chunk)
        inp = torch.full((len(chunk), t), BOS, dtype=torch.long)
        for j, (_, (a, _, _), _) in enumerate(chunk):
            inp[j, :len(a)] = torch.from_numpy(a)
        probs = F.softmax(model(inp).float(), dim=-1).numpy()
        for j, (i, (_, b, frm), k) in enumerate(chunk):
            pos = np.arange(frm, len(b))
            doc_pos = k * 256 + pos  # window k starts at byte k*STRIDE; target j is byte start+j
            pa[i][doc_pos] = probs[j, pos, b[pos]]
            pr = preds[i][doc_pos]
            pb[i][doc_pos] = np.where(pr >= 0, probs[j, pos, np.maximum(pr, 0)], 0.0)
    return pa, pb


def mixed_bits(data, pred, length, pa, pb):
    hits = np.zeros((MAXLEN // 4 + 1, PBINS))
    tot = np.zeros_like(hits)
    bits = 0.0
    for t in range(len(data)):
        if pred[t] < 0:
            bits -= math.log2(max(pa[t], 1e-12))
            continue
        lb = min(length[t], MAXLEN) // 4
        pbin = min(PBINS - 1, int(pb[t] * PBINS))
        q = (hits[lb, pbin] + 0.4) / (tot[lb, pbin] + 0.8)
        if data[t] == pred[t]:
            bits -= math.log2(q)
        else:
            bits -= math.log2(1 - q) + math.log2(max(pa[t], 1e-12) / max(1 - pb[t], 1e-12))
        hits[lb, pbin] += data[t] == pred[t]
        tot[lb, pbin] += 1
    return bits


def main():
    torch.set_num_threads(4)
    model = build("20m")
    model.load_state_dict(torch.load(CKPT / "round2_20m_big.pt", map_location="cpu"))
    model.eval()
    sets = {f"{dom} {size}": group for (dom, size), group in domains().items()}
    from steno import split_books
    sets["books ≤500 KB (5 full)"] = [p.read_bytes() for p in split_books()[1] if p.stat().st_size <= 500_000]
    out = {}
    for name, docs in sets.items():
        preds = [match_predictions(d) for d in docs]
        pa, pb = model_probs(model, docs, [p for p, _ in preds])
        model_only = [-np.log2(np.maximum(a, 1e-12)).sum() for a in pa]
        mixed = [mixed_bits(d, p, l, a, b) for d, (p, l), a, b in zip(docs, preds, pa, pb)]
        raw = sum(map(len, docs))
        row = {"model": sum(math.ceil(b / 8) + CONTAINER for b in model_only) / len(docs),
               "model + match": sum(math.ceil(b / 8) + CONTAINER for b in mixed) / len(docs),
               "model bpc": sum(model_only) / raw, "model + match bpc": sum(mixed) / raw,
               "match coverage": float(np.mean(np.concatenate([p >= 0 for p, _ in preds])))}
        out[name] = row
        print(name, {k: round(v, 3) for k, v in row.items()}, flush=True)
    (HERE / "results" / "match_model.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
