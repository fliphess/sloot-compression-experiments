"""Achievable switching estimate: n-gram table (E8b d3-sum-w32) vs neural model, per segment.

The text is cut into segments ending at the first token boundary >= S bytes after
the previous one. Per segment the encoder picks the model with fewer bits and
codes that choice with an adaptive binary flag (KT estimator). Both models keep
seeing the full history, so a decoder can follow. This is a lower bound on what
real mixing (cmix-style) would give. "oracle per token" (free choice per token,
no flag cost) is NOT achievable and only shows the ceiling of complementarity.

Usage: .venv/bin/python lab/exp_tinymodel/mix.py NAME [NAME ...]
"""
import json
import math
import sys

import numpy as np
import torch

from common import CKPT, CONTAINER, build, byte_bits, test_docs
from train import RESULTS

SEGMENTS = [16, 64, 256, 1024]


def switch_bits(tok, model_cum, seg):
    """tok: [[byte_end, table_cum_bits], ...]; model_cum[i] = model bits for bytes [0, i)."""
    total, last_b, last_t, n, ones, ends = 0.0, 0, 0.0, 0, 0, []
    for k, (b, t) in enumerate(tok):
        if b - last_b >= seg or k == len(tok) - 1:
            ends.append((b, t))
            last_b = b
    last_b, last_t = 0, 0.0
    for b, t in ends:
        cost_t, cost_m = t - last_t, model_cum[b] - model_cum[last_b]
        choice = cost_m < cost_t
        p = ((ones if choice else n - ones) + 0.5) / (n + 1)
        total += min(cost_t, cost_m) - math.log2(p)
        ones += choice
        n += 1
        last_b, last_t = b, t
    return total


def main():
    table = json.loads((RESULTS / "table_tokens.json").read_text())
    docs = test_docs()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    out = {}
    for name in sys.argv[1:]:
        model = build(name).to(device)
        model.load_state_dict(torch.load(CKPT / f"exp_tinymodel_{name}.pt", map_location=device))
        res = {}
        for size, toks in table.items():
            per_byte = byte_bits(model, docs[size], device)
            cums = [np.concatenate(([0.0], np.cumsum(pb))) for pb in per_byte]
            row = {"table": np.mean([math.ceil(t[-1][1] / 8) + CONTAINER for t in toks]),
                   "model": np.mean([math.ceil(c[-1] / 8) + CONTAINER for c in cums]),
                   "best_of_doc": np.mean([math.ceil((min(t[-1][1], c[-1]) + 1) / 8) + CONTAINER for t, c in zip(toks, cums)])}
            for seg in SEGMENTS:
                row[f"switch_{seg}"] = np.mean([math.ceil((switch_bits(t, c, seg) + 1) / 8) + CONTAINER
                                                for t, c in zip(toks, cums)])
            row["oracle_per_token"] = np.mean([math.ceil(sum(min(t[k][1] - (t[k - 1][1] if k else 0),
                                                                 c[t[k][0]] - c[t[k - 1][0] if k else 0])
                                                             for k in range(len(t))) / 8) + CONTAINER
                                               for t, c in zip(toks, cums)])
            res[size] = {k: round(float(v), 1) for k, v in row.items()}
            print(name, size, res[size], flush=True)
        out[name] = res
    (RESULTS / "mix.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
