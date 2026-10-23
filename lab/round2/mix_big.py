"""Round 2: switching estimate between a neural model and any table + E8b adaptive layer.

Same method as lab/exp_tinymodel/mix.py (E10): per segment of >= S bytes the encoder picks
the cheaper of table and model and codes that choice with an adaptive binary flag. Achievable,
ideal bits + 7 B container. "oracle_per_token" (free per-token choice) is not achievable.

  tokens  TABLE            per-token ideal bits of TABLE + d3-sum-w32 on the 200 snippets (CPU),
                           cached in results/table_tokens_TABLE.json
  mix     TABLE CKPT...    switching estimate for each model checkpoint (GPU); CKPT is a file
                           under tables/, e.g. exp_tinymodel_20m.pt or round2_20m_big.pt

Usage: .venv/bin/python lab/round2/mix_big.py tokens exp_table_3gb
       .venv/bin/python lab/round2/mix_big.py mix exp_table_3gb exp_tinymodel_20m.pt
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
sys.path[:0] = [str(LAB / "exp_tinymodel"), str(LAB), str(LAB / "exp_adaptive")]

RESULTS = HERE / "results"


def tokens(table_name):
    import run as e8b
    from codec import FrozenTable
    from short_docs import SIZES, snippets
    from steno import split_books
    from table_bits import token_bits

    table = FrozenTable.load(LAB.parent / "tables" / f"{table_name}.npz")
    cfg = e8b.make_configs()["d3-sum-w32"]
    texts = [p.read_text(encoding="utf-8") for p in split_books()[1]]
    res = {}
    for size in SIZES:
        res[str(size)] = [token_bits(table, cfg, s) for text in texts for s in snippets(text, size)]
        print(size, f"{sum(d[-1][1] for d in res[str(size)]) / 8 / 50:.0f} B ideal avg", flush=True)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"table_tokens_{table_name}.json").write_text(json.dumps(res))


def mix(table_name, ckpts):
    import torch
    from common import CKPT, CONTAINER, build, byte_bits, test_docs
    from mix import SEGMENTS, switch_bits

    table = json.loads((RESULTS / f"table_tokens_{table_name}.json").read_text())
    docs = test_docs()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    for ckpt in ckpts:
        arch = "20m" if "20m" in ckpt else "5m" if "5m" in ckpt else "1m"
        model = build(arch).to(device)
        model.load_state_dict(torch.load(CKPT / ckpt, map_location=device))
        res = {}
        for size, toks in table.items():
            cums = [np.concatenate(([0.0], np.cumsum(pb))) for pb in byte_bits(model, docs[size], device)]
            row = {"table": np.mean([math.ceil(t[-1][1] / 8) + CONTAINER for t in toks]),
                   "model": np.mean([math.ceil(c[-1] / 8) + CONTAINER for c in cums])}
            for seg in SEGMENTS:
                row[f"switch_{seg}"] = np.mean([math.ceil((switch_bits(t, c, seg) + 1) / 8) + CONTAINER
                                                for t, c in zip(toks, cums)])
            row["oracle_per_token"] = np.mean([math.ceil(sum(min(t[k][1] - (t[k - 1][1] if k else 0),
                                                                 c[t[k][0]] - c[t[k - 1][0] if k else 0])
                                                             for k in range(len(t))) / 8) + CONTAINER
                                               for t, c in zip(toks, cums)])
            res[size] = {k: round(float(v), 1) for k, v in row.items()}
            print(ckpt, size, res[size], flush=True)
        (RESULTS / f"mix_{table_name}__{Path(ckpt).stem}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "tokens":
        tokens(sys.argv[2])
    elif sys.argv[1] == "mix":
        mix(sys.argv[2], sys.argv[3:])
    else:
        sys.exit(__doc__)
