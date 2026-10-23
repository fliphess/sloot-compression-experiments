# exp_tinymodel: could a small self-trained neural model beat the table? (E10, step 1)

**Question:** count tables can't generalise ("the cat sat" teaches nothing about "the dog sat"). Does a small neural model,
trained ourselves on the same 90 books and frozen (so no drift), do better? Measured as *ideal* bits, before investing in real files.

**Verdict: ⚠️ yes, convincingly; superseded by `round2/`**, which trains the same model on 3 GB and does integer inference.
`common.py` (the model definition and evaluation) is still used by `round2/`.

## Key result (ideal bits + 7 B container; trained on the same 85 MB as the v1 table)

| model | size (int8) | 1 KB | 30 KB | books bpc |
|---|---|---|---|---|
| 1M params | 0.9 MB | 305 B | 7,597 B | 2.170 |
| 5M params | 5 MB | 256 B | 7,285 B | 2.101 |
| **20M params** | **19 MB** | **238 B** | **6,722 B** | **1.899** |
| *for comparison: 3 GB table + adaptive* | *545 MB* | *239 B* | *6,672 B* | *1.755* |

A 19 MB model trained on 85 MB matches a 545 MB table built from 35× more text. Fake-int8 costs only +0.1-1.5%.
Model and table make different mistakes: switching between them per ~16 bytes gains 6-7% (`mix.py`).

## How it works

A byte-level GPT-style transformer (vocab 256 bytes + BOS, context 512), trained on the M4 GPU with PyTorch. Each document is
coded from an empty context; documents longer than 512 B use a sliding window.

## Run (needs the project venv with torch: see RESULTS.md)

```sh
nohup caffeinate -i bash lab/exp_tinymodel/run_all.sh > lab/exp_tinymodel/logs/queue.log 2>&1 &   # 1m, 5m, 20m (~4.5 h)
.venv/bin/python lab/exp_tinymodel/summarize.py
.venv/bin/python lab/exp_tinymodel/mix.py 1m 5m 20m
```

Files: `common.py` (model + evaluation), `train.py`, `table_bits.py` + `mix.py` (switching estimate), `summarize.py`, `results/`.
Details: [RESULTS.md](RESULTS.md).
