# round2: the neural model on 3 GB, mixing, integer inference (R2, E12)

**Question:** E10 showed a small model trained on 85 MB matches the best table. What happens with much more data? Does mixing
model and table still help? And can the model run with integers only, so files decode identically on any machine?

**Verdict: ✅ current best result.** The 20M model trained on the 3 GB corpus beats every table everywhere, even on whole books.
Integer-only inference works and costs almost nothing in compression. The new bottleneck is speed: numpy decodes at
0.1-0.3 KB/s, so the model needs a C/NEON port.

## Key result

| | 1 KB | 30 KB | books bpc | type |
|---|---|---|---|---|
| gzip | 588 B | 12,519 B | - | real |
| 3 GB table + adaptive (`overnight/`) | 239 B | 6,672 B | 1.755 | real |
| **20M model, 3 GB corpus** (`train_big.py`) | **207 B** | **5,711 B** | **1.562** | ideal (int8: +0.2%) |
| that model + table, switching (`mix_big.py`) | 204.5 B | 5,640 B | - | ideal |
| 20M model (E10), integer-only, **real files** (`intmodel/`) | 238.5 B | 6,729 B | - | real, 200/200 round trips |

The model's validation loss was still falling at the end of training (only ~10% of the 3 GB was seen), so a longer run is
in progress (`20m_long`, 54,000 steps). With this model, switching to the table gains only ~1%. Real mixing could gain more (the oracle ceiling is 185 B at 1 KB).

## Files

| file | what |
|---|---|
| `train_big.py` | trains `exp_tinymodel`'s model on the 3 GB tier (leakage-checked), same evaluation |
| `mix_big.py`, `run_mix.sh` | switching estimate: model vs any table + adaptive |
| [`intmodel/`](intmodel/) | E12: integer-only inference, frequency rule, real files, determinism checks (see its RESULTS.md) |
| `logs/`, `results/` | training curves and results |

## Run (needs the project venv with torch)

```sh
nohup caffeinate -i ./.venv/bin/python lab/round2/train_big.py 20m_big 18000 > lab/round2/logs/20m_big.log 2>&1 &   # ~7 h GPU
./.venv/bin/python lab/round2/mix_big.py tokens exp_table_3gb
./.venv/bin/python lab/round2/mix_big.py mix exp_table_3gb round2_20m_big.pt
./.venv/bin/python lab/round2/intmodel/export.py tables/round2_20m_big.pt 20m_big       # integer model file
./.venv/bin/python lab/round2/intmodel/nncodec.py compress tables/intmodel_20m_big.bin IN OUT
```
