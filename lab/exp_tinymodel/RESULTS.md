# E10 step 1: tiny self-built model, measuring the potential

**Status: overnight sweep running** (started 2026-10-22 23:06, detached via `nohup caffeinate -i`).
Expected finish: 1M ≈ 00:15, 5M ≈ 02:20, 20M ≈ 04:50 (2026-10-23). The table below fills in per model:
run `python3 lab/exp_tinymodel/summarize.py` after a model finishes.

## Question

How much could a small neural model we train ourselves beat our n-gram table, before investing in integer-only inference?
These are **ideal code lengths** (sum of -log2 p), not real compressed files. A real arithmetic coder adds well under 1%
(E7: real ≈ ideal + container), and short-document sizes add the same 7 B container as codec v1.

## Setup

- **Data:** trained only on the same 90 training books as the tables (`steno.split_books()[0]`, 84.7 MB, concatenated).
  The last 1 MB of that stream is held out for the validation curve. Test = the shared test set: 200 short snippets
  (`short_docs`, 50 each of 1/3/10/30 KB, each coded independently) and the 10 full test books.
- **Model:** byte-level GPT-style transformer. Vocab = 256 bytes + BOS (no tokenizer, no OOV), learned positions,
  pre-LayerNorm blocks, GELU MLP (4×), output head tied to the input embedding, context 512 bytes.
  GPT-2 style init (std 0.02; residual projections 0.02/sqrt(2L)).

  | name | d_model | layers | heads | params |
  |---|---|---|---|---|
  | 1m | 128 | 4 | 4 | 0.89M |
  | 5m | 256 | 6 | 8 | 4.94M |
  | 20m | 512 | 6 | 8 | 19.31M |

- **Training:** AdamW (β 0.9/0.95, weight decay 0.1 on matrices), grad clip 1.0, 300 warmup steps, cosine to 10% of peak.
  Fixed step counts (reproducible apart from GPU nondeterminism), seed 0:
  - 1m: 8,000 steps × 64 × 512, peak lr 2e-3 (3.1 epochs);
  - 5m: 4,500 × 64 × 512, 1e-3 (1.8 epochs);
  - 20m: 5,000 × 32 × 512, 6e-4 (1.0 epoch).

  Budget ≈ 1-2 h each on the M4 GPU (MPS). Measured throughput: ~80K / 24K / 12K bytes/s (bf16 autocast gave no speed-up).
- **Evaluation:** context reset per document; documents longer than 512 B use a sliding window (stride 256), so every byte
  after the first window sees ≥ 256 B of context. Coverage was verified: each byte is scored exactly once.
- **int8:** fake quantisation. Weights are symmetric int8 per output channel (embeddings/tied head per row); activations
  entering every Linear and the head are symmetric int8 per token row. LayerNorm, softmax and GELU stay in float in this
  simulation. Real integer-only inference needs integer versions of those (I-BERT style), which isn't measured here.
- **CPU cost:** analytic FLOPs per byte at full context with a KV cache (2 × multiply-adds).
- **Versions:** Python 3.13.16 (uv venv `.venv`, gitignored), torch 2.14.1 (MPS), numpy 2.5.4.

## Results

<!-- results:start -->
| model | params | int8 size | epochs | train min | 1 KB | 3 KB | 10 KB | 30 KB | books bpc | int8 books bpc | MFLOP/byte |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 static n-gram (E7) | - | 31 MB table | - | - | 285 B | 837 B | 2,756 B | 8,207 B | 2.420 | - | - |
| E8a static combo | - | 31 MB table | - | - | 270 B | - | - | - | 2.289 | - | - |
| E8b adaptive n-gram | - | 31 MB table | - | - | 276 B | 793 B | 2,552 B | 7,484 B | 1.911 | - | - |
| byte transformer 1m | 0.89M | 0.9 MB | 3.13 | 57 | 305 B (2.371) | 812 B (2.132) | 2,576 B (2.044) | 7,597 B (2.012) | **2.170** | 2.203 | 2.7 |
| byte transformer 5m | 4.94M | 5.0 MB | 1.76 | 95 | 256 B (1.975) | 744 B (1.953) | 2,446 B (1.940) | 7,285 B (1.929) | **2.101** | 2.109 | 12.7 |
| byte transformer 20m | 19.31M | 19.4 MB | 0.98 | 108 | 238 B (1.837) | 688 B (1.803) | 2,260 B (1.792) | 6,722 B (1.780) | **1.899** | 1.901 | 44.3 |

Short docs: average bytes per document = ceil(ideal bits / 8) + 7 B container, with bpc in brackets. Model rows are **ideal code lengths, not real files**; baselines are real files.

- 1m: fake int8 vs fp32: 1000: +1.35%, 3000: +1.53%, 10000: +1.58%, 30000: +1.62%, books: +1.49%
- 5m: fake int8 vs fp32: 1000: +0.35%, 3000: +0.40%, 10000: +0.41%, 30000: +0.42%, books: +0.40%
- 20m: fake int8 vs fp32: 1000: +0.10%, 3000: +0.10%, 10000: +0.10%, 30000: +0.09%, books: +0.11%
<!-- results:end -->

## Not done (by design or time)

- Bigger-data run on E8c's corpus: skipped, because E8c hadn't finished when the sweep was queued.
- Mixing the model with the n-gram table: not estimated (token-level table vs byte-level model needs a projection of
  token probabilities to bytes; out of scope for step 1).

## Reproduce

```sh
uv venv .venv --python 3.13 && uv pip install --python .venv/bin/python torch numpy
nohup caffeinate -i bash lab/exp_tinymodel/run_all.sh > lab/exp_tinymodel/logs/queue.log 2>&1 &
python3 lab/exp_tinymodel/summarize.py     # after (each) model finishes
```

Logs: `lab/exp_tinymodel/logs/{queue,1m,5m,20m}.log`. Per-model JSON (curves, per-document bits): `lab/exp_tinymodel/results/`.
Checkpoints (fp32 state dicts): `tables/exp_tinymodel_{1m,5m,20m}.pt`. `run_all.sh` skips models whose JSON exists, so it can be re-run to resume.
