# E12: integer-only inference + real files for the E10 byte transformer

**Status: done.** All 200 snippets × 1m / 5m / 20m / 20m_big compressed, decompressed and verified; determinism checks done.
(The agent that wrote this file was stopped for token budget before it filled in the results; the main session added them.)

## Files

| file | what |
|---|---|
| `intlm.py` | integer inference (numpy only, no torch), model-file loader, self-test |
| `export.py` | fp32 checkpoint → `tables/intmodel_NAME.bin` (build time, floats allowed) + `.build.json` (calibration stats) |
| `nncodec.py` | frequency rule, container, compress/decompress, CLI |
| `run_eval.py` | the 200-snippet round trips, per-doc records in `results/*.jsonl` |
| `summarize.py` | results table vs E10's fp32 / fake-int8 ideal numbers |
| `det.sh` | determinism checks (backends, thread settings, separate processes) |

Exported models (fingerprint = first 4 bytes of the SHA-256 of the file):

| model | file | bytes | fingerprint | f_res | f_qkv per layer |
|---|---|---|---|---|---|
| 1m | `tables/intmodel_1m.bin` | 1,271,117 | `728dc430` | 16 | 14 14 14 14 |
| 5m | `tables/intmodel_5m.bin` | 5,403,148 | `6a038f64` | 17 | 16 15 14 14 15 15 |
| 20m | `tables/intmodel_20m.bin` | 19,906,852 | `5a7f554f` | 16 | 15 14 14 15 15 15 |

(~262 KB of each file is the exp table; weights are int8, multipliers int32, shifts int8, biases int32.)

## Integer design

Everything in the decode path is integer arithmetic on int64 numpy arrays (int32 for the int8 × int8 products),
plus Python ints in the coder. Number formats (QF = integer with F fractional bits):

- **Residual stream**: Q`f_res` (per model, from calibration: max |x| × 8 < 2^24), clamped to |x| < 2^24 at the
  start of every block and before the final LayerNorm.
- **Embeddings**: int8 per row + per-row (multiplier, shift) to Q`f_res`; the integer tables are built at load time
  by integer shifts. The tied head uses the same int8 rows with a second per-row (multiplier, shift) to Q10 logits.
- **Linear layers** (qkv, proj, fc, out, head): weights int8 per output channel (as E10's fake int8). Input rows are
  quantised to int8 dynamically with their own row max m (`q = round(127 x / m)`, round half away from zero, all
  integer). `acc = q · Wq` exactly; `y = floor((acc · m · M[c] + 2^(S[c]-1)) / 2^S[c]) + bias[c]`, where the
  per-channel integer multiplier M[c] ∈ [2^14, 2^15] and shift S[c] encode `s_w[c] · 2^(F_out − F_in) / 127`.
- **LayerNorm** (Q12 out): mean = round(Σx / d); var = floor(Σ(x−mean)² / d) + eps; std = **exact integer sqrt**
  (`math.isqrt` on Python ints); n = round((x − mean) · 2^12 / std); out = round(n · γ_Q16 / 2^16) + β_Q12.
- **q, k, v**: Q`f_qkv[l]` (per layer, calibration max × 4 < 2^20), clamped to |x| < 2^20. KV cache holds these ints.
- **Attention softmax**: scores = q·k (exact int), converted to Q10 of score/√dh by an integer multiplier + shift.
  Per query row: subtract the row max, e = EXP[min(max − s, 32768)] » 14 (Q16; EXP is a stored table of
  round(2^30 · exp(−i/1024))), masked keys e = 0. Output = round(Σ e·v / Σ e) in Q`f_qkv` (the division is exact
  integer floor division with rounding; no probabilities are materialised).
- **GELU** (exact erf GELU): fc output in Q10, clamped to ±2^20; a stored table of round(1024 · gelu(x)) for x ∈ [−8, 8)
  in steps of 1/1024; identity above 8, 0 below −8.
- **Rounding/shifts**: all "shift right" operations are floor divisions by 2^S (Python semantics in numpy, defined
  for negative numbers); C's implementation-defined `>>` on negative ints is never used. Left shifts only on
  non-negative values.
- **Calibration** (build time only): 64 random 512-byte windows of the training stream (seed 0), fp32 torch, max |·|
  per site; the formats are powers of two with a 4-8× margin, and every site is clamped, so the bounds below hold for
  any input whatsoever.

### Overflow bounds (int64, by construction)

| site | bound |
|---|---|
| int8 × int8 dot, K ≤ 2048 | |acc| ≤ 2048·127² < 2^25 (fits int32) |
| linear rescale acc · m · M | 2^25 · 2^21 (inputs clamped) · 2^16 = 2^62 |
| LayerNorm Σ(x−mean)², d ≤ 2048 | 2^50 · 2^11 = 2^61; (x−mean)·2^12 < 2^37; n·γ < 2^17 · 2^24 |
| q·k, dh ≤ 64 | 64 · 2^40 = 2^46; × multiplier < 2^62 |
| Σ e·v over ≤ 512 keys | 2^9 · 2^16 · 2^20 = 2^45 |
| frequency rule e · 2^24 | 2^30 · 2^24 = 2^54; Σe < 257 · 2^30 |

numpy integer arithmetic wraps silently on overflow, so these bounds are the guarantee; the exporter asserts the
architecture limits (4d ≤ 2048, dh ≤ 64) and |γ|, |β| < 2^24 in fixed point.

### Matmul backends (same integers, verified)

- `int`: numpy integer matmul (int32 for the int8 linears, int64 for attention). **No BLAS, no floats.** Reference.
- `f32` (used for the timing runs): the int8 × int8 linears go through fp32 BLAS in K-chunks of ≤ 1024
  (1024 · 127² = 16,516,096 < 2^24, so every product and partial sum is an exactly representable integer, in any
  summation order, blocking, FMA or thread count); chunk results are summed in int64. Attention uses f64 BLAS
  (all terms < 2^46 < 2^53).
- `f64`: everything through f64 BLAS (terms < 2^53).

The float backends are accelerators only, valid on any IEEE-754 BLAS that really computes in fp32/fp64 (not e.g.
TF32 or bf16 GPU modes). To catch that, every model file carries a **self-test**: the SHA-256 of the integer logits
for a fixed 77-token input, computed with the `int` backend at export; the codec recomputes it on load with the
chosen backend and refuses to run on a mismatch.

## Frequency rule

From the Q10 integer logits l of the 257 symbols: e_i = EXP[min(max(l) − l_i, 32768)] (Q30), f_i = 1 + floor(e_i · 2^24 / Σe),
cumulative in symbol order. Every f ≥ 1, total ≤ 257 + 2^24 < 2^30 (codec v1's limit). BOS keeps f ≥ 1 but is never coded.
Cost vs the exact softmax of the same integer logits: +0.001 B per 512 bytes (2^20: +0.015 B, 2^16: +0.26 B, 2^12: +4.7 B; 5m, 10 docs).

## Context semantics

Exactly E10's evaluation: context reset per document; [BOS] + data in windows of 512 inputs every 256 positions,
positions restart at 0 per window; window 0 codes all its targets, later windows only their last 256. The encoder
runs each window as one block (512 rows). The decoder, per window, prefills the 257 already-known inputs
(positions 0..256, one block) and then decodes token by token with a KV cache of fixed size 512. Every integer
operation is row-wise (per-row int8 quantisation, LayerNorm per row, softmax per query row over its own causal keys),
so block and incremental evaluation give identical integers, which the round trips confirm.
Cost: every byte after the first window is computed twice (once as context), like the evaluation.

## File format

```
byte 0      mode: 0 = stored (raw bytes follow), 2 = integer byte transformer v1   (1 = steno v1, lab/codec.py)
bytes 1-4   model fingerprint (first 4 bytes of the SHA-256 of tables/intmodel_*.bin)
varint      number of bytes
rest        arithmetic code (lab/codec.py Encoder/Decoder: WNC 32-bit); missing trailing bytes read as zero
```
Empty input → `00`; if the coded file isn't smaller than stored, stored mode (+1 byte).

## Results

Output of `summarize.py` (real files; ideal = sum of -log2 p + 7 B; speed in bytes/s):

| model | size | docs | verified | **real B (avg)** | int-model ideal + 7 B | fp32 ideal + 7 B (E10) | fake-int8 ideal + 7 B | real vs fp32 | 1 stream enc / dec B/s | batched enc / dec B/s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1m | 1 KB | 50 | 50/50 | **309.3** | 309.3 | 305.3 | 309.3 | +1.30% | 2,368 / 620 (50) | - |
| 1m | 3 KB | 50 | 50/50 | **823.6** | 823.8 | 811.5 | 823.7 | +1.49% | 1,847 / 552 (50) | - |
| 1m | 10 KB | 50 | 50/50 | **2,617.2** | 2,617.2 | 2,576.5 | 2,616.9 | +1.58% | 1,962 / 653 (50) | - |
| 1m | 30 KB | 50 | 50/50 | **7,721.2** | 7,720.3 | 7,597.2 | 7,719.8 | +1.63% | 2,343 / 714 (4) | 2,431 / 1,745 (46) |
| 5m | 1 KB | 50 | 50/50 | **256.4** | 256.5 | 255.6 | 256.4 | +0.34% | 655 / 223 (2) | 1,116 / 869 (48) |
| 5m | 3 KB | 50 | 50/50 | **747.0** | 747.1 | 744.2 | 747.0 | +0.38% | - | 897 / 704 (50) |
| 5m | 10 KB | 50 | 50/50 | **2,455.8** | 2,455.8 | 2,445.7 | 2,455.6 | +0.41% | - | 773 / 660 (50) |
| 5m | 30 KB | 50 | 50/50 | **7,317.0** | 7,316.0 | 7,285.2 | 7,316.0 | +0.44% | - | 950 / 969 (50) |
| 20m | 1 KB | 50 | 50/50 | **238.5** | 238.5 | 238.3 | 238.5 | +0.07% | 552 / 110 (50) | - |
| 20m | 3 KB | 50 | 50/50 | **688.2** | 688.2 | 687.7 | 688.3 | +0.07% | 425 / 108 (50) | - |
| 20m | 10 KB | 50 | 50/50 | **2,261.8** | 2,261.9 | 2,259.9 | 2,262.1 | +0.09% | 540 / 129 (12) | 488 / 322 (38) |
| 20m | 30 KB | 50 | 50/50 | **6,729.2** | 6,728.3 | 6,722.0 | 6,728.3 | +0.11% | - | 478 / 329 (50) |
| 20m_big | 1 KB | 50 | 50/50 | **207.2** | 207.3 | 206.7 | 207.2 | +0.25% | 1,081 / 214 (50) | - |
| 20m_big | 3 KB | 50 | 50/50 | **590.0** | 590.1 | 588.8 | 590.1 | +0.21% | 550 / 129 (50) | - |
| 20m_big | 10 KB | 50 | 50/50 | **1,926.6** | 1,926.6 | 1,923.0 | 1,926.8 | +0.19% | 462 / 118 (50) | - |
| 20m_big | 30 KB | 50 | 50/50 | **5,722.3** | 5,721.3 | 5,711.1 | 5,721.5 | +0.20% | 437 / 119 (50) | - |

| run | docs | identical to main f32 run | round trips ok |
|---|---|---|---|
| 1m_f32_detdefault_s0of1 (threads default) | 3 | 3/3 | 3/3 |
| 1m_f64_detdefault_s0of1 (threads default) | 2 | 2/2 | 0/0 |
| 1m_int_det_s0of1 (threads 1) | 10 | 10/10 | 10/10 |
| 20m_f32_detdefault_s0of1 (threads default) | 3 | 3/3 | 3/3 |
| 20m_f64_det_s0of1 (threads 1) | 2 | 2/2 | 0/0 |
| 20m_f64_detdefault_s0of1 (threads default) | 2 | 2/2 | 0/0 |
| 20m_int_det_s0of1 (threads 1) | 1 | 1/1 | 1/1 |
| 5m_f32_det4_s0of1 (threads 4) | 1 | 1/1 | 1/1 |
| 5m_f32_detdefault_s0of1 (threads default) | 3 | 3/3 | 3/3 |
| 5m_f64_detdefault_s0of1 (threads default) | 2 | 2/2 | 0/0 |
| 5m_int_det_s0of1 (threads 1) | 3 | 3/3 | 3/3 |

## Commands

```sh
uv run python lab/round2/intmodel/export.py tables/exp_tinymodel_20m.pt 20m      # also works for round2_20m_big.pt
export VECLIB_MAXIMUM_THREADS=1
nohup uv run python lab/round2/intmodel/run_eval.py 20m f32 0 2 > lab/round2/intmodel/logs/20m_s0.log 2>&1 &
nohup uv run python lab/round2/intmodel/run_eval.py 20m f32 1 2 > lab/round2/intmodel/logs/20m_s1.log 2>&1 &
nohup bash -c "uv run python lab/round2/intmodel/run_eval.py 1m f32 0 1 && uv run python lab/round2/intmodel/run_eval.py 5m f32 0 1" > lab/round2/intmodel/logs/1m_5m.log 2>&1 &
nohup bash lab/round2/intmodel/det.sh > lab/round2/intmodel/logs/det.log 2>&1 &
uv run python lab/round2/intmodel/summarize.py
uv run python lab/round2/intmodel/nncodec.py compress tables/intmodel_5m.bin IN OUT [--backend int|f32|f64]
```
