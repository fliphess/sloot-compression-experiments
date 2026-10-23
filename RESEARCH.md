# Jan Sloot revisited: research log

Research into a modern, honest version of the
[Sloot Digital Coding System](https://en.wikipedia.org/wiki/Sloot_Digital_Coding_System):
very high compression for long-term archival storage (eventually on paper), driven by a
large shared reference table. Started 2026-10-22. The original brief is in `prompt.txt`.

## Handover: read this first

- This file is the source of truth: goal, decisions, every experiment with results, and what's next.
  Update it after every experiment: append to the log, don't rewrite history.
- All code is in `lab/` (Python 3 + numpy, run from the project root); `lab/README.md` maps every script and experiment directory, and each directory has its own README (question, verdict, key result, how to run). Data is in `corpus/`, frozen tables in `tables/`.
- Running experiments write their own results to `lab/exp_*/RESULTS.md`; merge their conclusions here.
- The user wants to be part of the decisions: measure first, present numbers, and pick together.
  Don't publish anything externally (no claude.ai docs, Drive, etc.); everything stays on disk.

## Goal

- A storage format for **archival data** (text first; later "already compressed data and
  anything people want to keep forever") that compresses far beyond today's tools.
- Mechanism: **compression + reference mechanism + (later) a printable paper standard.**
- The "magic" sits in **one big shared reference table** ("one table to rule them all"),
  shipped separately once. A 10 GB table is fine: it targets TB-scale archives, where the
  savings dwarf the table size.
- Final shipping medium: paper (QR-like, printable, camera/scanner-readable). Deliberately
  postponed: first data + reference, then the shipping constraints follow.

## Decisions and constraints (from the user)

| date | decision | why |
|---|---|---|
| 2026-10-22 | **No LLMs in the decompression path** | hosted models are retuned weekly; a tiny change turns archives into useless blobs. Must be fully deterministic and reproducible forever. |
| 2026-10-22 | No images/video | big companies with many PhDs already work on that; focus on archival data (text, compressed data, arbitrary files). |
| 2026-10-22 | **No negative compression** | otherwise base64 would do. Implemented: worst case = input + 1 byte (stored mode). |
| 2026-10-22 | Python first; describe the algorithm so it can be ported later | any language can do the math. |
| 2026-10-22 | Focus on natural language first, per-language tables ("steno in overdrive") | official language; slang etc. handled losslessly via escapes. |
| 2026-10-22 | Postpone printer/camera and error correction | the hardware keeps improving; ECC complicates things while we're getting our feet wet. |
| 2026-10-22 | One frozen preset table, shipped separately, may be ~10 GB | shipping the table with each file would defeat the purpose. |
| 2026-10-22 | Run the next 3 experiments in parallel, measure, then pick the best | |
| 2026-10-22 | **The deliverable is the method to build the table, not one specific table** | small source differences are fine as long as the method is sound. Each built table instance is identified by its fingerprint; an archive keeps the instance it was encoded with. |

## Key insights so far

1. **Sloot's claim is impossible for arbitrary data** (pigeonhole/Shannon): a table with
   "all possible options" gains nothing, because the reference into it is as long as the data.
   It only works for **likely** data: the knowledge has to sit in the shared table.
2. **Sloot's demo is reproducible as a pointer:** data that's already *in* the reference
   compresses to almost nothing (1 MB → ~125 bytes, E4). Probably what his demo was.
3. **For new data the gain comes from statistics, not copies.** zstd using a 90 MB reference
   as a copy source saved ~21%. A frozen word n-gram table (statistics) does much better on the same data.
4. **Reference tables matter most for short documents**: classic compressors must learn
   everything from the document itself; the table brings the knowledge along (E6).
5. Recursive pattern replacement (the user's first idea = Re-Pair/BPE grammar compression)
   converges: the optimum is early (~round 87), and the table + references end up costing more than gzip (E2).
6. The table is frozen and identified by a 4-byte fingerprint; files made with another table
   are refused instead of decoding to garbage.

## Experiment log

Metric: **bpc = bits per input byte** (8 = no gain). Lower is better.

### E1: classic compressors (`lab/baselines.py`)
| sample | gzip -9 | bzip2 -9 | xz -9e | zstd -22 | brotli -q11 |
|---|---|---|---|---|---|
| enwik8 first 10 MB | 2.951 | 2.333 | 2.176 | 2.235 | 2.164 |
| enwik8 first 1 MB | 2.846 | 2.251 | 2.326 | 2.401 | 2.248 |
| Python source 1 MB (CPython 3.15.0 tarball) | 1.879 | 1.567 | 1.577 | 1.608 | 1.551 |
| gzipped text / random 1 MB | 8.00-8.05 (negative compression) | | | | |

### E2: recursive pattern replacement, v0 (`lab/repair.py`)
Re-Pair style: each round replaces the most frequent symbol pairs with new table entries, until no pair occurs twice.
- enwik 1 MB: best **3.164 bpc at round 87** (26,848 rules; table 87 KB + sequence 308 KB), adaptive order-0 coding.
  Ends after ~4,400 rounds / 46K rules, at a worse 3.287 bpc. With fixed-width references: 3.749 best.
- Python 1 MB (CPython 3.15.0 tarball sample): best 2.434 bpc (round 73).
- Conclusion: loses to gzip (2.85 / 1.86). Most cost is the sequence coded without context, not the table.

### E3: paper capacity estimate (`lab/paper.py`); postponed topic
A4, 10 mm margins, monochrome, 6% structure + 25% ECC: 600 dpi scan ≈ 288 KB/page;
1200 dpi ≈ 650 KB; phone 12 MP whole page ≈ 53 KB; 48 MP ≈ 160 KB. QR v40 = 2,953 bytes.
Not yet validated on real prints.

### E4: shared reference via zstd --patch-from (`lab/reference.py`)
Reference = first N MB of enwik8; targets 1 MB.
| target | 0 MB | 1 MB | 10 MB | 30 MB | 90 MB |
|---|---|---|---|---|---|
| wiki held-out | 2.468 | 2.307 | 2.133 | 2.044 | 1.953 |
| wiki already in reference | 2.517 | 2.366 | 2.159 | 2.065 | **0.001** |
| python source* | 1.601 | 1.567 | 1.552 | 1.543 | 1.535 |

\* measured on the earlier, non-reproducible local-stdlib Python sample (see Reproducibility).

The gain flattens (roughly logarithmic in reference size); none outside the reference's domain.

### E5: steno in overdrive, estimated (`lab/steno.py`)
Corpus: top-100 English Project Gutenberg books (`lab/fetch_gutenberg.py`, 93 MB). Test =
every 10th book by ID (10, 46, 145, 408, 1232, 1952, 2701, 5197, 20203, 37683); train = the other 90 (84M chars).
Tokens = word with leading space, or a single other character. Table = 136,525 words (count ≥ 2) + uni/bi/trigram counts, Witten-Bell interpolation, escapes spelled with an order-1 letter model.
Results over the 10 full test books (12.2 MB):

| xz -9e | zstd -19 | zstd + 84 MB ref | static o0 | static o1 | static o2 | adaptive o0 | adaptive o1 | **adaptive o2** |
|---|---|---|---|---|---|---|---|---|
| 2.306 | 2.342 | 2.249 | 2.834 | 2.471 | 2.464 | 2.716 | 2.164 | **2.005** |

Adaptive = a per-document layer on top of the frozen table, rebuilt identically by the decoder (nothing extra stored).
It wins 9/10 books. It loses on 37683, an etymological dictionary (4% unknown tokens). The gain is largest on short books (32 KB: xz 3.10 → 1.84).

### E6: short documents, estimated (`lab/short_docs.py`)
200 snippets (5 per test book per size, starting at a paragraph), each compressed alone; real container bytes for the tools.

| size | gzip | xz raw | zstd | brotli | zstd + 110 KB dict | zstd + 4 MB dict | steno static | steno adaptive | adaptive beats all |
|---|---|---|---|---|---|---|---|---|---|
| 1 KB | 588 B | 628 | 580 | 471 | 414 | 389 | 287 | **282** | 47/50 |
| 3 KB | 1,525 | 1,568 | 1,500 | 1,237 | 1,150 | 1,058 | 846 | **822** | 46/50 |
| 10 KB | 4,534 | 4,445 | 4,382 | 3,749 | 3,606 | 3,296 | 2,795 | **2,668** | 44/50 |
| 30 KB | 12,519 | 11,886 | 11,953 | 10,587 | 10,384 | 9,498 | 8,328 | **7,833** | 44/50 |

Steno ≈ 2.1-2.3 bpc regardless of length; static ≈ adaptive for short documents.

### E7: real codec v1 (`lab/codec.py`), the current baseline
Static frozen table `tables/en-gutenberg-v1.npz` (31 MB, fingerprint `1cd43ea2`), PPM method C
with full exclusion (orders 2 → 1 → 0), 32-bit integer arithmetic coder (Witten-Neal-Cleary),
container: mode byte + fingerprint + varint token count + code. Format spec is in the docstring of `lab/codec.py`.

| size | estimate (E6 static) | PPM ideal | **real file** | real bpc |
|---|---|---|---|---|
| 1 KB | 287 B | 278 | **285** | 2.271 |
| 3 KB | 846 | 830 | **837** | 2.220 |
| 10 KB | 2,795 | 2,749 | **2,756** | 2.192 |
| 30 KB | 8,328 | 8,200 | **8,207** | 2.176 |

All 200 round trips are byte-identical. Container overhead ≈ 7 bytes/file. Edge cases (empty,
binary, random, short Unicode) fall back to stored mode at +1 byte. Book 1952: 31,904 → 7,318 bytes,
verified via the CLI in separate processes. Speed ≈ 1 s per 30 KB (encode + decode, Python).
Full books, real bytes (`lab/eval_books.py`, output in `lab/results_v1_books.txt`; books ≤ 500 KB decode-verified):

| book | 10 | 46 | 145 | 408 | 1232 | 1952 | 2701 | 5197 | 20203 | 37683 | **total** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 static bpc | 2.250 | 2.017 | 1.853 | 2.020 | 1.876 | 1.835 | 2.155 | 1.835 | 2.024 | 3.943 | **2.420** |
| xz -9e (E5) | 1.908 | 2.878 | 2.400 | 2.714 | 2.521 | 3.102 | 2.653 | 2.393 | 2.639 | 2.543 | 2.306 |

v1 static beats xz on 9/10 books (2.08 vs 2.25 bpc without 37683). It loses heavily on the dictionary
(37683: 3.94; it's 30% of the output on its own) and on the Bible (10: long and repetitive, which favours
adaptive/LZ). So the full-book total (2.42) is worse than xz; this is exactly what E8b (adaptive) should fix.

## Running: E8 three parallel experiments (started 2026-10-22)

All measured against codec v1 (E7) on the same test set: the 200 short snippets (`short_docs.snippets`) and the 10 full test books.
Each lives in its own directory and must not modify shared files (`lab/codec.py`, `lab/steno.py`, `lab/short_docs.py`, `tables/en-gutenberg-v1.npz`, `corpus/gutenberg/`).

| exp | directory | question | status |
|---|---|---|---|
| E8a | `lab/exp_model/` | model improvements: case as a separate flag, line-break/column model, order-3/4 contexts, better escape estimation (PPMD/X); each alone, then combined | **done**, see E8a below |
| E8b | `lab/exp_adaptive/` | per-document adaptive layer inside the real codec (integer, deterministic) | **done**, see E8b below |

Note: all agents were stopped by the usage limit at ~23:20; their detached jobs finished on their own (E8c 3 GB, E9 verification) or are still running (E10 20M).
| E8c | `lab/exp_table/` | larger tables: 0.3 / 1 / 3+ GB of training text (more Gutenberg, no test-book duplicates), scaling curve vs table size on disk | **done**, see E8c below |

### E8a result: model improvements (`lab/exp_model/`, details + port-ready algorithm in its RESULTS.md)
Static mode, same 90 training books, real bytes. All 200 snippets were round-trip verified; books ≤ 500 KB were decoded, the larger ones only encoded.

| variant | short docs vs v1 | full books bpc | table |
|---|---|---|---|
| v1 (reproduced exactly) | - | 2.420 | 36.2 MB* |
| **wrap**: line-break flag instead of "\n" tokens | **-3.9%** | 2.329 | 34.7 MB |
| case / case2: lowercase word + case class (case2: also conditioned on the previous word's class) | -0.6 / -0.7% | 2.409 / 2.408 | ~35 MB |
| sp2: order-2 spelling model | -0.7% | 2.398 | +48 KB |
| S: escape rates measured on held-out training books | -0.5% | 2.402 | +8 KB |
| D / X: PPMD / PPMX escapes | +0.7 / +0.5% (worse) | 2.442 / 2.438 | |
| o3 / o4: 3- and 4-token contexts | +1.7 / +2.9% (worse) | 2.462 / 2.489 | 99 / 190 MB |
| **wrap+case2+sp2+S** | **-5.4%** (1 KB: 285 → 270 B) | **2.289** | 33.1 MB |

\* E8a's table files store extra arrays; the v1 table itself is 31 MB. The compression results are identical.

- Wrap newlines were 7.6% of all tokens and broke word contexts. **But that gain is mostly Gutenberg-specific:**
  a main-session check on 15 × 3 KB snippets with paragraphs unwrapped gave -1.5% vs v1 (vs -6.4% hard-wrapped).
  It doesn't hurt on normal text.
- Longer contexts are worse with 84 MB of training text, even with pruning and measured escapes. Re-test on E8c's bigger tables.
- Bits in the combination: words predicted from context 66%, the single-word fallback 16%, escapes 8%,
  spelling 6.5%, case 2%. Little room is left in the static model itself; bigger levers are adaptivity (E8b) and table size (E8c).
- Spot check (separate process): book 1952 → 7,007 B (v1 7,318), book 46 → 39,420 B (v1 39,933), round trips OK.

### E8c result: larger tables (`lab/exp_table/`, details in RESULTS.md + results/summary.md)
8,410 extra English Gutenberg books (3.0 GB) from the pglaf mirror. Leakage control (content shingles, excerpts,
title/author) excluded 85 books (mostly Bible editions). Codec v1 algorithm unchanged; real bytes; all 200 snippets
round-trip verified per table; books ≤ 500 KB decode-verified. The chunked builder reproduces v1 bit for bit and peaks at 2.5-4.6 GB.

| table | train text | disk | vocab | 1 KB | 3 KB | 10 KB | 30 KB | books bpc |
|---|---|---|---|---|---|---|---|---|
| v1 | 85 MB | 31 MB | 137K | 285 B | 837 | 2,756 | 8,207 | 2.420 |
| 300 MB | 300 MB | 94 MB | 299K | 262 | 769 | 2,534 | 7,527 | 2.209 |
| 1 GB | 1 GB | 240 MB | 573K | 249 | 735 | 2,424 | 7,204 | 2.095 |
| **3 GB** | 3 GB | 545 MB | 1.06M | **242** | **714** | **2,354** | **6,992** | **2.019** |
| 300 MB pruned (≥2) | 300 MB | 26 MB | | 269 | 789 | 2,603 | 7,735 | 2.275 |
| 1 GB pruned (≥2) | 1 GB | 68 MB | | 255 | 753 | 2,487 | 7,385 | 2.154 |
| 3 GB pruned (≥2) | 3 GB | 162 MB | | 247 | | 2,404 | 7,134 | 2.066 |

- Diminishing returns: ×3.5 text −8%, then ×3.3 −5%, then ×3 −3%. The 4-point fit value = A + B·MB^-α gives an asymptote of
  ~231 B for 1 KB docs (−19% vs v1), and 10 GB of text reaches ~238 B. **With 2 words of context, a 10 GB table can't be
  used well.** Using it needs longer contexts (E8a showed they need more data), more varied text, or a model that generalises.
- **Pruning** (drop n-grams seen once) removes ~73% of the table for +2.5-3% bytes. Per MB of table, more text +
  pruning beats less text kept whole (1 GB pruned 68 MB > 300 MB whole 94 MB).
- Disk grows ≈ MB^0.8 of text; a 10 GB table would need ~110 GB of text, more English than Gutenberg has (~20 GB).

### Overnight combination: larger tables × E8b adaptive (`lab/overnight/combo.py`, real bytes, all verified)
| table | 1 KB | 3 KB | 10 KB | 30 KB | books bpc |
|---|---|---|---|---|---|
| v1 + adaptive (= E8b) | 276 B | 793 | 2,552 | 7,484 | 1.911 |
| 300 MB pruned (26 MB) + adaptive | 263 | 759 | 2,462 | 7,240 | 1.884 |
| 300 MB + adaptive | 256 | 739 | 2,396 | 7,044 | 1.825 |
| 1 GB pruned (68 MB) + adaptive | 251 | 729 | 2,373 | 6,990 | 1.834 |
| 1 GB + adaptive | 245 | 711 | 2,314 | 6,822 | **1.784** |
| 3 GB pruned (162 MB) + adaptive | 244 | 708 | 2,308 | 6,807 | 1.798 |
| **3 GB + adaptive** | **239** | **694** | **2,260** | **6,672** | **1.755** |

- **The gains stack.** 3 GB + adaptive at 1 KB: 239 B, −16% vs v1 and **2.5× smaller than gzip (588 B)**.
  On books, 1 GB + adaptive (1.784) is 23% below xz (2.306).
- **Weights are robust (closes E8b's caveat):** on 10 fresh validation books (never in v1 training or test), d3-sum-w8/w32/w64
  differ by < 0.5% (1 KB: all 260 B; 30 KB: 7,403 / 7,377 / 7,369 B). The adaptive gain over static on fresh data
  is −6.5% (1 KB) to −9.5% (30 KB). On the 1 GB table, w8/w32/w64 are also flat (30 KB: 6,839 / 6,822 / 6,814 B).

### E8b result: adaptive layer in the real codec (`lab/exp_adaptive/`, details in its RESULTS.md)
Recommended config **`d3-sum-w32`**:
- first a document-only context of the last 3 tokens;
- then table orders 2 → 1 → 0, each with the document's own counts added at weight 32;
- tokens not in the table get a new id after their first spelling.

The frozen table is never modified; the decoder rebuilds the layer, so nothing extra is stored.

| test | v1 static | **d3-sum-w32** | gain |
|---|---|---|---|
| 10 full books | 2.420 bpc | **1.911 bpc** | −21% (xz 2.306, zstd + 84 MB ref 2.249) |
| 1 KB snippets | 285 B | 276 B | −3% |
| 3 KB | 837 B | 793 B | −5% |
| 10 KB | 2,756 B | 2,552 B | −7% |
| 30 KB | 8,207 B | 7,484 B | −9% |

- Beats xz on 9/10 books; the etymological dictionary goes from 3.94 to 2.55 bpc (xz 2.54).
- 6/200 snippets come out 1-2 B larger than static; an encoder trying both modes (mode byte) makes it never worse.
- All 200 snippets + all 10 books (incl. the 4.3 MB Bible) are round-trip verified. Main-session spot check, in a
  separate process, agrees (book 1952: 7,208 B vs v1 7,318; book 46: 36,911 vs 39,933).
- Learned: document counts need a large weight (16-64; 1 is far too weak). Document-only contexts in *front* of
  table orders 1-2 hurt, but a document-only order 3 helps. New-word ids are worth ~1.6% at 30 KB; adapting the
  spelling model barely matters.
- Caveats: the weights were chosen on the test set (coarse powers of 2, neighbours within 2%); there's no count
  rescaling yet (limit ~30M tokens per document); 1.6× slower than static in Python; the best weight may shift
  with E8a/E8c.

## Overnight 2026-10-22 → 23: what's running and where to look

| job | what | where |
|---|---|---|
| E8c (agent) | table scaling 84 MB → 300 MB → 1 GB (+ pruned) | `lab/exp_table/RESULTS.md` |
| E9 (agent) | bootstrap chain + METHOD.md | `lab/exp_chain/RESULTS.md`, `METHOD.md` |
| E10 (agent, GPU) | tiny models 1/5/20M params, ideal bpc | `lab/exp_tinymodel/RESULTS.md` |
| overnight worker 1 | E8b adaptive (d3-sum-w32) × tables 1 GB, 300 MB | `lab/overnight/logs/worker1.log` |
| overnight worker 2 | same × 1 GB pruned, 300 MB pruned, v1 (sanity check vs E8b) | `logs/worker2.log` |
| overnight worker 3 | **validation of E8b weights on 10 fresh books** (v1 table; static, d3-sum-w8/32/64, sum-w32) | `logs/worker3.log` |
| overnight worker 4 | weight tuning on the 1 GB table (d3-sum-w8, w64; w32 is in worker 1) | `logs/worker4.log` |
| worker 5 (started 02:25) | E8b adaptive × the 3 GB table | `logs/worker5.log` |
| worker 6 (started 02:25) | build pruned 3 GB table → static eval (E8c's evaluate.py) → adaptive | `logs/worker6.log` (ends with WORKER6_DONE) |
| E10 queue (no agent) | 20M training (ETA ~03:45), then `mix.py` (model + table mixing estimate) | `lab/exp_tinymodel/logs/`, `summarize.py` |

Status at 02:30: workers 1-4 done, E8c's 3 GB run done, E9 verification done. **07:30: everything done**: workers 5-6, E10 20M + mix (merged above).

Workers are plain detached processes (no agent needed): `lab/overnight/combo.py`. Check with
`pgrep -fl overnight/combo.py` and `python3 lab/overnight/combo.py summary`. If the agents stopped (session limit), their
RESULTS.md files say which commands were left running or still have to run.
Validation-set caveat: the 10 fresh books are checked against test-book leakage only, not against other editions of training books.

## Round 2 (started 2026-10-23 ~08:00): user said "go for the next round"

| job | what | where | status |
|---|---|---|---|
| R2-1 train (detached, GPU) **done** | E10's 20M model trained on the **3 GB corpus** (8,206 leakage-checked books), 18,000 steps × 32 × 512 = 295 MB of fresh text sampled (E10: 82 MB, 1 epoch of 85 MB); same 1 MB val holdout (its books excluded) | `lab/round2/train_big.py`, `logs/20m_big.log`, checkpoint `tables/round2_20m_big.pt` | running, ETA ~6.5 h |
| R2-2 mix (detached) | switching estimate model × **3 GB table + adaptive**; first with E10's 20M, then automatically with the R2-1 model | `lab/round2/mix_big.py`, `run_mix.sh`, `logs/mix.log`, `results/mix_*.json` | first part done |
| E11 (agent) | **C port** of codec v1 + adaptive d3-sum-w32: mmap table format, bit-identical to Python, speed | `lab/c/RESULTS.md` | **done**, see E11 below |
| E12 (agent) | **integer-only inference** of the byte transformer (numpy), integer frequency rule, real files + round trips, determinism checks | `lab/round2/intmodel/RESULTS.md` | agent stopped for token budget at ~09:30, but its runs had finished: **done** for 1m/5m/20m; the 3 GB model is running (main session) |
| R3-1 train (detached, GPU) | 20M trained longer on books only (54,000 steps) | `logs/20m_long_stopped.log` | **stopped** at step ~1,000 in favour of R3-2 (E13 showed domain coverage matters more) |
| R3-2 train (detached, GPU) | 20M on a **domain mix**: books 60% / Wikipedia 25% / code 15%, 54,000 steps, ~21 h | `lab/round2/train_mixed.py`, `logs/20m_mixed.log`, `tables/round2_20m_mixed.pt` | running since 2026-10-23 ~18:40 |

### R2-2 first result: 20M model (E10, 85 MB) × 3 GB table + adaptive (ideal bits + 7 B container, achievable switching)
| size | table (3 GB + adaptive) | model (20M) | **switch per 16 B** | oracle per token (not achievable) |
|---|---|---|---|---|
| 1 KB | 238.8 B | 238.3 | **223.8** | 198.1 |
| 3 KB | 694.0 | 687.7 | **645.8** | 569.3 |
| 10 KB | 2,260.4 | 2,259.9 | **2,107.0** | 1,851.5 |

Equal alone, **-6 to -7% together**: they make their mistakes in different places. Real cmix-style mixing should land between the switch and the oracle.

### E11 result: C port (`lab/c/`, details in RESULTS.md)
`lab/c/steno` (C11, no dependencies, `make -C lab/c`) + `convert_table.py` (npz → flat mmap file with precomputed helpers).
- **Bit-identical to Python** for static (mode 1) and adaptive d3-sum-w32 (mode 2), with both the v1 table and the 3 GB table:
  216 inputs (6 edge cases, 200 snippets, 10 full books) × 2 tables × 2 modes, C output == Python encoder output, and every C round trip OK.
  Fingerprints are recomputed in C (v1 `1cd43ea2`, 3 GB `9330c838`); a wrong table is refused.
  Main-session spot check: book 46, adaptive v1, C = 36,911 B (same as Python), C round trip OK, **Python decodes the C file**.
- **Speed** (shared machine, noisy): static 2-4 MB/s, adaptive 0.5-1.6 MB/s, i.e. **20-250× Python**; decode 50-110× Python.
  Table open+mmap 0.1-0.2 ms (Python load: 0.6 s / 5-7 s). Book 46 (158 KB) decodes in 0.17 s.
- Converted tables are large (precomputed helpers, uncompressed): v1 238 MB (npz 31 MB), 3 GB table 4.1 GB (npz 545 MB).
  The helpers rely on a nesting property (every order-2 follower appears in the order-1 context), which holds for 100% of entries
  and is flagged in the file; a general fallback path is also bit-identical.
- **Spec issues found** (fix in the format before freezing it): (1) mode byte 2 is used by two different models (adaptive.py
  d3-sum-w32 and the E9 chain variant with count_unk), so files are ambiguous: give each model its own mode/version id;
  (2) the chain has no stored fallback; (3) the `<` vs `≤` 2^30 total limit; (4) UTF-8 validity must be defined as strict decoding;
  (5) the new-word limit (2^19) needs a defined error; (6) the order-0 decode search is free (any correct search is fine).

### R2-1 result: 20M model trained on the 3 GB corpus (ideal bits + 7 B; int8 = fake-int8)
295 MB sampled from 3.0 GB (~10% seen), 427 min on the M4 GPU. Validation bpc 1.835 (step 5k) → 1.724 (10k) → 1.684 (15k) → **1.658 (18k)**:
still falling, so not saturated.

| model | 1 KB | 3 KB | 10 KB | 30 KB | books bpc |
|---|---|---|---|---|---|
| 20M, 85 MB (E10) | 238 B | 688 | 2,260 | 6,722 | 1.899 |
| **20M, 3 GB corpus, fp32** | **207** | **589** | **1,923** | **5,711** | **1.562** |
| 20M, 3 GB corpus, int8 | 207 | 590 | 1,927 | 5,722 | 1.565 |
| *3 GB table + adaptive (real bytes)* | *239* | *694* | *2,260* | *6,672* | *1.755* |

**The model now beats the best table everywhere**, including whole books (1.56 vs 1.76) despite its 512 B context:
-13% at 1 KB and **2.8× smaller than gzip** (588 B). Mixing with the 3 GB table (switch per 16 B): 204.5 / 584 / 1,904 / 5,640 B,
only -1% now. The oracle (185 / 528 / 1,713 / 5,062) shows real mixing could still gain ~10%.

### E12 result: integer-only inference (`lab/round2/intmodel/`, details in RESULTS.md)
No float anywhere in the decode path: int8 weights with integer multiplier+shift, int64 accumulators, integer LayerNorm/softmax/GELU,
a Q10-logits → frequency rule (f_i = 1 + floor(e_i · 2^24 / Σe), total < 2^30). Float BLAS is optional as an *accelerator* only where all
terms are exactly representable (< 2^24 for f32, < 2^53 for f64). Each model file has a **self-test hash** of the integer logits for a fixed input, so
a backend that doesn't reproduce exactly is refused. Exported files are ~19.9 MB for 20M, with fingerprints.

| model | 1 KB | 3 KB | 10 KB | 30 KB | vs fp32 ideal | round trips |
|---|---|---|---|---|---|---|
| 1m | | 823.6 B | 2,617 | 7,721 | +1.5% | 200/200 |
| 5m | 256.4 B | 747.0 | 2,456 | 7,317 | +0.4% | 200/200 |
| **20m (E10)** | **238.5 B** | **688.2** | **2,262** | **6,729** | **+0.07-0.11%** | **200/200** |
| **20m_big (3 GB)** | **207.2 B** | **590.0** | **1,926.6** | **5,722.3** | **+0.19-0.25%** | **200/200** |

Determinism: identical output across int/f32/f64 backends and thread settings (subsets, see RESULTS.md). **Speed is the new
bottleneck**: numpy runs at ~0.5 KB/s encode and 0.1-0.3 KB/s decode (20M); the model needs a C/NEON port.

### E13: does it generalise beyond Gutenberg? (`lab/round2/domains.py`, `mixed_table.py`)
30 snippets per size per domain, never in any training data. Wiki = raw enwik8 dump with markup, from enwik8[90 MB:];
code = CPython 3.15.0 top-level `Lib/*.py`. Real bytes (C codec, round trips checked) unless marked *ideal*.

| 1 KB / 10 KB per doc | gzip | brotli | v1 + adaptive | 3 GB books + adaptive | **mixed + adaptive** | 20M, books 85 MB (*ideal*) | 20M, books 3 GB (*ideal*) |
|---|---|---|---|---|---|---|---|
| books | 565 / 4,410 | 462 / 3,669 | 254 / 2,345 | 224 / 2,100 | 256 / 2,382 | 220 / 2,077 | **197 / 1,819** |
| wiki | 538 / 4,201 | 406 / 3,245 | 363 / 3,412 | 321 / 3,062 | **229 / 2,195** | 493 / 5,029 | 278 / 2,749 |
| code | 421 / 2,936 | 359 / 2,531 | 384 / 3,017 | 370 / 2,989 | **220 / 1,980** | 407 / 3,930 | 301 / 2,796 |

**Mixed table** = the same 85 MB of books as v1 + enwik8[0:85 MB] + 35 MB of CPython `Lib/<subdir>` code (67 MB npz, 310K words,
fingerprint `bc33e372`, built in 62 s).
- **Tables and models are domain-bound.** The books-only static tables lose to gzip on code, and the books-only 20M model loses
  to gzip on Wikipedia.
- **Coverage beats volume:** the 67 MB mixed table beats the 545 MB books-only table by -29% on wiki and -40% on code, while books stay level
  with v1 (256 vs 254 B). It's far ahead of brotli everywhere (code: 220 vs 359 B).
- Caveat: the code test files come from the same project as the code training files (different files, same style), so the code gain is optimistic.
  A fair code test needs another project.
- More (and more varied) books make the model much more general: the 3 GB-books model beats brotli in every domain, though it never saw
  Wikipedia or code. But domain data still wins outside books (mixed table: wiki 229 vs 278, code 220 vs 301 B).
- Consequence: the long model run was switched from books-only to a domain mix (R3-2).

**Milestone (2026-10-23):** 20m_big as an integer-only model gives **real, round-trip-verified files of 207.2 B per 1 KB: 2.84× smaller than
gzip (588 B)** and 13% smaller than the best table (239 B). Speed in numpy: ~0.4-1.1 KB/s encode, ~0.12-0.21 KB/s decode.

### Tricks round (2026-10-23 evening): routing, match model, code tokenizer
| trick | what | result | verdict |
|---|---|---|---|
| **1. table routing** (`routing.py`) | the encoder tries every table and keeps the smallest; the fingerprint in the header names it, so it costs 0 extra bytes | picks the right table for 30/30 books and code and 59/60 wiki; result = best of both: books 224, wiki 229, code 220 B (1 KB) | ✅ adopt: "a set of tables, chosen per file" |
| **2. match model** (`match_model.py`) | PAQ-style match model (hash of 6 bytes, follow the earlier occurrence) + an APM table (match length × model probability), next to the 20M model; ideal bits | code -5% (1 KB) / **-17%** (10 KB), wiki -3% / -8%, books +0.8% / -0.3%; matches cover 39-56% of code bytes | ✅ adopt, with a header flag (on/off) so books never lose; full books pending |
| **3. code-aware tokenizer** (`code_tokens.py`) | identifiers with `_`/digits, digit runs and indentation runs as one token; mixed table rebuilt | code **+4.5% worse** (220 → 230 B), wiki -0.5%, books equal | ❌ hand-written splitting loses: whole identifiers are often unique (spelled), while the old pieces are reusable. A tokenizer change should be *learned* (BPE) |

## Reproducibility (2026-10-22)

- Git tracks code, manifests and results; `corpus/` and `tables/` are gitignored and rebuilt by
  `lab/fetch_corpus.py` (corpus, SHA-256 verified against `lab/corpus_manifest.json`) and `lab/codec.py build`.
- The samples are now deterministic: Python source from the pinned CPython 3.15.0 tarball (was the local stdlib),
  random data from SHAKE-256 with a fixed seed (was /dev/urandom), gzip sample with mtime=0 and no filename.
  E1/E2 were re-measured on them (numbers above updated; the E4 python row is still from the old sample).
- Rebuilding the table from the corpus gives the same fingerprint `1cd43ea2` (verified).
- A fresh rebuild in an empty directory took ~3 min: 105/106 files identical. **Gutenberg book 2852
  (The Hound of the Baskervilles) was corrected the same day** ("there came a" → "there came a sound of").
  It's a training book, so a table built from today's download would get a different fingerprint.
  Resolved by the user's decision that **the method is the deliverable**: source drift is a warning
  (`fetch_corpus.py` reports it; `--strict` makes it an error), and results may differ slightly from the log.
  Two levels to keep apart: the *method* (corpus selection rules, tokenizer, counting, cutoffs, format)
  is what we publish; a *table instance* (fingerprint) is what an archive must keep next to its files.

## Discussion 2026-10-22: Sloot's own framing, and what to optimise next

The user found clues in the Dutch Wikipedia article: Sloot called it **"coding", not "compression"**; his system was
"not built on the binary system but compatible with it"; and "with clever helper tables, stronger compression than
Shannon allows could be achieved, though not every arbitrary film could be shrunk that way".

Interpretation: helper tables don't break Shannon. They change the bound from H(X) to the **conditional entropy H(X | T)**
given the shared table T (Kolmogorov: K(x | T)). Random data stays incompressible (we measured 8.00 bpc).
"Running with it" therefore means two levers: (1) more *relevant* knowledge in T, (2) a coder that exploits T more fully.

Baggage we don't carry (unlike streaming video): real-time decoding, low decoder memory, cheap hardware, fast
encoding, constant bitrate, seeking. **The encoder may be very slow and search; the decoder may use GBs.**

Proposed next steps (after E8c):
- **Speed:**
  - A: C core for coder + model (50-100×), a prerequisite for context mixing;
  - B: memory-mappable table format;
  - C: table compaction (quantised counts, pruning);
  - D: parallel per document;
  - E: GPU only for table building.
- **Size:**
  - #1 context mixing instead of PPM (literature: -10..20%);
  - #2 encoder-side search, choice recorded in header bits;
  - #3 domain-sharded table, chosen per file;
  - #4 learned subword tokens;
  - #5 larger table + longer contexts (E8c);
  - #6 the archive as its own table (dedup/delta against earlier items or a public library; needs a realistic archive dataset).
- Recommendation: speed first (A+B with the current best E8a+E8b model), then #1/#2/#3/C in parallel, then #6.
  The user decided to **wait for E8c** first, and then also look into newer research.

### Newer research, including quantum (from memory, to be verified before relying on it)
- **Quantum gives no way around Shannon for classical data:**
  - Holevo bound: n qubits yield at most n classical bits.
  - Schumacher compression is for quantum states.
  - Quantum random access codes are probabilistic and reveal one bit at a time.
  - Grover search needs a quantum computer.
- **Superdense coding** (2 bits per qubit with pre-shared entanglement) is the quantum analogue of the Sloot principle:
  a pre-shared resource lowers what must be sent, but never below the bound that applies given that resource.
- **Usable spin-offs and other post-1999 work:**
  - **tensor networks** (MPS/tensor trains, Born machines) for compact n-gram tables;
  - **FM-index / r-index** (2000/2018), compressed self-indexes: find any passage in a huge reference library. The
    mechanism for "the archive as its own table";
  - **Relative Lempel-Ziv** (2010): compress a collection relative to a reference;
  - **context mixing** (PAQ 2002 → cmix);
  - **ANS** (Duda): fast entropy coding;
  - succinct data structures / minimal perfect hashing: mmap tables;
  - **fountain codes** (LT 2002, Raptor 2006): any k of n pages;
  - content-defined chunking (FastCDC 2016): dedup.
- **Long-lived media** (modern physics, not quantum computing): femtosecond-laser glass (Southampton 5D, Microsoft Project Silica),
  ceramic (Cerabyte), DNA storage (DNA Fountain 2017). The format must stay medium-agnostic; paper is one target.

### Recursion: compressing the table with the algorithm itself (2026-10-22)
- Compressing compressed output again gains nothing (output is near-random; E1: .gz stays at 8.00 bpc).
- Compressing T with T is circular. But **T is a deterministic function of corpus + method**, and adaptive mode can
  start from an empty table. So ship the *seed corpus* compressed adaptively, and let the recipient rebuild T.
  Measured: training corpus 84.7 MB → xz -9e 21.1 MB, zstd -19 21.8 MB, vs the table it produces (v1 npz) 31.0 MB.
  **The compressed seed is smaller than the table** (it can't hold less information than its derivative). Our adaptive
  codec would give ~20 MB at 1.91 bpc. Exception: pruned tables discard information and can be smaller than their seed;
  ship those in a compact structural format (delta-coded sorted keys, log-scale counts).
- **Bootstrap chain** (the useful form of recursion, and the answer to "the table must last as long as the archive"):
  T0 = tiny/empty base + the method in human-readable form (printable) → T1 = seed corpus compressed with T0, rebuilt
  by the method → archive compressed with T1. Fully self-contained on one medium; the seed is a one-off cost spread
  over the whole archive.

### Second round with a "table for random data"? (2026-10-22)
The user's question: random doesn't exist in computers, so could a second table compress the output again?
Measured on 326,371 B of v1 output (5 test books, headers stripped):
xz -9e +0.03%, zstd -19 +0.01%, brotli +0.00%, adaptive byte models of order 0/1/2: +0.05% / +1.79% / +0.05%. **No gain.**
- Why: a good coder uses (nearly) every output bit pattern, so there's no structure left for a second table. Any structure
  a second table could find is a flaw of round 1, and belongs in the round-1 model (where it's worth more).
- "Pseudo-random isn't random" is true for PRNG output: `random_1M.bin` is SHAKE-256 of a 23-byte string, so its
  Kolmogorov complexity is ~30 bytes plus the algorithm. But finding the seed from the output means breaking the hash, and
  compressed output isn't PRNG output: it's a nearly 1-to-1 function of the input, carrying the input's information.
- Changing the alphabet/character set is a relabelling: the same information in other symbols (log2 conversion).

### The "magic table" the user wants (2026-10-22)
Goal: **one published table (per language or for everything), never shipped with files**, like Brotli's built-in
dictionary (RFC 7932, ~120 KB, inside every browser), but much larger. Files carry only the 4-byte fingerprint.
- Technically we already have this (`en-gutenberg-v1`, fingerprint `1cd43ea2`). What's hard:
  (1) **coverage**: no table is best for everything (E4: the gain vanishes outside the table's domain), so it must be
  trained on a broad, diverse corpus, possibly as domain shards inside one published table;
  (2) **stability**: published versions are frozen forever, and improvements become new versions with new fingerprints;
  (3) **preservation**: the published table itself is preserved/distributed via the bootstrap chain (E9), not per file.

### E9: bootstrap chain (`lab/exp_chain/`), started 2026-10-22
Seed corpus compressed with T0 (empty/tiny table, adaptive), decoded in a fresh process, T1 rebuilt and checked for
fingerprint `1cd43ea2`, plus a first human-readable METHOD.md. **Status: done, fully verified** (details in RESULTS.md).
- T0 = empty table + adaptive codec (+ `count_unk`: new words also increment UNK). **Progressive** seed: chunks at 1, 2, 4,
  8, … MB; chunk k is coded with a table rebuilt (v1 method) from chunks 0…k-1. Knowledge built on knowledge.
- **Seed = 19.09 MB (1.802 bpc)** for 84.7 MB of text: **38% smaller than the 31.0 MB table** it produces, 10% smaller than
  xz (21.1 MB). Per-book from T0 only: 21.30 MB (fully parallel, any book decodes alone).
- Verified in a fresh process: 90/90 books identical, rebuilt fingerprint `1cd43ea2`, all 12 arrays identical, and 41/41
  test files made with the original table decode with the rebuilt one. Decode took 27 min in Python.
- A small early table can hurt a big self-contained book (book 100: T0 2.088 vs table 2.166). Picking per book (1 flag) saves ~0.6%.
- `lab/exp_chain/METHOD.md`: the first human-readable spec of the whole chain (tokeniser, table, model, coder, container).

### Lessons from machine learning for the table (2026-10-22, from memory, to be verified)
Principle: **use ML only when building the table; the decoder reads only frozen integers** (keeps the no-drift rule).
1. **Scaling laws** (Kaplan 2020, Chinchilla 2022): predictable but diminishing returns from data. For count tables, more
   data mainly unlocks **longer contexts** (E8a: order 3 lost at 84 MB). E4/E8c measure our curve.
2. **Quality > quantity:** deduplication (Lee et al. 2021), filtered data (FineWeb-Edu 2024), optimised domain mixtures
   (DoReMi 2023). For us: dedup editions, balance domains, measure bits per GB.
3. **Generalisation is the gap vs LLMs:** counts can't share statistics between similar words. Option: word clusters
   learned offline (Brown clusters 1992, or modern embeddings at build time), frozen as integer class ids and used as
   extra contexts. **Most promising.**
4. **Retrieval** (kNN-LM 2020, RETRO 2021): a big lookup datastore improves prediction, and growing the datastore helps without
   retraining. That's Sloot's library / "archive as its own table" (#6, FM-index).
5. **Redundancy:** quantisation/pruning barely hurt. For us: log-scale 4-8-bit counts, **entropy-based pruning**
   (Stolcke 1998), turning the 10 GB question into "most bits saved per GB of table".
6. **Distillation** (open model fills unseen contexts, frozen as integers): possible, but makes the build method depend
   on a specific model. A user decision; only after the options above are exhausted.

Candidate experiments after E8c: (a) frozen word classes as extra context, (b) entropy pruning + quantised counts
(size-vs-bits curve), (c) corpus quality: dedup + domain mix vs raw volume.

### Dropping "no drifting models"? A tiny self-built model (2026-10-22, under discussion)
The user is open to models if they can be built reproducibly. Key reframing: drift came from *hosted, retuned* models.
A model we train, freeze and fingerprint ourselves doesn't drift. Quantised to integers (int8 weights, integer
accumulators, fixed exp/softmax tables) it's **a table of integers plus a fixed integer program**. The hard requirement
remains bit-identical inference on every machine (integer-only, no floats). Training needn't be bit-reproducible:
the method is the deliverable, and each trained instance gets a fingerprint, like tables.
- Literature (from memory, to verify): small pretrained models ~1.0-1.5 bpc on English vs our 1.9-2.3; Bellard's ts_zip
  (small pretrained model + arithmetic coding, deterministic inference); Hutter Prize leaders (cmix, nncp) < 0.9 bpc
  on enwik9 including decompressor size, with online-trained networks; DeepMind 2023 for large LLMs (model size ignored).
- Others optimise one big file or ignore model size. **Our niche:** short documents with a shared published model,
  archival determinism + bootstrap chain + printable method, and the standard/versioning system. Strategy: adopt the
  best model ideas, mix model + n-gram table + document layer (cmix-style), and focus our own work on the archival system.
- Proposed **E10**: (1) train 1/5/20M-parameter models on the same 90 books (PyTorch on the M4 GPU, project .venv) and measure
  ideal test bpc; (2) only if promising: int8 integer-only inference, real bytes, mixing with the table.
- **E10 step 1 results so far** (`lab/exp_tinymodel/RESULTS.md`; byte-level transformer, context 512 B, trained on the same
  90 books; *ideal* bits + 7 B container, not real files):

  | model | int8 size | 1 KB | 3 KB | 10 KB | 30 KB | books bpc | int8 cost |
  |---|---|---|---|---|---|---|---|
  | 1M params | 0.9 MB | 305 B | 812 | 2,576 | 7,597 | 2.170 | +1.5% |
  | **5M params** | **5 MB** | **256** | **744** | **2,446** | **7,285** | 2.101 | **+0.4%** |
  | **20M params** | **19 MB** | **238** | **688** | **2,260** | **6,722** | **1.899** | **+0.1%** |

  **A 19 MB model trained on 85 MB of text equals the 3 GB table + adaptive (545 MB, trained on 35× more text) on short
  docs** (238 vs 239 B at 1 KB, 6,722 vs 6,672 at 30 KB), at 1/28 of the size. On books it trails (1.899 vs 1.755) because of its
  512 B context. The 20M model saw only 0.98 epochs of 85 MB, so more data should help it a lot.
  **Mixing estimate** (`mix.py`, achievable: the encoder picks the model or the table per ~16-byte segment and codes the choice with an
  adaptive flag; ideal bits, table = v1 + adaptive): 20M + table: 236 / 678 / 2,202 / 6,494 B (1/3/10/30 KB); 5M + table:
  249 / 717 / 2,321 / 6,833 B. The oracle per token (not achievable) is 214 B at 1 KB, so they're strongly complementary and real
  cmix-style mixing should land in between. Not yet measured: mixing with the 3 GB table.
  Earlier: a 5 MB model trained on 85 MB already matches the 1 GB pruned table (68 MB, 255 B at 1 KB) and beats v1 + adaptive on
  short docs (276 B). That's the generalisation advantage. On whole books, table + adaptive wins (1.78-1.91 vs 2.10), because the model only sees
  512 B of context. Mixing model + table + document layer should combine both (`mix.py` is queued after the 20M run).
  Fake-int8 costs only +0.4% for 5M, so integer-only inference looks plausible (LayerNorm/softmax/GELU not yet integerised).
- 2026-10-22: user said **go**. E10 step 1 is running in `lab/exp_tinymodel/` (models ~1/5/20M params, same 90 books, ideal bpc,
  fp32 vs fake-int8, size, speed; optional bigger-data run and table-mix estimate).

## Open questions and parked ideas

- **Recompression of already-compressed data** (precomp/preflate style: unpack deflate/zip/png/pdf streams and rebuild them bit-exactly). The main lever for "already compressed data".
- Grammar compression with search instead of greedy (the user's recursive table idea, made stronger); parked after E2.
- Context mixing (PAQ/cmix-style, deterministic, no external weights) as the coder behind the table.
- Other languages: one table per language, or one multilingual table.
- Table longevity: the table must be preserved as long as the archive (archive it once, redundantly, open format, hash-pinned).
- Paper layer: symbology, module size, ECC, multi-page erasure coding (postponed by the user).
- Speed: Python ≈ 30 KB/s; port to C/Rust once the format settles.
