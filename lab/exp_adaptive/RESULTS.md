# E8b: per-document adaptive layer in the real codec

Question: how much does a per-document adaptive layer add, in **real compressed bytes**,
on top of the frozen table (`tables/en-gutenberg-v1.npz`, fingerprint `1cd43ea2`),
compared with codec v1 (static, `lab/codec.py`)? The table is never modified. The layer
lives in memory only and the decoder rebuilds it identically, so nothing extra is stored.

Code: `lab/exp_adaptive/adaptive.py` (codec), `lab/exp_adaptive/run.py` (measurements).
Raw results: `lab/exp_adaptive/results/*.json` and `*.log`. No shared files were modified.

## Result in one line

**Recommended config `d3-sum-w32`: full books 2.420 → 1.911 bpc (−21%), short documents
−3% (1 KB) to −9% (30 KB). Never worse than static by more than 2 bytes on any of the 200 snippets.
Every snippet and all 10 full books decode byte-identically.**

## What was tried

The model is a PPM walk over a configurable list of levels, then a final order-0 level.
Level types: `T k` = table-only order-k context; `M k W` = table counts + W × document
counts in the same order-k context; `D k W` = document-only order-k context (× W).

| config | levels (highest first) | order-0 doc weight W0 | new-word ids | spelling doc weight |
|---|---|---|---|---|
| static | T2, T1 | 0 | no | 0 |
| sum-wW | M2 W, M1 W | W | yes | 1 |
| sum-w4-nonew | M2 4, M1 4 | 4 | **no** | 1 |
| sum-w4-wsS | M2 4, M1 4 | 4 | yes | S ∈ {0, 4, 16} |
| cache-wW | D2 W, D1 W, T2, T1 | W | yes | 1 |
| cache3-wW | D3 W, M2 W, M1 W | W | yes | 1 |
| d3-sum-wW | D3 1, M2 W, M1 W | W | yes | 1 |
| d3d2-sum-w32 | D3 1, D2 1, M2 32, M1 32 | 32 | yes | 1 |

Sanity check: `static` produces output byte-identical to codec v1 (except the mode byte)
for every snippet. This was asserted in the run.

## Short documents: 200 snippets (`short_docs.snippets`), real bytes, all round trips verified

Average real file size per document (mode + fingerprint + count + code), and bpc:

| config | 1 KB | bpc | 3 KB | bpc | 10 KB | bpc | 30 KB | bpc |
|---|---|---|---|---|---|---|---|---|
| static (= v1) | 285 | 2.271 | 837 | 2.220 | 2,756 | 2.192 | 8,207 | 2.176 |
| sum-w1 | 280 | 2.232 | 813 | 2.157 | 2,635 | 2.096 | 7,746 | 2.053 |
| sum-w2 | 280 | 2.227 | 811 | 2.150 | 2,624 | 2.087 | 7,707 | 2.043 |
| sum-w4 | 279 | 2.223 | 809 | 2.144 | 2,613 | 2.079 | 7,669 | 2.033 |
| sum-w8 | 279 | 2.218 | 806 | 2.138 | 2,603 | 2.071 | 7,634 | 2.023 |
| sum-w16 | 278 | 2.214 | 804 | 2.132 | 2,593 | 2.063 | 7,600 | 2.015 |
| sum-w32 | 278 | 2.210 | 802 | 2.126 | 2,584 | 2.056 | 7,571 | 2.007 |
| sum-w64 | 277 | 2.206 | 800 | 2.120 | 2,576 | 2.049 | 7,549 | 2.001 |
| sum-w128 | 277 | 2.203 | 798 | 2.115 | 2,570 | 2.045 | 7,537 | 1.998 |
| sum-w4-nonew | 281 | 2.236 | 817 | 2.167 | 2,647 | 2.106 | 7,793 | 2.066 |
| sum-w4-ws0 | 279 | 2.223 | 809 | 2.144 | 2,614 | 2.079 | 7,672 | 2.034 |
| sum-w4-ws4 | 279 | 2.224 | 808 | 2.143 | 2,612 | 2.078 | 7,664 | 2.031 |
| sum-w4-ws16 | 279 | 2.222 | 808 | 2.142 | 2,610 | 2.076 | 7,654 | 2.029 |
| cache-w1 | 280 | 2.227 | 805 | 2.134 | 2,597 | 2.066 | 7,647 | 2.027 |
| cache-w4 | 294 | 2.343 | 858 | 2.275 | 2,798 | 2.226 | 8,289 | 2.197 |
| cache-w16 | 316 | 2.514 | 937 | 2.484 | 3,096 | 2.462 | 9,246 | 2.451 |
| cache3-w1 | 277 | 2.206 | 798 | 2.116 | 2,576 | 2.049 | 7,567 | 2.006 |
| cache3-w4 | 277 | 2.207 | 799 | 2.119 | 2,581 | 2.053 | 7,596 | 2.013 |
| cache3-w16 | 278 | 2.213 | 803 | 2.130 | 2,600 | 2.068 | 7,674 | 2.034 |
| d3-sum-w8 | 276 | 2.198 | 795 | 2.108 | 2,561 | 2.037 | 7,514 | 1.992 |
| **d3-sum-w32** | **276** | **2.194** | **793** | **2.102** | **2,552** | **2.030** | **7,484** | **1.984** |
| d3-sum-w64 | 275 | 2.192 | 792 | 2.099 | 2,548 | 2.027 | 7,474 | 1.981 |
| d3d2-sum-w32 | 275 | 2.193 | 792 | 2.100 | 2,559 | 2.036 | 7,527 | 1.995 |

Per snippet, `d3-sum-w32` vs static (`run.py perdoc`):

| size | better | equal | worse | worst case |
|---|---|---|---|---|
| 1 KB | 31 | 16 | 3 | +1 byte |
| 3 KB | 46 | 1 | 3 | +2 bytes |
| 10 KB | 50 | 0 | 0 | — |
| 30 KB | 50 | 0 | 0 | — |

## Full books: the 10 unseen test books, real bytes

bpc per book; `*` = decoded and verified in the sweep (books ≤ 500 KB). For `d3-sum-w32`,
**all 10 books** were also decoded and verified byte-identical afterwards (`run.py verify`,
log in `results/verify_d3-sum-w32.log`).

| config | 10 | 46 | 145 | 408 | 1232 | 1952 | 2701 | 5197 | 20203 | 37683 | **TOTAL** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| static (= v1) | 2.250 | 2.017* | 1.853 | 2.020* | 1.876* | 1.835* | 2.155 | 1.835 | 2.024* | 3.943 | **2.420** |
| sum-w4 | 1.728 | 1.890* | **1.723** | **1.965*** | 1.781* | 1.821* | **2.032** | **1.733** | 1.935* | 2.655 | 1.947 |
| sum-w32 | 1.696 | 1.871* | 1.733 | 1.967* | **1.772*** | 1.816* | 2.041 | 1.737 | 1.922* | 2.616 | 1.931 |
| sum-w128 | 1.765 | 1.875* | 1.804 | 2.000* | 1.790* | 1.816* | 2.109 | 1.795 | 1.946* | 2.692 | 1.995 |
| **d3-sum-w32** | **1.664** | **1.864*** | 1.745 | 1.971* | 1.774* | **1.807*** | 2.048 | 1.749 | **1.921*** | **2.552** | **1.911** |
| d3-sum-w64 | 1.693 | 1.864* | 1.769 | 1.981* | 1.778* | 1.808* | 2.071 | 1.766 | 1.927* | 2.583 | 1.936 |

Total = 12,186,248 raw bytes. For reference (E5, other tools on the same books): xz −9e
2.306, zstd −19 2.342, zstd + 84 MB reference 2.249, earlier estimate "adaptive o2" (lab/steno.py) 2.005.
`d3-sum-w32` beats xz on 9 of 10 books. The exception is 37683 (etymological dictionary):
3.943 → 2.552, vs xz 2.543 (0.4% worse).

## Conclusions

1. **The adaptive layer is a clear win and costs nothing in file size.** Books −21% (2.420 → 1.911).
   That's 17% below xz and 15% below zstd with the same 84 MB as reference. It's also
   better than the estimate from E5 (2.005), mostly because PPM with exclusion beats the interpolated estimate.
2. **For short documents the gain is small but consistent** (−3% at 1 KB up to −9% at 30 KB):
   the frozen table does most of the work there, as E6 predicted.
3. **Document counts need a large weight.** W = 1 is too weak; W 16–64 is best for short
   documents. On long books W ≈ 4–32 is best and W = 128 starts to hurt (the document's
   own early statistics drown out the table).
4. **Doc-only contexts must not sit in front of the table at the orders the table also has.**
   `cache-w*` (doc-only order 2/1 before the table) gets worse as W grows (cache-w16: 2.45 bpc,
   worse than static). Each doc-only escape costs bits, and the exclusions throw away the table's better statistics.
   **A doc-only order-3 level in front works** (the table has no order 3, so it adds
   information): `d3-*` is the best family. Weighting it at 1 is best.
5. **New-word ids help:** without them (`sum-w4-nonew`), 30 KB docs cost 1.6% more. The effect is larger on name-heavy text.
6. **Adapting the spelling model barely matters** (ws 0…16 within 0.3%); keep ws = 1 or drop it for simplicity.
7. Per-document worst case vs static: +2 bytes (6 of 200 snippets ≤ 2 bytes worse). If a strict
   "never worse than static" guarantee is wanted, the encoder can try both modes and keep the
   smaller one (the mode byte already says which); that doubles encode time and doesn't change the decoder.

## Recommendation

Use **`d3-sum-w32`**: levels `D3 ×1, M2 ×32, M1 ×32`, order-0 doc weight 32, new-word ids
on, spelling doc weight 1. It's the best total on full books and within 0.1–0.2% of the best
(`d3-sum-w64`) on short documents. Caveat: W was chosen on the same test set it's
reported on. The choice is coarse (powers of 2) and neighbouring values differ by < 2%, so the
risk of over-fitting is small, but a fresh held-out set should confirm it.

## Algorithm (precise enough to port)

State per document (starts empty for every file; identical in encoder and decoder):
- `V` = table vocab size. New-word ids are `V, V+1, ...` in order of first appearance.
- `doc_uni[id]`: count of each coded id. `doc_total`: number of coded tokens.
- `doc_ctx[k][(h_{-k}..h_{-1})][id]` for k = 1, 2, 3: counts of id after the last k history ids.
- `doc_spell[prev][b]`: letter-pair counts of spelled words (prev/b ∈ 0..256, 256 = start/end).
- History `h` = last 3 coded ids (new words by their new id). For **table lookups** the history
  is mapped: any id ≥ V becomes UNK (0), because the table was trained with rare words as UNK.

Coding one token `t` with id `w` (`w` = table id, else new-word id if seen before in this document, else UNK):
1. Walk the levels in order: `D3 ×1`, `M2 ×32`, `M1 ×32`. A level is skipped if the history is
   shorter than k or the context has no entries.
   - `D3`: candidates = `doc_ctx[3][last 3 ids]`, count × 1, sorted by id.
   - `M k`: candidates = the table's order-k group for the mapped history (sorted by id; count = table count +
     32 × doc count for that id in `doc_ctx[k][last k actual ids]`), followed by ids that occur only in
     the document context (sorted by id; count = 32 × doc count).
   - Remove every id already excluded by a higher level. If nothing is left, skip the level.
   - Escape count = number of remaining candidates. Total = sum of counts + escape.
   - If `w` is a candidate: code it with its cumulative range (candidates in the order above), done.
   - Otherwise code the escape `[total − esc, total)` and add all remaining candidates to the excluded set.
2. Order 0: every id in `0 .. V + new_words − 1`, in id order, count = table count (0 for new
   ids) + 32 × `doc_uni[id]`, excluded ids get zero width. No escape. (Fenwick tree over
   `doc_uni` for the prefix sums; the decoder binary-searches the id.)
3. If `w` = UNK: spell the token's UTF-8 bytes, then 256 (end). Each byte is coded with counts
   `table_spell[prev][b] + 1 + 1 × doc_spell[prev][b]` over b ∈ 0..256.
4. Update (after coding, same in decoder): if `w` = UNK, assign it the next new-word id and update
   `doc_spell` with its letter pairs. Then with the (possibly new) id: `doc_uni[id] += 1`,
   `doc_total += 1`, and for k = 1, 2, 3 with enough history `doc_ctx[k][last k ids][id] += 1`.
   Append the id to the history.

Arithmetic coder, container and table: identical to `lab/codec.py` (32-bit WNC coder,
totals < 2³⁰), except mode byte = 2. Totals stay below 2³⁰ for documents up to ~30M tokens at W = 32
(table order-0 total ≈ 20M + 32 × doc tokens); larger documents need a rescale rule (not implemented).

## Speed (Python, one core)

`d3-sum-w32` ≈ 1.6× slower than static: Bible (4.3 MB) encode 87 s / decode 97 s; all 10 books
350 s encode vs 217 s for static. Short snippets: 200 docs encode + decode in ~75 s.

## Not done / caveats

- No rescaling of counts for documents > ~30M tokens (would hit the 2³⁰ total limit at W = 32).
- W was tuned on the reported test set (see the recommendation).
- Interacts with E8a (model changes: case, line breaks, escapes, longer contexts) and E8c (bigger tables):
  the best W may shift with a bigger table (more table weight → probably a larger W needed).

## Reproduce

From the project root, with `tables/en-gutenberg-v1.npz` built (`uv run python lab/codec.py build`):

```
uv run python lab/exp_adaptive/run.py short                     # all configs, 200 snippets (~25 min, 3 processes)
uv run python lab/exp_adaptive/run.py books static sum-w4 sum-w32 sum-w128 d3-sum-w32 d3-sum-w64   # ~20 min
uv run python lab/exp_adaptive/run.py verify d3-sum-w32         # decode + compare all 10 books (~4 min)
uv run python lab/exp_adaptive/run.py perdoc d3-sum-w32         # per-snippet comparison with static
uv run python lab/exp_adaptive/run.py report                    # print the tables from results/*.json
```
