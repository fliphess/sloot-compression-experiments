# E8a: model improvements for the steno codec

Question: which model changes make codec v1 (`lab/codec.py`, static frozen table, PPM-C
orders 2→1→0) smaller, measured on **real compressed bytes with round-trip checks**?

Run 2026-10-22. Same data as v1: table trained on the 90 training books
(`steno.split_books()`), test = the 200 short snippets of `short_docs.py` (50 each of 1/3/10/30 KB)
plus the 10 full unseen test books.

## Verdict

**Recommended: `wrap+case2+sp2+S` (order 2) = −5.4% on short docs and −5.4% on full books vs v1.**
It wins on every book (−1.3% to −7.0%) and at every snippet size; table 33.1 MB + 32 KB of side tables.

- The only substantial win is **wrap** (the line-break model): −3.9%. The rest each give 0.5-0.9%, and they add up.
- **Longer contexts (order 3/4) make things worse** with this table size (+1.7% / +2.9%), even with SEE escapes or
  minimum-count pruning (best o3 attempt is still no better than o2). Re-test once E8c has bigger tables.
- **PPMD and PPMX escapes are worse than PPMC** (+0.7% / +0.5%). Only measured escapes (SEE, "S") help.
- Model tweaks have limited headroom: in the combination, words found at order 2/1 cost 66% of the bits,
  order-0 fallbacks 16%, spelling unknown words 6.5%, escapes 8%, case 2%, wrap 0.8%. The bigger levers are
  outside E8a: adaptivity (names re-spelled every time in static mode → E8b) and table size (E8c).

## Results (real bytes)

Average compressed size per document (bpc in brackets), full-book total bpc, main table file size.

| variant | 1 KB | 3 KB | 10 KB | 30 KB | short docs vs base (mean of sizes) | full books bpc | books vs base | table on disk |
|---|---|---|---|---|---|---|---|---|
| base (= v1) | 285 B (2.271) | 837 B (2.220) | 2756 B (2.192) | 8207 B (2.175) | +0.0% | 2.420 | +0.0% | 36.2 MB |
| case | 284 B (2.259) | 833 B (2.208) | 2741 B (2.180) | 8154 B (2.162) | -0.6% | 2.409 | -0.4% | 34.5 MB |
| case2 | 283 B (2.256) | 832 B (2.205) | 2736 B (2.177) | 8140 B (2.158) | -0.7% | 2.408 | -0.5% | 34.9 MB |
| **wrap** | 274 B (2.185) | 804 B (2.131) | 2649 B (2.107) | 7888 B (2.091) | **-3.9%** | 2.329 | **-3.8%** | 34.7 MB |
| o3 | 290 B (2.307) | 851 B (2.257) | 2806 B (2.232) | 8352 B (2.214) | +1.7% | 2.462 | +1.8% | 99.4 MB |
| o4 | 293 B (2.332) | 861 B (2.283) | 2839 B (2.259) | 8452 B (2.240) | +2.9% | 2.489 | +2.9% | 190.0 MB |
| D | 287 B (2.287) | 843 B (2.235) | 2776 B (2.208) | 8263 B (2.190) | +0.7% | 2.442 | +0.9% | 36.2 MB |
| X | 287 B (2.284) | 841 B (2.231) | 2771 B (2.204) | 8250 B (2.187) | +0.5% | 2.438 | +0.8% | 36.2 MB |
| S | 284 B (2.260) | 833 B (2.208) | 2743 B (2.182) | 8170 B (2.166) | -0.5% | 2.402 | -0.7% | 36.2 MB |
| sp2 | 283 B (2.255) | 831 B (2.203) | 2736 B (2.176) | 8145 B (2.159) | -0.7% | 2.398 | -0.9% | 36.2 MB |
| **wrap+case2+sp2+S** | **270 B (2.152)** | **792 B (2.099)** | **2609 B (2.075)** | **7760 B (2.057)** | **-5.4%** | **2.289** | **-5.4%** | 33.1 MB |

"base" reproduces v1 exactly (identical file sizes; v1's own table file is 31.0 MB because it packs keys
differently). The table sizes above exclude the SEE (8 KB) and sp2 (24-48 KB) side files.

Ideal-bits checks on the 50 × 10 KB snippets (`breakdown.py`, no container; base = 2.186 bpc):

| variant | ideal bpc |
|---|---|
| o3 + min context count 4 / 16 / 64 | 2.208 / 2.197 / 2.190 |
| o3 + S | 2.196 |
| wrap+case2+sp2+S | 2.069 |
| wrap+case2+sp2+S + o3 + min count 16 | 2.075 (no gain from order 3) |

Per book, recommended vs base (real bytes): 10 −4.9%, 46 −1.3%, 145 −6.0%, 408 −6.3%, 1232 −7.0%,
1952 −4.2%, 2701 −6.2%, 5197 −6.9%, 20203 −5.9%, 37683 −5.0%.

### What was verified

- Every variant: all 200 short snippets compressed, decompressed and compared byte-for-byte, plus edge cases
  (empty, 1 char, Unicode with CRLF, binary, random, unknown words, a mixed case/wrap sample).
- Full books: all 10 encoded for real bytes; the 5 books ≤ 500 KB (46, 408, 1232, 1952, 20203) also decoded and
  compared. **The 5 large books (10, 145, 2701, 5197, 37683) were encoded only**: the same code path, but their
  decode wasn't run (slow in Python).
- Not tested: text that isn't hard-wrapped (modern prose, one paragraph per line). The wrap model should then
  cost only a few bits per document (see below), but that's unmeasured.

## What was tried

1. **case**: word tokens are coded lowercased plus a case class: 0 lower, 1 Capitalized (incl. single upper
   letter), 2 ALLCAPS (≥ 2 letters), 3 mixed (token kept verbatim, e.g. " McDonald"). The context is the word and a
   sentence-position flag. It saves on spelling and order 0 (fewer distinct words), but the lowercased history
   loses information and the flag costs 2.2% of the bits; net −0.6%. Most case cost: Capitalized words mid-sentence
   (names, ~2 bits each), ALLCAPS mid-sentence (~6.4 bits each, headings), lowercase after "." (~3.9 bits).
   **case2** adds the previous word's case class to the context (ALLCAPS runs, multi-word names): −0.7%.
2. **wrap**: Gutenberg text is hard-wrapped at ~70 columns; v1 coded each wrap "\n" as a token (7.6% of tokens in
   Moby Dick), which also broke the n-gram context. A wrap newline is folded into the next word's leading space
   plus a binary flag, predicted from the column the word would end at and the line widths seen so far in the
   document. The wrap flag costs 0.8% of the bits; −3.9% overall.
3. **o3 / o4**: longer PPM contexts via hierarchical context ids. Worse: too sparse with 84 MB of training text,
   so confident sparse contexts cost escapes and wrong predictions. Requiring a minimum context count (m4/m16/m64)
   or SEE escapes reduces the damage but never beats order 2.
4. **Escapes**: PPMD (counts 2c−1, escape d) and PPMX (escape = singletons + 1) are both worse than PPMC.
   **S (SEE)**: escape odds measured per bucket on held-out *training* books (never test books): −0.5%.
5. **sp2**: spell unknown words with an order-2 byte model blended over the order-1 one: spelling −8%, −0.7% overall.

## Algorithm (recommended variant), precise enough to port

Everything not listed is identical to v1 (`lab/codec.py` docstring): tokenizer, vocab (count ≥ 2, sorted by
count desc then UTF-8 bytes asc, id 0 = UNK), order-0 counts, PPM with full exclusion, order-0 without escape,
WNC 32-bit coder, container (here mode byte = 2), token count = number of events.

**Events** (text → list of (token, case class or none, wrap flag or none)):
1. Split with `TOKEN = " ?[A-Za-z]+|[^A-Za-z]"`. A word token = matches `" ?[A-Za-z]+"`.
2. Wrap merge: a `"\n"` token at index i > 0, whose previous token is not `"\n"`, and whose next token is a word
   *without* leading space, becomes the single event token `" " + next` with wrap = 1. Every other word token with a
   leading space gets wrap = 0. Other tokens get no wrap flag.
3. Case: for every word event: class 0 if all letters lower; 1 if the first is upper and the rest lower (or a single
   upper letter); 2 if all upper (≥ 2 letters); else 3. For classes 0-2 the token is lowercased; class 3 keeps it verbatim.
4. Rendering back: apply the case (1: uppercase the first letter after the optional space; 2: uppercase all), then
   if wrap = 1 replace the leading space with `"\n"`.

**Per event, coding order:** token id (PPM, then the spelling if UNK) → case class (if word) → wrap flag (if word
with leading space). Then update the state with the rendered string and advance the contexts. The vocab, n-grams,
case and wrap tables are all built from training events produced by the same rules, with the same state machine
run per book.

**State** (from the rendered output so far; columns count Unicode code points):
- `col`: characters since the last `"\n"`.
- `last_sig`: the last output character that's not `"\n"` and not in `SKIP = space tab " ' ( ) [ ] _ * “ ” ‘ ’`;
  `newline` = a `"\n"` was output after it.
- `max_nowrap` = −1, `min_wrap` = 2^30, `last_cls` = 4 (no word yet).

**Case class:** flag = 1 if `last_sig` is none or in `.!?`; else 2 if `newline`; else 3 if `last_sig` in `:;`; else 0.
Table `case[w][flag][last_cls][cls]` counts training events. Prior `P[flag][last][cls] = 1 + floor(60 * N / max(1,
sum over cls of N))` with N = the counts summed over all words. Frequency = `16 * case[w][flag][last][cls] + P[flag][last][cls]`.
After coding: `last_cls = cls`.

**Wrap flag:** `end = min(col + len(token), 127)` with the leading space counted; `rel` = 0 if `end ≤ max_nowrap`,
1 if `end ≥ min_wrap`, else 2. Frequency = `wrap[end][rel][flag] + 1`. After coding: flag 1 → `min_wrap =
min(min_wrap, end)`, flag 0 → `max_nowrap = max(max_nowrap, end)`. On text without hard wraps, `max_nowrap`
reaches 127 after the first long line, so later flags cost almost nothing.

**SEE escapes (S):** for a context visited during the PPM walk, with exclusions applied: d = remaining candidates,
n = sum of their counts, first = no higher order was visited before. Key = (order − 1, first,
min(bitlen(d), 10), min(bitlen(n), 20)). `shift = min(8, max(0, 25 − bitlen(n)))`; symbol frequencies = `count <<
shift`; escape frequency = `max(1, ((n << shift) * R[key]) >> 16)`, where R = escape odds in 16.16 fixed point
(int array 2×2×11×21 for order 2). R is estimated once:
- build a table on 80 training books (every 9th training book held out);
- walk the PPM contexts over the first 400,000 characters of each of the 10 held-out books, counting visits and escapes per key;
- rate = (escapes + 0.5) / (visits + 1). Buckets with fewer than 20 visits use the pool summed over the n bucket,
  then over the d and n buckets, and finally 0.5;
- clip the rate to [0.0001, 0.94]; R = round(65536 · rate / (1 − rate)).

**sp2 spelling:** bytes of the UNK token, then end symbol 256; contexts start at (256, 256).
`f1[b] = spell1[p1][b] + 1` (the v1 order-1 counts), `f1s = floor(f1 · 4096 / sum f1) + 1`. If the order-2
context (p2, p1) was seen: `f = c2 · sum(f1s) + T2 · f1s`, with c2 = order-2 counts over the same rare training tokens
and T2 = the number of nonzero entries in c2. Otherwise `f = f1s`.

**Order-k context ids** (`o3`/`o4`, not recommended): order 1 = the previous token id; the order-(k+1) context id =
the index of `(order-k context id << b) | token` in the sorted order-k key array. Unseen means stop going higher.

## Reproduce from scratch

Requires the corpus in `corpus/gutenberg/` (see `RESEARCH.md` / `lab/fetch_corpus.py`). Run from the project root.
Tables are built automatically on first use into `tables/exp_model_*` (~1-2 min each for order 2, ~3-5 min for
o3/o4; SEE ~1 min; sp2 ~1 min). Peak RAM ~3 GB while building; ~0.4 GB when only loading.

```sh
uv run python lab/exp_model/run.py base                 # builds tables/exp_model_o2.npz; equals v1
for v in case case2 wrap o3 o4 D X S sp2 wrap+case2+sp2+S; do
  uv run python lab/exp_model/run.py $v                  # ~10 min each; writes lab/exp_model/results/$v.json
done
uv run python lab/exp_model/summarize.py                 # the results table above
uv run python lab/exp_model/breakdown.py wrap+case2+sp2+S 10000   # bits per category (ideal)
uv run python lab/exp_model/breakdown.py o3+m16 10000             # min-count pruning checks
uv run python lab/exp_model/run.py VARIANT --quick       # 15 × 1 KB smoke test
```

Variant names combine with `+`: `case`, `case2`, `wrap`, `sp2`, `o2`/`o3`/`o4`, `C`/`D`/`X`/`S`, `m<N>`.

## Files

- `model.py`: generalized codec (events, state, table build/load, PPM, SEE, sp2, container).
- `run.py`: evaluation with round trips → `results/*.json`.
- `breakdown.py`: ideal bits per category. `summarize.py`: the results table.
- Tables: `tables/exp_model_<table>.npz`, `_see.npy`, `_sp2.npz` (generated, not tracked).
