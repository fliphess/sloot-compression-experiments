# E11: C port of the steno codec (static v1 + adaptive d3-sum-w32), bit-identical to Python

Status (2026-10-23): **done**. Over 864 test cases (216 inputs × 2 tables × 2 modes), the C encoder's output
is byte-identical to the Python encoder's, and C decode(C encode(x)) == x in every case. The C SHA-256
recomputation of both table fingerprints matches the Python FrozenTable value.

## Files

| file | what |
|---|---|
| `lab/c/convert_table.py` | `.npz` FrozenTable -> flat mmap-able `STNCTAB1` file (full format in its docstring) |
| `lab/c/steno.c`, `lab/c/Makefile` | C11 codec, 1384 lines, no dependencies (libc + POSIX mmap): compress / decompress / info / verify |
| `lab/c/ref.py` | writes the test inputs (`data/inputs/`, `data/manifest.json`) and the Python reference encoder outputs (`data/py/`) |
| `lab/c/test.py` | harness: C encode vs Python encode byte for byte, then C decode(C encode) == input; writes `results/<table>_<mode>.json`, `data/c/*/times.tsv` |
| `lab/c/bench.py` | speed, table load time, peak RSS; writes `results/bench.json` |
| `lab/c/logs/` | logs of all runs (conversion, reference, tests, benchmarks) |
| `tables/c_en-gutenberg-v1.bin` (238 MB), `tables/c_exp_table_3gb.bin` (4.1 GB) | converted tables |

## Commands

```
make -C lab/c                                                    # cc -std=c11 -O2 -Wall -Wextra -pedantic: no warnings
uv run python lab/c/convert_table.py tables/en-gutenberg-v1.npz tables/c_en-gutenberg-v1.bin      # 6 s
uv run python lab/c/convert_table.py tables/exp_table_3gb.npz  tables/c_exp_table_3gb.bin         # 120 s, 3.1 GB peak RSS
lab/c/steno info   TABLE                                         # header, open+mmap time
lab/c/steno verify TABLE                                         # recompute the SHA-256 fingerprint in C (1 s / 15 s)
lab/c/steno compress [-s|-a] [-u] [-v] TABLE IN OUT [IN OUT ...] # -s static (mode 1, default), -a adaptive d3-sum-w32 (mode 2)
lab/c/steno decompress [-u] [-v] TABLE IN OUT [IN OUT ...]       # mode taken from byte 0 (0 stored, 1 static, 2 adaptive)
uv run python lab/c/ref.py inputs && uv run python lab/c/ref.py encode v1|3gb --procs N
uv run python lab/c/test.py v1|3gb [static|adaptive]
uv run python lab/c/bench.py c ; uv run python lab/c/bench.py py v1|3gb
```

`-u` = the E9 chain variant (`lab/exp_chain/chain.py`, `count_unk=True`: every spelled word also counts as UNK
at order 0). `-v` = per-file coding time on stderr (excludes process start and table mapping).

## Converted table format (`STNCTAB1`, version 1)

All integers are little-endian; every section starts on a 64-byte boundary. Header: 512 bytes. It holds the
magic, the version, the fingerprint (bytes 16-19), `V`, `b`, `total0`, the context/entry counts, `blob_len`,
`hash_slots`, `n_sections` = 22, a flags word (bit 0 = "nested", see below) and `c2_hash_bits`. At byte 128
comes a table of 22 × (u64 offset, u64 length). Sections, in order:

| section | type | count | |
|---|---|---|---|
| vocab_blob | u8 | blob_len | fingerprint array |
| vocab_lens | u32 | V | fingerprint array |
| vocab_off | u64 | V+1 | helper: byte offset of each token |
| uni | u32 | V | fingerprint array |
| uni_cum | u64 | V | helper: inclusive prefix sum |
| spell | u32 | 257² | fingerprint array |
| spell_cum | u32 | 257² | helper: per row, inclusive prefix of (spell + 1) |
| c1_ctx, c1_start, c1_sym, c1_cnt | i64, u32, u32, u32 | | fingerprint arrays, as in the npz |
| c1_cum | u32 | n_ent1 | helper: inclusive prefix of cnt within each context |
| c2_ctx, c2_start, c2_sym, c2_cnt | i64, u32, u32, u32 | | fingerprint arrays |
| c2_cum | u32 | n_ent2 | helper |
| vocab_hash | u32 | 2^k ≥ 2V | helper: token bytes -> id, FNV-1a-64, linear probing, 0 = empty |
| c1_unicum | u32 | n_ent1 | helper: inclusive prefix of uni[sym] within each order-1 context |
| c2_c1cum | u32 | n_ent2 | helper: inclusive prefix, over an order-2 context's followers, of their order-1 counts in the context of its last token |
| c1_index | u32 | V | helper: token id -> order-1 context index + 1 |
| c2_hash | u32 | 2^bits + spill | helper: order-2 context key -> index + 1 (multiplicative hash, linear probing upward, no wrap) |

Fingerprint arrays: 116 MB (v1) and 1977 MB (3 GB). Helpers: 122 MB and 2118 MB. Without the fingerprint
check, `c*_cnt` could be dropped (derivable from `c*_cum`): −32 MB / −770 MB.

The helpers make the hot path O(log n) instead of O(context size):
- **Order-1 exclusion of a whole order-2 context.** The converter checks that every order-2 follower (a, b) -> s
  is also an order-1 follower b -> s. This holds for 100.000000% of entries in both tables, as it must for
  tables built from one token stream, and the converter records it in the "nested" flag. With the flag set,
  the order-1 level removes the order-2 symbols virtually through `c2_c1cum`. Without it, a general path
  materializes the exclusions; that path is also bit-identical (tested with `STENO_NO_NESTED=1`: 140 snippets
  plus book 408).
- **Order-0 exclusion of a whole order-1 context** via `c1_unicum`.
- Context lookup through a hash or direct index instead of a binary search over 2.4M / 33.6M sorted keys.
  On this machine a binary search miss costs about 100 ns, so this halved the per-token time.

## Bit-identity results

Inputs (`ref.py inputs`):
- the 6 edge cases of `codec.py cmd_test`: empty, one char, Unicode, binary, random, unknown words;
- the 200 snippets of `short_docs.py` (4 sizes × 10 test books × 5);
- the 10 full test books.

Python references:
- static: `codec.compress` for the snippets and edge cases; for the full books
  `exp_table/evaluate.compress_fast`, which is codec v1 with a faster order 0 and was asserted identical in E8c;
- adaptive: `adaptive.compress` with `run.make_configs()["d3-sum-w32"]`;
- the 3 GB table is loaded as a LeanTable: uint32 arrays, the same coding, and a fingerprint computed in slices
  that equals FrozenTable's.

| table | mode | edge | snip 1K | snip 3K | snip 10K | snip 30K | 10 books | C round trip |
|---|---|---|---|---|---|---|---|---|
| v1 (1cd43ea2) | static (mode 1) | 6/6 | 50/50 | 50/50 | 50/50 | 50/50 | 10/10 | 216/216 |
| v1 | adaptive (mode 2) | 6/6 | 50/50 | 50/50 | 50/50 | 50/50 | 10/10 | 216/216 |
| 3 GB (9330c838) | static | 6/6 | 50/50 | 50/50 | 50/50 | 50/50 | 10/10 | 216/216 |
| 3 GB | adaptive | 6/6 | 50/50 | 50/50 | 50/50 | 50/50 | 10/10 | 216/216 |

(cells: C output byte-identical to the Python encoder output / total)

Sizes match the earlier experiments exactly:
- v1: 10 books 2.420 bpc static and 1.911 bpc adaptive (E7 and E8b); 1952 = 7,318 B static and 7,208 B adaptive.
- 3 GB: 2.019 bpc static and 1.755 bpc adaptive. Book 408 = 85,558 B static and 85,504 B adaptive, the same
  as `exp_table` and `overnight`.

Also checked:
- C decodes a Python-written file (book 1952, mode 1). All C files are byte-identical to the Python files, so
  C decoding all of its own files covers decoding the Python files too.
- A wrong fingerprint is refused: "made with table 00000000, this table is 1cd43ea2".
- `steno verify` recomputes the fingerprint in C over the converted file: 1cd43ea2 OK, 9330c838 OK.
- `-u` (chain `count_unk` variant) against `chain.compress_doc(table, d3-sum-w32, text, True)`: 25/25 snippets
  (20 × 3K, 5 × 30K) identical and round-trip OK. That is all that was tested for `-u`: no books, and no T0 or
  W=1/W=8 chain configs.
- Not verified: Python decoding of C output (not needed: the bytes are identical), and inputs beyond the test set.

## Speed, load time, memory

Apple M4. The machine was shared with a GPU training run and a CPU mixing job (load average 5-8), so the
numbers carry noise. The C numbers are the best of 2 runs (run 1 sometimes pages the table in from disk) and
cover coding only, from `-v`. Python encode times come from the reference run: 1 process for v1, 2 parallel
processes for 3 GB. "Py dec" is `codec.decompress` / `adaptive.decompress` on 10 × 30 KB snippets + book 1952.

| table | mode | set | C enc MB/s | C dec MB/s | Py enc MB/s | C/Py enc | C peak RSS MB | C footprint MB |
|---|---|---|---|---|---|---|---|---|
| v1 | static | 50 × 1 KB | 1.83 | 1.57 | 0.035 | 53× | 183 | 4 |
| v1 | static | 50 × 30 KB | 3.79 | 2.68 | 0.027 | 141× | 201 | 6 |
| v1 | static | 10 books (12.2 MB) | 3.97 | 2.82 | 0.058 | 69× | 211 | 16 |
| v1 | adaptive | 50 × 1 KB | 0.85 | 0.84 | 0.042 | 20× | 195 | 16 |
| v1 | adaptive | 50 × 30 KB | 1.59 | 1.33 | 0.034 | 47× | 215 | 20 |
| v1 | adaptive | 10 books | 0.70 | 0.66 | 0.031 | 23× | 517 | 268 |
| 3 GB | static | 50 × 1 KB | 0.77 | 0.71 | 0.006 | 137× | 629 | 12 |
| 3 GB | static | 50 × 30 KB | 1.80 | 1.63 | 0.007 | 247× | 2845 | 29 |
| 3 GB | static | 10 books | 2.37 | 1.95 | 0.012 | 191× | 3315 | 36 |
| 3 GB | adaptive | 50 × 1 KB | 0.45 | 0.42 | 0.007 | 70× | 693 | 39 |
| 3 GB | adaptive | 50 × 30 KB | 1.02 | 0.89 | 0.009 | 115× | 2886 | 58 |
| 3 GB | adaptive | 10 books | 0.54 | 0.51 | 0.010 | 52× | 3644 | 361 |

Python decode MB/s (sample above): v1 static 0.024, v1 adaptive 0.029, 3 GB static 0.011, 3 GB adaptive 0.016.
So C decodes 50-110× faster than Python.

- **Peak RSS** (`/usr/bin/time -l`, maximum resident set size) is mostly *file-backed, shared, clean* table
  pages that the process touched. They live in the page cache, can be reclaimed, and are shared between
  processes. The "footprint" column (peak memory footprint) is the process's own memory: the document layer,
  buffers. The adaptive layer for a 4.3 MB book takes ~270-360 MB (doc contexts, `V + 2^19` count arrays).
- **Table load.** C open + mmap + header check takes 0.23 ms (v1, 238 MB file) and 0.07-0.10 ms (3 GB table,
  4.1 GB file). A whole process that maps the table and adaptively codes one 1 KB snippet takes 16 ms (v1) or
  20 ms (3 GB) wall time, warm page cache. Cold (pages read from disk), the first 50 × 1 KB snippets take
  0.9-1.0 s in total instead of 0.03-0.06 s. Python loads in 0.56 s (v1 FrozenTable, 550 MB RSS) and 5-7 s
  (3 GB, LeanTable, 2.6 GB RSS); FrozenTable itself would need about 4-5 GB for the 3 GB table because it
  upcasts to int64.
- One run is excluded: in the 3 GB adaptive books row, the warm run was 2-5× slower than the first run under
  memory pressure from the other jobs (system-wide 5M pageouts), so the table shows the first run. Raw numbers
  are in `results/bench.json`.
- **Where the time goes.** Static mode is bound by memory latency: about 10 cache misses per token across the
  big arrays, about 1 µs per token. Adaptive is O(d) per token at the M levels, where d is the size of the
  document's own context. On the Bible the average d is 86 at M2 and 485 at M1. The order-0 visits (~6% of
  tokens) pay O(n1) to gather the document counts of the excluded order-1 context. If adaptive speed matters, the
  next step is per-context prefix structures for the document counts; this was not done.

## Spec notes: where `lab/exp_chain/METHOD.md` (draft 1) disagrees with or underspecifies the code

1. **§5.3 "UNK counting" is not in adaptive.py.** METHOD.md describes mode 2 as d3-sum-wW "with new-word ids
   and UNK counting enabled". Its learning step 1 does `doc_uni[0] += 1; doc_total += 1` for every spelled
   word. That is the E9 chain variant (`chain.ChainDoc`, `count_unk=True`). `adaptive.py`'s d3-sum-w32 (E8b,
   the recommended config) does **not** count UNK. Both write mode byte 2 with the same container, so a mode-2
   file does not say which model it needs: the same bytes decode differently. The C port implements both (`-u`).
   Fix: give the chain variant its own mode byte, or fix one variant in the spec.
2. **Stored fallback.** METHOD §6 says the encoder uses mode 0 when the coded file would not be smaller. The
   chain's `compress_doc` never falls back. Seed blobs are always mode 2, which is fine, but the spec should say so.
3. **Total limit: `<` or `≤`.** The `codec.py` docstring says frequency totals `< 2^30`. The code asserts
   `total <= MAX_TOTAL = 2^30`, and METHOD §4 also says `≤ 2^30`. The code is the reference, so it's `≤`. On
   overflow (an adaptive document of roughly 2^30 / W tokens) Python raises an `AssertionError` in the
   encoder; the decoder has no check. C stops with an error in both directions.
4. **UTF-8 validity is unspecified in detail.** "Valid UTF-8" means exactly what Python's strict
   `bytes.decode("utf-8")` accepts. It rejects overlongs, surrogates (ED A0-BF ..), code points > U+10FFFF and
   truncated sequences, and it accepts U+0000, noncharacters and a BOM. A BOM is kept as an ordinary
   one-code-point token, not stripped. C implements exactly this.
5. **MAX_NEW = 2^19 new words per document.** METHOD §8 mentions it, but past it Python fails with an
   `IndexError` (an undefined failure). C stops with an explicit error.
6. **Order-0 decoding.** METHOD only says "the symbol whose interval contains target". The reference uses a
   binary search over `0 … V+new−1` for the smallest `w` with `cum_before(w+1) > target`. C reproduces it
   exactly. Any correct search gives the same result, because zero-width (excluded) symbols can never be the
   smallest such `w`.
7. **The nesting property** (order-2 followers ⊆ order-1 followers of the last token) isn't stated in METHOD
   §3, but it follows from building both orders from one stream. Fast implementations (like this one) can use
   it; with a "nested" flag in a derived format and a general fallback, nothing depends on it silently.
