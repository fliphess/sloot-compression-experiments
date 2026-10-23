# E8c: larger reference tables, results

Question: how does codec v1 (`lab/codec.py`: PPM-C with exclusion, orders 2/1/0, same tokenizer
and container) improve as the frozen table is built from more text? The algorithm is unchanged;
only the training data, and therefore the table, varies. All numbers are **real compressed
bytes** from the codec, on the fixed E7 test set: 200 short snippets (`short_docs.snippets`,
5 per test book per size) and the 10 full test books (`steno.split_books`).

Status: **done**. Tiers 85 MB / 300 MB / 1 GB / 3 GB (+ pruned variants); the final table incl. 3 GB and the 4-point fit
are in `results/summary.md` (and summarised in RESEARCH.md, E8c).

## Data

- Source: English Project Gutenberg books (`pg_catalog.csv`: Type = Text, Language = en; 61,766
  candidates), downloaded from the `gutenberg.pglaf.org` mirror. Rate limit: max 1.4 requests/s
  over 5-8 workers, descriptive User-Agent, resumable. Licence header/footer stripped as in
  `lab/fetch_gutenberg.py`. Books skipped if < 2 KB, > 15 MB, or not prose (word-length filter).
- Order: seeded shuffle (seed 20261022) of the candidates, so tiers are nested prefixes:
  tier N = the 90 codec-v1 training books, then manifest books in order, minus exclusions, until N MB.
- Downloaded: 8,410 books, 3,009 MB (download stopped at the 3 GB target).
- **Leakage control** (`leakage.py`, results per book in `leakage.tsv`). A book is excluded if any of:
  - **content:** > 1% of a test book's distinct 8-word shingles occur in the candidate;
  - **excerpt:** > 5% of the candidate's own shingles come from one test book (catches single chapters/excerpts);
  - **title/author:** matches a test book (other editions or parts of the same work; any title containing "Bible").

  Up to the 1 GB tier: 41 of 3,421 checked books excluded (mostly Bible editions and Bible-heavy
  religious works, plus e.g. a Melville biography quoting Moby Dick at 1.1%). None of the 90
  codec-v1 training books trips any check; the highest is Hobbes' *Leviathan* at 0.27% of the Bible.
  Final counts for 3 GB: see `manifest.json` → `excluded` after the pending run.

## Results

Short docs: average real bytes per document (bpc). Books: total bpc over the 10 full test books.
v1 = the 90-book tier; my builder reproduces `tables/en-gutenberg-v1.npz` bit for bit (same fingerprint `1cd43ea2`).

| tier | train MB | books | tokens | vocab | o1 + o2 entries | prune | disk MB | 1 KB | 3 KB | 10 KB | 30 KB | books bpc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 (85 MB) | 85 | 90 | 24M | 136,525 | 2.4M + 7.9M | - | 31 | 285.3 B (2.271) | 837.2 (2.220) | 2,756.1 (2.192) | 8,207.4 (2.175) | 2.420 |
| 300 MB | 300 | 705 | 79M | 299,050 | 6.9M + 25.3M | - | 94 | 262.1 B (2.087) | 768.9 (2.039) | 2,534.3 (2.016) | 7,527.3 (1.995) | 2.209 |
| 1 GB | 1,000 | 2,574 | 260M | 573,280 | 16.1M + 67.8M | - | 240 | **249.3 B (1.985)** | 734.5 (1.947) | 2,424.0 (1.928) | 7,204.2 (1.910) | **2.095** |
| 3 GB | 3,000 | 8,206 | 776M | 1,064,949 | 33.7M + 159.2M | - | 545 | 242.3 B (1.929) | 714.1 (1.893) | 2,354.3 (1.873) | 6,992.2 (1.853) | 2.019 |
| 300 MB pruned | 300 | 705 | 79M | 299,050 | 2.7M + 6.0M | ≥2 | 26 | 269.2 B (2.143) | 789.0 (2.092) | 2,602.8 (2.071) | 7,735.2 (2.050) | 2.275 |
| 1 GB pruned | 1,000 | 2,574 | 260M | 573,280 | 6.5M + 16.9M | ≥2 | 68 | 255.4 B (2.034) | 752.6 (1.995) | 2,486.7 (1.978) | 7,384.7 (1.958) | 2.154 |

Verification:
- All 200 short snippets round-trip byte-identically with every table, using the unchanged `codec.decompress`.
- Full books: all 10 encoded for real bytes. The 5 books ≤ 500 KB (46, 408, 1232, 1952, 20203) are also decoded and verified.
  The 5 larger books are **not decoded** (Python decode of multi-MB books is slow); their sizes are real encoder output.
- `evaluate.py` encodes with a faster order-0 step (O(exclusions) instead of O(vocab)). With
  `--check-identical` it produced the same bytes as `codec.compress` on all 200 snippets (v1 table).

Per-tier gain (unpruned):
- 85 → 300 MB (×3.5 text): **−8.1%** on 1 KB docs, −8.7% on books.
- 300 MB → 1 GB (×3.3): **−4.9%** on 1 KB docs, −5.2% on books.
- v1 → 1 GB overall: −12.6% on 1 KB docs (285 → 249 B), −12.2% on 30 KB, −13.4% on books (2.42 → 2.10 bpc).

**Pruning** (drop order-1/2 n-grams seen once) removes ~73% of entries and disk, at a cost of
+2.5-3% bytes. Per byte of table on disk, more text plus pruning beats less text kept whole:
the 1 GB pruned table (68 MB) beats the unpruned 300 MB table (94 MB), and the 300 MB pruned table (26 MB) beats v1 (31 MB).

## Scaling curve and extrapolation

Fit over the unpruned tiers: value = A + B · MB^-α (`summarize.py`). **With 3 points and 3
parameters the fit is exact, so the extrapolation is fragile; the 3 GB point will test it.**

| metric | asymptote A | α | fit at 10 GB text | fit at 30 GB text | measured 85 / 300 / 1000 MB |
|---|---|---|---|---|---|
| 1 KB doc | 231.0 B | 0.44 | 237.6 B | 235.1 B | 285.3 / 262.1 / 249.3 |
| 3 KB doc | 694.2 B | 0.51 | 706.6 B | 701.3 B | 837.2 / 768.9 / 734.5 |
| 10 KB doc | 2,298.7 B | 0.52 | 2,336.1 B | 2,319.7 B | 2,756.1 / 2,534.3 / 2,424.0 |
| 30 KB doc | 6,868.5 B | 0.56 | 6,960.9 B | 6,918.4 B | 8,207.4 / 7,527.3 / 7,204.2 |
| books bpc | 1.939 | 0.46 | 1.994 | 1.973 | 2.420 / 2.209 / 2.095 |

Table disk size grows ≈ MB^0.83 (unpruned). A 10 GB table would hold ≈ 90 GB of training
text at that rate: more English book text than Project Gutenberg has (≈ 20 GB in total).

**Honest extrapolation (preliminary):** with this model (2 words of context), more text gives
diminishing returns. The fit says the remaining gain beyond 1 GB is about 7% (1 KB docs: 249 → ~231 B
at infinity), and 10 GB of text gets ~5% of it. A 10 GB *table* is not the bottleneck: orders
2/1/0 can't use it. Making good use of a 10 GB table needs longer contexts (E8a), more varied text
(other domains), or both. This is preliminary until the 3 GB point is in.

## Engineering notes

- `build_table.py` reproduces `FrozenTable.build` exactly: same fingerprint for v1 (`1cd43ea2`),
  and stable fingerprints across builder rewrites (300 MB `abec13e6`, 1 GB `efdfa5ee`).
- Memory: the first in-memory builder peaked at 7.2 GB for 1 GB of text. The external builder
  writes sorted runs per key range to `tables/exp_table_tmp_*/` and merges one range at a time
  (contexts never straddle a range). It peaks at **2.7 GB for 1 GB of text**, with identical output.
  Pass 2 memory doesn't grow with the corpus.
- Limits of codec v1 tables: vocab < 2^21 (3 ids per int64 key) and total tokens < 2^30 (coder
  limit; 3 GB ≈ 0.8 G tokens fits). Beyond ~4 GB of text, counts need deterministic scaling.
- Speed: building is ~70 s per GB of text on 3 cores. Evaluation of a tier: ~5 min (Python codec).
- Tables are written as `.partial_*` and renamed when complete, so readers never see half a file.

## The 3 GB run (detached, finished 2026-10-22 23:31)

- Command: `nohup lab/exp_table/run_3gb.sh > lab/exp_table/logs/run_3gb.log 2>&1 &` (started 2026-10-22 23:05).
- Steps: leakage check of the new books → manifest → build `tables/exp_table_3gb.npz` → evaluate →
  `lab/exp_table/results/summary.md` (all tables + refit with 4 points). The log ends with `RUN_3GB_DONE`.
- Expected finish: ~23:35. Results: `lab/exp_table/results/exp_table_3gb.{build,eval}.json` and `results/summary.md`.

## Reproduce from scratch

`corpus/` and `tables/` are not tracked; `lab/exp_table/manifest.json` and `leakage.tsv` are.

```sh
uv run python lab/fetch_corpus.py                       # the 100 base books (main session's script)
mkdir -p corpus/gutenberg_big
curl -sSfL -o corpus/gutenberg_big/pg_catalog.csv https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv
                                                  # only needed for leakage title checks
uv run python lab/exp_table/fetch_big.py --from-manifest   # exactly the manifest books, sha256-verified
uv run python lab/exp_table/leakage.py --procs 3        # uses the cached leakage.tsv; checks only new books
for t in "300 300mb" "1000 1gb" "3000 3gb"; do set -- $t
  uv run python lab/exp_table/build_table.py --mb $1 --out tables/exp_table_$2.npz
  uv run python lab/exp_table/evaluate.py tables/exp_table_$2.npz
done
uv run python lab/exp_table/build_table.py --mb 300 --prune 2 --out tables/exp_table_300mb_p2.npz
uv run python lab/exp_table/build_table.py --mb 1000 --prune 2 --out tables/exp_table_1gb_p2.npz
uv run python lab/exp_table/summarize.py
```

To grow the corpus instead: `uv run python lab/exp_table/fetch_big.py --target-mb N`, then `leakage.py`, then `manifest.py`.
Expected fingerprints: v1 `1cd43ea2`, 300 MB `abec13e6`, 1 GB `efdfa5ee`, 300 MB pruned `29ac6e25`, 1 GB pruned `f8f5fc12`.
If Gutenberg has changed a text since, `--from-manifest` warns and the fingerprints will differ slightly.

## Files

- `fetch_big.py`: download / re-fetch from the manifest. `leakage.py`: leak checks. `manifest.py`: writes `manifest.json`.
- `build_table.py`: chunked external table builder. `evaluate.py`: real-byte evaluation. `summarize.py`: tables + fit.
- `results/*.build.json`, `results/*.eval.json`: raw numbers (the build JSON includes the exact book ID list per tier).
- `run_3gb.sh`, `logs/`: the detached 3 GB run.
