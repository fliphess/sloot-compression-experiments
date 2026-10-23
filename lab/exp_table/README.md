# exp_table: how far does a bigger table get us? (E8c)

**Question:** codec v1's table is built from 85 MB of text. What happens with 300 MB, 1 GB, 3 GB? Is the "one 10 GB table
to rule them all" worth it?

**Verdict: ✅ bigger helps, with clearly diminishing returns.** With this model (2 words of context) a 10 GB table isn't worth it:
the curve flattens at ~231 B per 1 KB. The 3 GB table is the best *table* we have. Pruning (dropping n-grams seen once)
is a cheap way to shrink tables. The chunked builder and the leakage control are reusable tools.

## Key result (codec v1 algorithm unchanged, real bytes, all snippets round-trip verified)

| table | text | on disk | 1 KB doc | books bpc |
|---|---|---|---|---|
| v1 | 85 MB | 31 MB | 285 B | 2.420 |
| 300 MB | 300 MB | 94 MB | 262 B | 2.209 |
| 1 GB | 1 GB | 240 MB | 249 B | 2.095 |
| **3 GB** | 3 GB | 545 MB | **242 B** | **2.019** |
| 1 GB pruned | 1 GB | 68 MB | 255 B | 2.154 |
| 3 GB pruned | 3 GB | 162 MB | 247 B | 2.066 |

Each tripling of the text gains less: -8%, then -5%, then -3%. Pruning removes ~73% of a table for +2.5-3% output.
More text + pruning beats less text kept whole.

## How it works

- `fetch_big.py`: downloads English Gutenberg books from the pglaf mirror, politely; `--from-manifest` re-fetches exactly the listed books.
- `leakage.py`: drops books that duplicate a test book: other editions (title/author), shared 8-word shingles, excerpts. 85 of 8,443 were dropped.
- `build_table.py`: an external, chunked table builder (peak 2.5-4.6 GB for 3 GB of text), bit-identical to `codec.FrozenTable.build`.
- `evaluate.py`: real-byte evaluation; `summarize.py`: tables + the scaling fit.

## Run

See "Reproduce from scratch" in [RESULTS.md](RESULTS.md). In short:

```sh
python3 lab/exp_table/fetch_big.py --from-manifest
python3 lab/exp_table/build_table.py --mb 3000 --out tables/exp_table_3gb.npz
python3 lab/exp_table/evaluate.py tables/exp_table_3gb.npz
```

`manifest.json` (book IDs + hashes) and `leakage.tsv` are tracked; the books and tables aren't.
