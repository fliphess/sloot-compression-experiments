# overnight: combining bigger tables with document learning (E8b × E8c)

**Question:** E8c made the table bigger, E8b made the coder learn from the document. Do the two gains stack? And are E8b's
weights still right on fresh data, and on a big table?

**Verdict: ✅ yes, they stack, and the weights are robust.** 3 GB table + `d3-sum-w32` is the best *table-based* codec we have.

## Key result (real bytes, every round trip verified)

| table + document learning | 1 KB | 30 KB | books bpc |
|---|---|---|---|
| v1 (31 MB) | 276 B | 7,484 B | 1.911 |
| 300 MB pruned (26 MB) | 263 B | 7,240 B | 1.884 |
| 1 GB pruned (68 MB) | 251 B | 6,990 B | 1.834 |
| 1 GB (240 MB) | 245 B | 6,822 B | 1.784 |
| 3 GB pruned (162 MB) | 244 B | 6,807 B | 1.798 |
| **3 GB (545 MB)** | **239 B** | **6,672 B** | **1.755** |

For scale: gzip needs 588 B for the same 1 KB; xz reaches 2.306 bpc on the books.
Weight validation on 10 fresh books (v1 table): w8 / w32 / w64 within 0.5% of each other.

## Run

```sh
uv run python lab/overnight/combo.py run exp_table_3gb               # a table + d3-sum-w32: snippets + books
uv run python lab/overnight/combo.py sweep en-gutenberg-v1 val d3-sum-w8 d3-sum-w32 d3-sum-w64   # weight sweep, fresh books
uv run python lab/overnight/combo.py summary
```

Files: `combo.py`, `results/*.json`, `logs/`.
