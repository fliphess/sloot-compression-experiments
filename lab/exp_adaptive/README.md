# exp_adaptive: learning from the document while coding it (E8b)

**Question:** the frozen table knows the language, but not this document's names and phrases. Does it help if the coder
also learns from the document as it goes (the decoder does exactly the same, so nothing extra is stored)?

**Verdict: ✅ adopted.** Config **`d3-sum-w32`** is the best table coder. It's used by `overnight/`, `exp_chain/` and the C port (`c/`).

## Key result (real bytes, round trips verified incl. the 4.3 MB Bible)

| test | v1 static | **d3-sum-w32** |
|---|---|---|
| 10 full books | 2.420 bpc | **1.911 bpc** (-21%; xz: 2.306) |
| 1 KB / 30 KB snippets | 285 / 8,207 B | 276 / 7,484 B |

The weights were later validated on 10 fresh books (`overnight/`): w8/w32/w64 differ by < 0.5%.

## How it works

The table stays frozen. A per-document layer counts what this document has shown so far:
- first a document-only context of the last 3 tokens (`d3`);
- then the table's contexts of 2, 1 and 0 tokens, each with the document's own counts added at weight 32 (`sum-w32`);
- a word not in the table is spelled once, then gets its own id, so later occurrences are cheap.

What didn't work: weight 1 (far too weak), and document-only contexts placed *in front of* the table's 1- and 2-word levels.

## Run

```sh
uv run python lab/exp_adaptive/run.py short                  # all configs on 200 snippets (~25 min)
uv run python lab/exp_adaptive/run.py books d3-sum-w32       # full books (~20 min)
```

Files: `adaptive.py` (codec with the adaptive layer), `run.py` (configs + evaluation), `results/`. Details: [RESULTS.md](RESULTS.md).
