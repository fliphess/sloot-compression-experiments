# lab/

All code of the research. Run everything from the project root. `RESEARCH.md` has the full story and every result;
this file is the map.

**Verdicts:** ✅ in use (a building block for later work) · ⚠️ superseded (kept for the record, still informative) ·
❌ dead end (measured, doesn't work) · ⏸ postponed.

## Scripts in this directory

| script | exp | what it does | verdict |
|---|---|---|---|
| `fetch_corpus.py` | tooling | downloads + rebuilds `corpus/` and checks it against `corpus_manifest.json` | ✅ |
| `fetch_gutenberg.py` | tooling | downloads/strips Project Gutenberg books (`download_book()` is used by others) | ✅ |
| `baselines.py` | E1 | gzip/bzip2/xz/zstd/brotli on the samples: the numbers to beat | ✅ reference |
| `repair.py` | E2 | the "recursive pattern table" idea (Re-Pair): replace frequent pairs with table entries, round after round | ❌ converges early and loses to gzip |
| `paper.py` | E3 | how many bytes fit on an A4 page per printer/scanner/camera | ⏸ paper layer postponed |
| `reference.py` | E4 | zstd against a shared reference file; shows the "Sloot pointer" effect (1 MB → 125 B if it's *in* the reference) | ⚠️ insight, superseded by statistical tables |
| `steno.py` | E5 | word n-gram "steno" model, **estimated** bits; also defines `TOKEN` (tokenizer) and `split_books()` (train/test split) | ⚠️ the estimator is superseded by `codec.py`, but `TOKEN`/`split_books` are used everywhere |
| `short_docs.py` | E6 | the short-document test set (200 snippets of 1/3/10/30 KB) + comparison with existing tools | ✅ test set used by every experiment |
| `codec.py` | E7 | **codec v1**: real encoder/decoder, frozen table, PPM, integer arithmetic coder, container | ✅ core, imported by almost everything |
| `eval_books.py` | E7 | codec v1 on the 10 full test books | ✅ baseline numbers |
| `corpus_manifest.json` | tooling | SHA-256 of every corpus file | ✅ |

## Experiment directories

| directory | exp | question | verdict | headline |
|---|---|---|---|---|
| [`exp_model/`](exp_model/) | E8a | do model refinements (case, line breaks, longer contexts, escapes) help? | ⚠️ partly adopted | -5.4% (mostly Gutenberg line breaks); longer contexts worse |
| [`exp_adaptive/`](exp_adaptive/) | E8b | does learning from the document itself help? | ✅ adopted | books -21% (2.42 → 1.91 bpc) |
| [`exp_table/`](exp_table/) | E8c | how does a bigger table (85 MB → 3 GB of text) scale? | ✅ | -15% with diminishing returns; 10 GB isn't worth it with this model |
| [`overnight/`](overnight/) | E8b × E8c | do bigger tables and document learning stack? | ✅ | yes: 239 B per 1 KB (gzip 588 B) |
| [`exp_chain/`](exp_chain/) | E9 | can the table be shipped as a compressed seed corpus and rebuilt identically? | ✅ proven | seed 19 MB < table 31 MB; fingerprint identical. `METHOD.md` = spec draft |
| [`exp_tinymodel/`](exp_tinymodel/) | E10 | could a small self-trained neural model beat the table? | ⚠️ superseded by `round2/` | a 19 MB model matches the 545 MB table |
| [`c/`](c/) | E11 | C port of the codec, bit-identical to Python | ✅ | 20-250× faster; needs a code review |
| [`round2/`](round2/) | R2, E12 | model trained on 3 GB; mixing; integer-only inference | ✅ current best | 207 B per 1 KB (2.8× smaller than gzip) |

## Dependencies

`steno.py` (tokenizer, split) and `short_docs.py` (test set) are imported by everything. `codec.py` is the base of the
table codecs (`exp_adaptive`, `exp_table`, `exp_chain`, `overnight`, `c/ref.py`). `exp_tinymodel/common.py` defines
the neural model and is used by `round2/`. Large outputs always go to `tables/` (gitignored), never into `lab/`.
