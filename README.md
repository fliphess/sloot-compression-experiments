# sloot-compression-experiments

A Friday-evening hobby experiment: what could Jan Sloot's legendary "film on a chip card" compression have been, and how far can
you get in 2026 with shared tables, small neural networks and deterministic arithmetic coding? Done for fun and to learn; not research.

**Status: concluded.** Short answer: shared knowledge makes text 1.5-2.9× smaller than gzip, but on paper the knowledge itself has to be
printed too, which makes the approach unrealistic for its original goal. Along the way: bit-identical integer-only neural inference.

## Where to start

1. **[WRITEUP.md](WRITEUP.md)**: the story in plain words: what we tried, what worked, what didn't, and why.
2. **[RESEARCH.md](RESEARCH.md)**: the detailed log, with every decision and every number, in order.
3. **[lab/README.md](lab/README.md)**: a map of all code and experiments, each with its verdict. Every experiment directory has its own README.

## Try it without training anything

The [v1.0 release](https://github.com/fliphess/sloot-compression-experiments/releases/tag/v1.0) has the frozen files, so you can
compress and decompress right away (numpy only, no PyTorch or GPU needed). Put them in `tables/`:

```sh
uv sync
uv run python lab/round2/intmodel/nncodec.py compress   tables/intmodel_20m_mixed.bin IN OUT   # neural, integer-only (slow: ~0.2 KB/s decode)
uv run python lab/round2/intmodel/nncodec.py decompress tables/intmodel_20m_mixed.bin OUT BACK
uv run python lab/codec.py compress   IN OUT                                                    # table codec, needs tables/en-gutenberg-v1.npz
uv run python lab/codec.py decompress OUT BACK
```

## Layout

| path | in git | contents |
|---|---|---|
| `WRITEUP.md`, `RESEARCH.md` | yes | the story and the detailed log |
| `LICENSE` | yes | MIT (code); see the release notes for the data behind the models |
| `pyproject.toml`, `uv.lock` | yes | Python dependencies (numpy; PyTorch as the optional `neural` extra) |
| `lab/` | yes | all code, experiment results and logs |
| `lab/corpus_manifest.json`, `lab/exp_table/manifest.json` | yes | SHA-256 of every corpus file, so a rebuilt corpus can be checked |
| `corpus/` | no | downloaded and derived data (~3 GB), rebuilt by script |
| `tables/` | no | frozen tables and models (several GB), rebuilt by script |

## Reproduce

Requirements: [uv](https://docs.astral.sh/uv/), the command-line compressors `gzip bzip2 xz zstd brotli` (e.g. `brew install xz zstd brotli`)
and a C compiler.

```sh
uv sync                                  # Python env with numpy (add --extra neural for PyTorch)
uv run python lab/fetch_corpus.py        # base corpus: enwik8, 100 Gutenberg books, samples (~200 MB), checked against the manifest
uv run python lab/codec.py build         # the v1 table, tables/en-gutenberg-v1.npz (fingerprint 1cd43ea2)
uv run python lab/codec.py test          # round trip + size report on 200 short documents
make -C lab/c                            # the C codec (see lab/c/README.md)
```

The neural experiments need `uv sync --extra neural`. Training used the Apple M4 GPU (MPS) and takes hours per model. The bigger
experiments (the 3 GB corpus, the neural models, v2) have their exact commands in their own READMEs. A few shell scripts call
`./.venv/bin/python` directly; that's the environment `uv sync` creates.

Sources drift: Project Gutenberg occasionally corrects a book. That's reported as a warning, and a table built from drifted sources gets a
different fingerprint, so numbers can differ slightly from the log. `uv run python lab/fetch_corpus.py --strict` requires an exact match.
