# rjbs-compression-experiments

Research into a modern, honest version of the
[Sloot Digital Coding System](https://en.wikipedia.org/wiki/Sloot_Digital_Coding_System): extreme
compression for long-term archival storage, driven by one large, frozen, shared reference table.

**Start with [RESEARCH.md](RESEARCH.md)**: the goal, the decisions made, every experiment with its results, and what's next.

## Layout

| path | in git | contents |
|---|---|---|
| `RESEARCH.md` | yes | research log and handover |
| `prompt.txt` | yes | the original brief |
| `lab/` | yes | all code: experiments, codec, tooling |
| `lab/corpus_manifest.json` | yes | SHA-256 of every corpus file, so a rebuilt corpus is verifiably identical |
| `lab/exp_*/` | yes | parallel experiments, each with its own `RESULTS.md` |
| `corpus/` | no | downloaded and derived data, rebuilt by `lab/fetch_corpus.py` |
| `tables/` | no | frozen reference tables, rebuilt by the build commands |

## Reproduce

Requirements: Python 3.12+ with numpy (`pip install -r requirements.txt`), plus the command-line
compressors for the baselines: `gzip bzip2 xz zstd brotli` (e.g. `brew install xz zstd brotli`).

```sh
python3 lab/fetch_corpus.py          # download + build the corpus, verify every file (~200 MB, a few minutes)
python3 lab/codec.py build           # build the frozen table tables/en-gutenberg-v1.npz (fingerprint 1cd43ea2)
python3 lab/codec.py test            # round trip + size report on 200 short documents
```

Each experiment's script is listed with its results in `RESEARCH.md`. `python3 lab/fetch_corpus.py verify`
checks an existing corpus.

The deliverable is the *method* to build a table, not one specific table. Sources drift (Project Gutenberg
occasionally corrects its texts); that's reported as a warning, and a table built from drifted sources gets a
different fingerprint, so results can differ slightly from the log. Use `--strict` to require an exact match.
