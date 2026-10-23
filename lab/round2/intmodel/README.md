# round2/intmodel: integer-only neural inference (E12)

**Question:** an archive must decode bit-identically on any machine, forever. Floating point doesn't guarantee that.
Can the neural model run with integers only, and still compress as well?

**Verdict: ✅ works.** Real files, all 200 snippets round-trip losslessly, and the integer version costs only +0.07-0.11% (20M)
over the fp32 ideal. **Limitation:** numpy speed (encode ~0.5 KB/s, decode 0.1-0.3 KB/s); a C/NEON port is needed for practical use.

## Key result (real files, 200/200 round trips per model)

| model | 1 KB | 30 KB | vs fp32 ideal |
|---|---|---|---|
| 5M | 256.4 B | 7,317 B | +0.4% |
| 20M (E10, 85 MB) | 238.5 B | 6,729 B | +0.07-0.11% |
| 20M_big (3 GB) | running | | |

Determinism: identical output across the `int`/`f32`/`f64` backends and thread settings (on subsets; see RESULTS.md).

## How it works

- int8 weights with an integer multiplier + shift per channel, int64 accumulators, and integer LayerNorm, softmax and GELU.
- An integer rule turns logits into coder frequencies (every frequency ≥ 1, total < 2^30).
- Float BLAS may be used only as an exact accelerator, where every term is exactly representable. Each model file
  carries a self-test hash of the integer logits for a fixed input, and the codec refuses a backend that doesn't reproduce it.
- Model files: `tables/intmodel_*.bin`, with fingerprints.

## Run (project venv)

```sh
./.venv/bin/python lab/round2/intmodel/export.py tables/exp_tinymodel_20m.pt 20m
./.venv/bin/python lab/round2/intmodel/nncodec.py compress tables/intmodel_20m.bin IN OUT [--backend int|f32|f64]
./.venv/bin/python lab/round2/intmodel/nncodec.py decompress tables/intmodel_20m.bin IN OUT
VECLIB_MAXIMUM_THREADS=1 ./.venv/bin/python lab/round2/intmodel/run_eval.py 20m f32 0 1    # 200 snippets
./.venv/bin/python lab/round2/intmodel/summarize.py
bash lab/round2/intmodel/det.sh                                                             # determinism checks
```

Files: `export.py`, `intlm.py` (integer model), `nncodec.py` (codec + container), `run_eval.py`, `run_books.py`, `summarize.py`,
`det.sh`/`det2.sh`. Details: [RESULTS.md](RESULTS.md). Note: RESULTS.md was written by an agent that was stopped before
it finished; its "Results" section is incomplete. `summarize.py` prints the current numbers.
