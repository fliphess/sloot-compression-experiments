# c: C port of the table codec (E11)

**Question:** Python runs at ~30 KB/s. Can the codec (v1 static + adaptive `d3-sum-w32`) be ported to plain C, byte for byte
identical, so it's fast and portable?

**Verdict: ✅ works, bit-identical, 20-250× faster.** It's the candidate reference implementation for the table codec.
**Not yet code-reviewed**: one 54 KB file written for correctness and speed, not for readability. Review and split it into
modules before it becomes the official reference.

## Key result

- **Bit-identical to Python:** 216 inputs (edge cases, 200 snippets, 10 books) × 2 tables (v1, 3 GB) × static/adaptive, and C decodes Python files.
- **Speed:** static 2-4 MB/s, adaptive 0.5-1.6 MB/s (Python: 0.01-0.06 MB/s). The table opens via mmap in 0.2 ms (Python: up to 7 s).
- **Cost:** converted tables are large (uncompressed + precomputed helpers): v1 238 MB, 3 GB table 4.1 GB.
- **Found spec gaps**, listed in `RESEARCH.md` (E11). The important one: mode byte 2 is used by two different models.

## Run

```sh
make -C lab/c                                                   # builds lab/c/steno (no dependencies)
python3 lab/c/convert_table.py tables/en-gutenberg-v1.npz tables/c_en-gutenberg-v1.bin
lab/c/steno compress -a tables/c_en-gutenberg-v1.bin IN OUT     # -s static (default), -a adaptive
lab/c/steno decompress tables/c_en-gutenberg-v1.bin IN OUT
lab/c/steno verify tables/c_en-gutenberg-v1.bin                 # recompute the fingerprint in C
python3 lab/c/ref.py inputs && python3 lab/c/ref.py encode v1 --procs 3   # Python reference outputs (slow)
python3 lab/c/test.py v1 [static|adaptive]                      # bit-identity vs Python (v1 or 3gb)
python3 lab/c/bench.py c                                        # C speed; `bench.py py v1` for Python
```

Files: `steno.c`, `Makefile`, `convert_table.py` (table format in its docstring), `ref.py` (Python reference outputs),
`test.py`, `bench.py`, `results/`. Test outputs go to `data/` (gitignored). Details: [RESULTS.md](RESULTS.md).
