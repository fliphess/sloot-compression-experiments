# exp_chain: the bootstrap chain, or the table without shipping the table (E9)

**Question:** a frozen table is derived from a corpus by a fixed method. Can we preserve and distribute the *seed corpus*
(compressed) plus the method instead of the table, and have anyone rebuild the identical table? This is the useful form
of "recursion": knowledge built on knowledge, down to a human-readable base.

**Verdict: ✅ proven and fully verified.** The compressed seed is smaller than the table it produces, and the rebuild is bit-identical.
`METHOD.md` is the first human-readable spec of the whole method. The C port (`c/`) found some gaps in it; see `RESEARCH.md`, E11.

## Key result

| | size |
|---|---|
| v1 table (what it rebuilds) | 31.0 MB |
| seed corpus (84.7 MB of text), xz -9e | 21.1 MB |
| **seed, progressive chain** | **19.09 MB** (-38% vs the table) |
| seed, every book coded alone | 21.30 MB (any book decodes on its own) |

Verified in a fresh process: 90/90 books identical, rebuilt fingerprint `1cd43ea2`, all 12 arrays identical, and 41/41 files
made with the original table decode with the rebuilt one.

## How it works

T0 = an empty table plus the adaptive codec. The seed is cut into chunks at 1, 2, 4, 8, … MB. Chunk k is coded with a table rebuilt
(by the unchanged v1 method) from chunks 0…k-1, and the decoder rebuilds the same tables as it goes. Finally T1 is rebuilt from everything.

## Run

```sh
uv run python lab/exp_chain/chain.py encode progressive tables/exp_chain_seed_progressive.bin    # ~13 min (Python)
uv run python lab/exp_chain/chain.py decode tables/exp_chain_seed_progressive.bin tables/exp_chain_decoded   # ~27 min
uv run python lab/exp_chain/chain.py crosscheck tables/exp_chain_decoded
```

Files: `chain.py`, [METHOD.md](METHOD.md) (spec draft), [RESULTS.md](RESULTS.md), `results/`.
