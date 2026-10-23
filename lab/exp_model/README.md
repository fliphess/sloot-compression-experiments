# exp_model: model refinements for the static codec (E8a)

**Question:** can a better model of the text, with the same 90-book table, make codec v1 smaller?

**Verdict: ⚠️ partly adopted, not yet merged into the main codec.** Only the line-break flag gives a real gain, and that's
mostly specific to Gutenberg's hard-wrapped text. Longer contexts and the textbook escape methods make it worse.

## Key result (real bytes, all 200 snippets round-trip verified)

| variant | what it does | short docs vs v1 | verdict |
|---|---|---|---|
| `wrap` | codes "line break here?" as a flag instead of a `\n` token (Gutenberg wraps at ~70 columns) | **-3.9%** | ✅, but only -1.5% on normal unwrapped text |
| `case` / `case2` | lowercase the word + a separate case class | -0.6 / -0.7% | small win |
| `sp2` | better spelling model for unknown words | -0.7% | small win |
| `S` | escape rates measured on held-out training books | -0.5% | small win |
| `D` / `X` | PPMD / PPMX escapes | +0.7 / +0.5% | ❌ worse |
| `o3` / `o4` | 3- or 4-word contexts | +1.7 / +2.9% | ❌ worse with 85 MB of training text (too sparse) |
| **`wrap+case2+sp2+S`** | best combination | **-5.4%** (1 KB: 285 → 270 B; books 2.42 → 2.29 bpc) | ⚠️ |

Where the bits go in the best combination: words predicted from context 66%, single-word fallback 16%, escapes 8%,
spelling 6.5%, case 2%. Little room is left in this static model; the bigger levers turned out to be document learning
(`exp_adaptive`), bigger tables (`exp_table`) and neural models (`round2`).

## Run

```sh
uv run python lab/exp_model/run.py base                  # = codec v1, builds tables/exp_model_o2.npz
uv run python lab/exp_model/run.py wrap+case2+sp2+S      # any variant: parts joined with +
uv run python lab/exp_model/summarize.py                 # results table
uv run python lab/exp_model/breakdown.py wrap+case2+sp2+S 10000   # bits per category
```

Files: `model.py` (variants, encoder/decoder), `run.py` (evaluation), `summarize.py`, `breakdown.py`, `results/*.json`.
Full details and a port-ready algorithm description: [RESULTS.md](RESULTS.md).
