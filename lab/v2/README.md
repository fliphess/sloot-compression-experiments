# v2: everything we learned in one codec (router + match model)

**Question:** combine the methods that won: the tables with document learning (E8b/E8c/E13), the integer neural model (E12),
and the match model (tricks round, trick 2). Per file the encoder picks the smallest (trick 1). Does that beat every single method,
on books, Wikipedia and code?

**Verdict: ✅ works.** Never worse than the best single method. Real files vs gzip: books 2.9× (1 KB) to 2.4× (10 KB) smaller, Wikipedia 2.4× to 1.9×, code 1.9× to 1.5×. The router picks the neural model for books and the mixed table for Wikipedia and code. Full table in `RESEARCH.md` (v2 section).

## How it works

- **`codec2.py`**: the v2 container. Byte 0 is a unique method id (0 stored, 1 table static, 2 table adaptive, 3 reserved
  for the E9 chain variant, 4 neural, 5 neural + match), followed by that method's existing format with its 4-byte table/model fingerprint.
  This fixes the E11 finding that mode byte 2 meant three different things. Routing between tables and models costs no extra bytes, because the fingerprint
  already names the resource. Never worse than the best single candidate, and never more than input + 1 byte.
- **`neuralmatch.py`**: method 5, all integer. The E12 neural model, plus a match model that follows the earlier occurrence of the last 6 bytes.
  When the match model predicts a byte, a binary "is it that byte?" is coded with an adaptive integer probability (indexed by match length ×
  model confidence); on a miss, the byte is coded with that byte excluded.
- **`evaluate.py`**: real bytes for every candidate on the domains test set (C codec for the tables; integer model for neural),
  with every chosen file decode-verified.

## Run (needs `uv sync --extra neural`)

```sh
for i in 0 1 2 3; do VECLIB_MAXIMUM_THREADS=1 nohup uv run python lab/v2/evaluate.py run $i 4 > lab/v2/logs/eval_$i.log 2>&1 & done
uv run python lab/v2/evaluate.py summary
uv run python lab/v2/codec2.py compress IN OUT      # Python reference: tries all tables + the model (slow)
uv run python lab/v2/codec2.py decompress IN OUT
```

Needs `tables/` with `en-gutenberg-v1.npz`, `exp_table_3gb.npz`, `round2_mixed.npz` (+ their `c_*.bin` conversions) and `intmodel_20m_big.bin`.
