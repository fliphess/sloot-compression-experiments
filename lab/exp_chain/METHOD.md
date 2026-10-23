# The steno chain: method description (draft 1, 2026-10-22)

This document describes the whole chain: tokenizer, table build, models, arithmetic coder, containers and
the bootstrap of a published table from its seed. It should be complete enough to reimplement everything in
another language **without our code**. It's written to be printable: it is the human-readable "T0" of the chain.
The Python in `lab/` is the reference implementation; where this text and the code disagree, the code
defines the current behaviour, and the disagreement is a bug in this text.

Conventions: all arithmetic is on unbounded non-negative integers unless stated otherwise; `//` is floor
division; `x << n` is `x · 2^n`; `|` is bitwise OR; arrays are 0-indexed; "ascending" means by numeric value;
byte strings compare lexicographically by unsigned byte value.

## 1. Overview

```
T0 (this text + an empty table)
  └─ decodes the SEED (compressed corpus)  ──build method (§3)──▶  T1 (published table, fingerprint)
                                                                     └─ decodes ARCHIVE FILES (§5, §6)
```

- A **table** is a deterministic function of (a list of texts, the build method). It is identified by a
  4-byte **fingerprint**; every compressed file records the fingerprint of the table it needs.
- A **seed** is the list of texts a table is built from, compressed with this method (§7). Shipping the seed
  instead of the table is smaller, and anyone can rebuild the table bit-identically.

## 2. Tokenizer

Input: a Unicode text (a sequence of code points). Bytes that aren't valid UTF-8 aren't text (see §6, mode 0).

The text is split, left to right, into **tokens**. At each position take the first rule that matches:

1. a word: optionally **one** ASCII space (U+0020) followed by **one or more** ASCII letters `A-Z a-z`, taking
   as many letters as possible;
2. otherwise: exactly **one** code point (any code point that isn't an ASCII letter, including space, newline,
   digits, punctuation and all non-ASCII letters).

So `" house"` is one token, `"Hello"` is one token, `" "` before a non-letter is a token on its own, and
`"é"` is a token. Concatenating the tokens gives back the text exactly. (Regex: ` ?[A-Za-z]+|[^A-Za-z]`,
with code-point semantics, not bytes or UTF-16 units.)

A token's **bytes** are its UTF-8 encoding.

## 3. Building a table from texts

Input: an ordered list of texts. Concatenate the token streams of all texts in order into one stream
(n-grams **do** run across text boundaries).

1. **Vocabulary.** Count every distinct token. Keep the tokens that occur **at least twice**. Sort them by
   count descending, then by token bytes ascending. Token id 0 is reserved for **UNK** (a token not in the
   vocabulary); the sorted tokens get ids 1, 2, 3, … The vocabulary size `V` includes UNK.
   For the fingerprint the vocabulary is stored as the concatenation of the UTF-8 bytes of
   `"<UNK>"` followed by every kept token in id order (`vocab_blob`), plus the byte length of each
   (`vocab_lens`, `V` entries).
2. **Id stream.** Map each token of the stream to its id (UNK = 0 for tokens with count 1). Let `b` be the
   number of bits of `V` (the bit length: the smallest `b` with `V < 2^b`).
3. **Order 0.** `uni[i]` = number of times id `i` occurs in the id stream (`V` entries). `uni[0]` is therefore
   the number of singleton occurrences. `total0 = Σ uni` must be `< 2^30`.
4. **Order 1 and 2 contexts.** For every position `p ≥ k` in the id stream (k = 1, 2):
   - order 1: context `c = id[p-1]`;
   - order 2: context `c = (id[p-2] << b) | id[p-1]`.

   Count how often each symbol `s = id[p]` follows each context `c`.
   Stored per order k: the distinct contexts ascending (`ck_ctx`); for each context, its followers with
   symbol ids ascending (`ck_sym`) and their counts (`ck_cnt`), all contexts' followers concatenated in
   context order; and `ck_start`, the start offset of each context's followers plus a final entry equal to the
   total number of followers.
5. **Spelling counts.** For every token whose count is **exactly 1**, take the sequence
   `256, its bytes…, 256` (256 = "boundary"), and count each adjacent pair `(prev, next)` in a 257 × 257
   matrix `spell[prev][next]`.
6. **Fingerprint.** SHA-256 over, for each array in this order: `vocab_blob, vocab_lens, uni, spell, c1_ctx,
   c1_start, c1_sym, c1_cnt, c2_ctx, c2_start, c2_sym, c2_cnt`: the array's name in ASCII, followed by every
   value (row-major for `spell`) as a little-endian signed 64-bit integer. The fingerprint is the first 4 bytes
   of the digest.

The **empty table T0** has `vocab_blob = "<UNK>"`, `vocab_lens = [5]`, `uni = [1]`, `spell` all zeros, no
contexts (`ck_start = [0]`, other context arrays empty). Its fingerprint follows from the same rule.

## 4. Arithmetic coder (Witten-Neal-Cleary, 32 bits)

Constants: `TOP = 2^32 − 1`, `HALF = 2^31`, `QUARTER = 2^30`. A symbol is coded as an interval
`[low_count, high_count)` of a `total` with `0 ≤ low_count < high_count ≤ total ≤ 2^30`.

**Encoder.** State `low = 0`, `high = TOP`, `pending = 0`; an output bit list. To emit bit `x`: append `x`,
then `pending` copies of `1 − x`, then set `pending = 0`.

```
encode(lc, hc, total):
    span = high − low + 1
    high = low + span·hc // total − 1
    low  = low + span·lc // total
    repeat:
        if high < HALF:                              emit 0
        elif low ≥ HALF:                             emit 1; low −= HALF; high −= HALF
        elif low ≥ QUARTER and high < HALF+QUARTER:  pending += 1; low −= QUARTER; high −= QUARTER
        else: stop
        low = 2·low; high = 2·high + 1
finish:
    pending += 1; emit (0 if low < QUARTER else 1)
    pad with 0 bits to a whole number of bytes; pack 8 bits per byte, first bit = most significant bit
    remove trailing zero bytes (the decoder reads missing bytes as zero)
```

**Decoder.** Read bits most-significant first; past the end every bit is 0. State `low = 0`, `high = TOP`,
`value` = the first 32 bits.

```
target(total) = ((value − low + 1)·total − 1) // (high − low + 1)
```

The decoded symbol is the one whose interval contains `target`. Then `consume(lc, hc, total)` updates `low`
and `high` exactly as the encoder does, with the same loop, except that in the loop it subtracts the same
amount from `value` (`HALF` or `QUARTER`; nothing in the `high < HALF` case) and does
`value = 2·value + next_bit` together with doubling `low`/`high`.

## 5. Models

Each model codes a sequence of tokens. The **history** is the list of ids already coded (the newest last).

### 5.1 Spelling (for UNK)

After coding UNK, the token's bytes are coded one by one, then the boundary symbol 256. Each symbol `x` is
coded with `prev` = the previous byte (256 before the first byte), over symbols 0…256 in ascending order
with width `spell[prev][x] + 1` (+ `ws · doc_spell[prev][x]` in the adaptive model, §5.3).

### 5.2 Static model (mode 1)

For each token: `w` = its id (0 if not in the vocabulary). Excluded set `X = {}`.

1. **Levels k = 2, then 1.** Skip a level if the history has fewer than k ids, or if its context (built from
   the last k history ids exactly as in §3.4: `id[p-1]` lowest) doesn't exist in the table. Candidates = the
   context's followers that aren't in `X`, in ascending id, with their counts. Skip the level if no candidate
   remains. Escape count `E` = number of candidates; `total = Σ counts + E`.
   - If `w` is a candidate: code `[c, c + count(w))` where `c` = sum of the counts of the candidates before it. Done.
   - Else code the **escape** `[total − E, total)`, add all candidates to `X`, and go to the next level.
2. **Order 0.** Symbols = all ids `0 … V−1` not in `X`, ascending, with width `uni[id]`. Code `w`. (UNK is an
   ordinary symbol here; there is no escape.)
3. If `w = 0`: spell the token (§5.1, `ws = 0`).
4. Append `w` to the history and keep only the last 2 ids.

### 5.3 Adaptive model (mode 2; E8b + E9)

A per-document **document layer** on top of the frozen table, learned while coding; the decoder rebuilds it
identically. Parameters: weight `W` (positive integer), `ws = 1`. Configuration "d3-sum-wW": levels
`D3`, `M2`, `M1`, then order 0, with new-word ids and UNK counting enabled.

State:
- **New words.** A token that is neither in the vocabulary nor seen before is coded as UNK + spelling; it then
  gets the next id `V, V+1, …` (in order of first appearance), and from then on is coded as that id.
- `doc_uni[id]`, `doc_total`: order-0 document counts.
- `doc_ctx[k][(last k history ids)][id]`: document counts for k = 1, 2, 3.
- `doc_spell[prev][next]`: letter pairs of spelled tokens.
- The history holds the *actual* ids (including new ids), last 3 kept. For **table** lookups the history ids
  `≥ V` are replaced by 0 (UNK).

Per token, with `w` = table id, else new id, else 0 (UNK); excluded set `X = {}`; levels in order:

- **D3** (document-only, order 3; weight 1): exists if the history has ≥ 3 ids and the document has seen
  that 3-id context. Candidates: its symbols ascending, count = document count.
- **M2, M1** (table + weighted document, order k): exists if the history has ≥ k ids and the table context
  (from the UNK-mapped history) or the document context (from the actual history) exists.
  Candidates: the table followers in ascending id, each with count `table count + W · document count`,
  followed by the symbols that only the document has seen, in ascending id, with count `W · document count`.

In every level, drop candidates in `X`; skip the level if none remain; escape count `E` = number of remaining
candidates; code `w` or the escape exactly as in §5.2 (the candidate **order** is as listed above). After an
escape, add the level's candidates to `X`.

**Order 0.** Symbols `0 … (V + number of new words) − 1`, ascending, excluding `X`; width of `id` =
`uni[id]` (0 for new ids) `+ W · doc_uni[id]`. No escape.

If `w = 0`: spell the token (§5.1) with `ws = 1`.

**Learning after each token** (encoder and decoder do exactly this):
1. If the token was coded as UNK: `doc_uni[0] += 1; doc_total += 1` (new words make UNK more likely, PPM
   method C at order 0); then give the token its new id `w`; add its spelling pairs (`256, bytes…, 256`) to
   `doc_spell`.
2. `doc_uni[w] += 1; doc_total += 1`.
3. For k = 1, 2, 3 with at least k history ids: `doc_ctx[k][last k history ids][w] += 1`.
4. Append `w` to the history (keep the last 3).

## 6. File container

| bytes | meaning |
|---|---|
| 1 | mode: 0 = stored (the rest of the file is the input, unchanged), 1 = static (§5.2), 2 = adaptive (§5.3) |
| 4 | fingerprint of the table used (modes 1 and 2) |
| varint | number of tokens (unsigned LEB128: 7 bits per byte, least significant group first, high bit = more bytes follow) |
| rest | arithmetic code (§4) |

The encoder uses mode 0 when the input isn't valid UTF-8, or when the coded file would not be smaller than
`1 + input length`. A decoder must refuse a file whose fingerprint differs from its table's.

## 7. The bootstrap chain (E9)

**Seed corpus of T1** (`en-gutenberg-v1`, fingerprint `1cd43ea2`): the 90 training books, in ascending
Project Gutenberg ID (see `lab/corpus_manifest.json`; how the books were selected and stripped is in
`lab/fetch_gutenberg.py`). T1 = the table built (§3) from these texts in this order.

**Compressing the seed, progressive mode.** Books in seed order are grouped into chunks by the running total
`T` of UTF-8 bytes of the books *before* the current one (1 MB = 10^6 bytes): start with chunk 0 and a limit
of 1 MB; before taking each book, if `T ≥ limit`, move to the next chunk and double the limit. So chunks start
at cumulative 1, 2, 4, 8, 16, … MB, and a chunk always holds at least one book. Each book is coded as one
adaptive document (§5.3) with:
- chunk 0: table T0 (empty), `W = 1`;
- chunk k ≥ 1: the table built (§3) from all books of chunks 0 … k−1 in seed order, `W = 8`.

Books in the same chunk are independent and can be coded in parallel.

**Seed file:** `"STNSEED1"` (8 ASCII bytes), varint number of books, then per book: varint chunk index,
varint name length, name (UTF-8; the Gutenberg ID), varint file length, file (§6).

**Decoding the seed:** for each chunk in order, build its table (T0, or from the books decoded so far), then
decode its books. Finally build T1 from all decoded books in seed order and check the fingerprint.

## 8. Known gaps and ambiguities (draft 1)

- **Count limits.** All totals must be ≤ 2^30. The table build doesn't rescale (fine up to ~1G tokens); the
  adaptive model doesn't rescale document counts, so documents of more than roughly 2^30 / W tokens aren't
  defined yet. The reference code also allows at most 2^19 new words per document.
- **Zero widths.** A symbol with width 0 at order 0 can't be coded. Not reachable with our tables (every kept
  token has count ≥ 2; UNK has the singleton count, ≥ 1 in practice), but the rule should say what happens.
- **Reading corpus files.** The reference build reads texts with Python's universal-newline translation
  (`\r\n` and lone `\r` become `\n`). Our stored books contain no `\r`, but a spec must fix this.
- **Corpus selection and stripping** (which books, how Gutenberg licence text is removed) is described only by
  code (`lab/fetch_gutenberg.py`) and a manifest. Sources drift (decision 2026-10-22: the method is the
  deliverable), so a rebuilt table can get a different fingerprint; the archive must keep the table instance
  (or its exact seed) that its files name.
- **Model constants** (`W`, chunk sizes, levels) were tuned coarsely on our test set; they're part of the
  format and must be fixed per format version.
- **No error correction or paper layout** yet (postponed by decision).
- Line-break (wrap) and case models from E8a aren't included in this draft.
