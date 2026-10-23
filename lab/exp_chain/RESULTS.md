# E9: bootstrap chain T0 → T1 → archive

Question: can the published table T1 (`tables/en-gutenberg-v1.npz`, 31.0 MB, fingerprint `1cd43ea2`) be
distributed and preserved as a **compressed seed corpus + method** instead of as a table file, and be rebuilt
bit-identically by anyone? And how big is that seed?

Status: **done and fully verified** (2026-10-22 23:35); see "Verification" at the end.

## Method (summary; full description in METHOD.md)

- **T0 = the empty table** (vocab only UNK, no contexts, Laplace spelling) + E8b's adaptive codec. One addition
  (`count_unk`): every spelled new word also increments UNK's order-0 count, so the probability of a new word
  follows the observed rate of new words (PPM method C at order 0). Without it, a new word's probability
  vanishes as `1/(W·N)`.
- **Seed** = the 90 training books in ascending ID, the exact input of `uv run python lab/codec.py build`.
- **progressive**: books grouped into chunks starting at a cumulative 1, 2, 4, 8, … MB. Chunk 0 is coded with T0;
  chunk k with a table built by the unchanged v1 method from all books of the earlier chunks, which the decoder
  rebuilds the same way after decoding them. **Knowledge built on knowledge**: 7 rebuilds of 0-5 s each. Books in a
  chunk are independent, so 3 run in parallel.
- **perbook**: every book coded alone with T0, fully parallel, no rebuilds.
- Seed container: `STNSEED1` + per book (chunk index, name, adaptive-codec file). Format in METHOD.md §7.

## Tuning (coarse, on training books only)

T0 on book 41 (69,928 B), empty table, round trips OK:

| W | count_unk | bpc |
|---|---|---|
| **1** | **yes** | **3.114** |
| 1 | no | 3.607 |
| 2 | yes | 3.182 |
| 4 | yes | 3.276 |
| 32 | yes | 3.644 |

With a table (book 829, 584 KB; tables from the first 1.4 MB and 20.5 MB of seed books):

| table from | W=8 | W=32 |
|---|---|---|
| 1.4 MB | **2.088** | 2.223 |
| 20.5 MB | **1.834** | 1.855 |

count_unk makes no measurable difference with a real table (identical to 3 decimals). Chosen: chunk 0 `W=1`,
chunks ≥ 1 `W=8` (E8b's W=32 was tuned for the full 84 MB table).

## Results: seed size

84,740,376 bytes of seed text (UTF-8):

| method | seed size | bpc | vs v1 table (31.0 MB) | time |
|---|---|---|---|---|
| xz -9e (one stream) | 21.1 MB | 1.99 | -32% | |
| zstd -19 --long (one stream) | 21.8 MB | 2.06 | -30% | |
| **E9 progressive** | **19,087,840 B (19.09 MB)** | **1.802** | **-38%** | 795 s wall / 1,287 s CPU (3 cores) |
| E9 perbook (T0 only) | 21,302,064 B (21.30 MB) | 2.011 | -31% | 582 s wall / 1,164 s CPU |
| best-of per book (estimate) | 18,968,076 B + 1 flag byte per book | 1.791 | -39% | |

Per chunk (progressive vs perbook, bpc):

| chunk | books | MB | cumulative MB | progressive | perbook |
|---|---|---|---|---|---|
| 0 | 6 | 1.3 | 1.3 | 2.455 | 2.455 |
| 1 | 3 | 0.8 | 2.1 | 2.099 | 2.375 |
| 2 | 5 | 2.4 | 4.5 | 2.054 | 2.279 |
| 3 | 1 | 5.4 | 9.9 | 2.166 | 2.088 |
| 4 | 7 | 6.6 | 16.6 | 1.960 | 2.143 |
| 5 | 18 | 16.5 | 33.0 | 1.797 | 2.081 |
| 6 | 35 | 37.3 | 70.4 | 1.668 | 1.881 |
| 7 | 15 | 14.4 | 84.7 | 1.825 | 2.074 |

## Findings

1. **The compressed seed (19.09 MB) is 38% smaller than the table it produces (31.0 MB)**, and 10% smaller than
   xz on the same text. Shipping "seed + method" instead of the table is cheaper, and the method is the deliverable anyway.
2. **The progressive chain is itself a working recursion:** each chunk is compressed with the knowledge of all
   chunks before it. That gains 10.4% over coding every book from scratch (perbook), while staying parallel within a chunk.
3. **Trade-off progressive vs perbook:**
   - progressive: smaller, but chunks are sequential, and decoding chunk k needs chunks 0…k-1;
   - perbook: any book decodes on its own, fully parallel, 11.6% bigger.

   For preserving one published table, progressive is the right choice: you always rebuild the whole table anyway.
4. **A small table can hurt a large self-contained book.** Book 100 (5.4 MB, chunk 3, a 21K-word table built from
   4.5 MB of prose) is better from T0 (2.088) than with the table (2.166). The same holds for 27509 and 61985. An
   encoder-side choice per book (one flag) would save ~0.6% (estimate: the choice doesn't affect later tables,
   which are built from the decoded text).
5. **Tiny printable T0** (top-K words): not measured. For one long progressive stream it can only save the
   first occurrences in chunk 0 (≈ K words × ~20 bits ≈ 10 KB for K = 4,000, < 0.1% of the seed), so I skipped it.
   It would matter for perbook or very small seeds.

## Verification

Started 2026-10-22 23:08 CEST; expected done ≈ 23:45-23:55 (decoding is ~2.5× slower than encoding in Python: the
order-0 decoder does a binary search with Python-level prefix sums; + ~10 s rebuild + ~1 min cross-check):

```sh
# from the project root
nohup sh -c 'date; time uv run python lab/exp_chain/chain.py decode tables/exp_chain_seed_progressive.bin tables/exp_chain_decoded \
  && uv run python lab/exp_chain/chain.py crosscheck tables/exp_chain_decoded; date' > lab/exp_chain/results/decode.log 2>&1 &
```

- Log: `lab/exp_chain/results/decode.log`. Decoded books go to `tables/exp_chain_decoded/*.txt`, the rebuilt table
  to `tables/exp_chain_decoded/rebuilt-T1.npz`.
- **Success criteria** (printed in the log):
  1. `books identical to corpus: 90/90`;
  2. `rebuilt fingerprint 1cd43ea2 (expected 1cd43ea2)`;
  3. `all 12 arrays identical: True`;
  4. `test files ... 41/41 identical` (20 snippets × static v1 + adaptive E8b with the original table, decoded
     with the rebuilt one, plus book 1952).
- The perbook seed isn't decoded (same codec as chunk 0, whose round trip is covered by the progressive decode;
  it was measured for size only).

**Result (2026-10-22 23:35, all four criteria met):** 90/90 books identical to the corpus (decode 1,604 s in Python, T1 rebuild 7 s);
rebuilt fingerprint `1cd43ea2` = expected; all 12 arrays identical; 41/41 test files made with the original table decode
identically with the rebuilt one.

## Caveats

- Weights (W=1 for T0, W=8 with tables) and the chunk schedule (doubling from 1 MB) were tuned coarsely on 2
  training books; no search was done.
- Python speed: ≈ 15 s CPU per MB to encode. Fine for a one-off seed; a C port would make it minutes.
- No count rescaling: the largest seed book (5.9 MB) is far below the 2^30 / W limit, but a spec needs a rule.
- The seed depends on the exact corpus files: a drifted Gutenberg book gives a different seed *and* a different T1
  (fingerprint); that's consistent with "the method is the deliverable".

## Reproduce

```sh
uv run python lab/exp_chain/chain.py tune 41                                                # T0 weight check (~10 s)
uv run python lab/exp_chain/chain.py encode progressive tables/exp_chain_seed_progressive.bin  # ~13 min, 3 cores
uv run python lab/exp_chain/chain.py encode perbook tables/exp_chain_seed_perbook.bin          # ~10 min, 3 cores
uv run python lab/exp_chain/chain.py decode tables/exp_chain_seed_progressive.bin tables/exp_chain_decoded  # ~15 min
uv run python lab/exp_chain/chain.py crosscheck tables/exp_chain_decoded                       # ~1 min
```

Per-book sizes: `lab/exp_chain/results/encode_{progressive,perbook}.json`; logs in `lab/exp_chain/results/`.
