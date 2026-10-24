# Jan Sloot, revisited on a Friday evening

*A hobby experiment, done for fun and to learn. Not research, not a paper, not a new invention.
Everything here was measured on one Mac mini (Apple M4) over two days, built together with an AI coding assistant (Claude Code).
The detailed log with every number is in [RESEARCH.md](RESEARCH.md); this is the story.*

## The myth

In the 1990s the Dutch TV repairman Jan Sloot claimed he could store a complete feature film in a few kilobytes: a key on a
chip card, played back by a small decoder box. He died in 1999, just before a deal with investors, and the code was never found.
The [Wikipedia article](https://en.wikipedia.org/wiki/Sloot_Digital_Coding_System) tells the story. The Dutch article adds a nice
detail: Sloot hated the word "compression". He called it *coding*, and said that "with clever helper tables" you could go
beyond what Shannon's information theory allows, though not for every arbitrary film.

The question for a Friday evening: **with 2026 hardware, what *could* such a system do, and what can't it do?** The
original wish was very concrete: a QR-code-like format to store as much data as possible on paper, so it lasts forever.

## The one idea that explains everything

You can't shrink *every* file. There are simply more possible films than possible 8 KB keys (the pigeonhole principle). A
compressor can only make data smaller by *knowing* what to expect. So the question is never "how small can it get", but **where does
the knowledge live?**

- **If the data is already in the decoder, the key is just a pointer.** We measured it: a 1 MB piece of Wikipedia that is
  inside a 90 MB shared reference compresses to **125 bytes**. A piece that is *not* in it: about 240,000 bytes. A demo of a
  "film in a few KB" would look exactly like the first case.
- **Otherwise the knowledge has to be statistics.** "Clever helper tables" don't beat Shannon. They change the question from
  "how surprising is this text?" to "how surprising is this text *for someone who already knows the table*?" (conditional entropy).
  That's real, but bounded. And random data stays incompressible: we tried a "second round with a table for random data" and every method came out slightly *bigger*.

## What we tried, in order

| step | idea | result |
|---|---|---|
| 1 | **Recursive pattern table**: replace common pairs with one symbol, again and again, until it's all references | ❌ converges after ~90 rounds; worse than gzip (the table costs as much as it saves) |
| 2 | **A shared reference file** that both sides have (zstd `--patch-from`) | the pointer effect above; for new text only ~20% better |
| 3 | **"Steno in overdrive"**: a frozen table of word statistics (which word follows which), plus arithmetic coding | ✅ for short texts ~2× smaller than gzip |
| 4 | **Learning from the document itself** while coding (the decoder learns the same way, so nothing extra is stored) | ✅ books -21% |
| 5 | **Bigger tables** (85 MB → 3 GB of Project Gutenberg books) | helps, but each tripling gains less (-8%, -5%, -3%) |
| 6 | **A small neural network we trained ourselves** (20M parameters, 19 MB) | ✅ as good as the 545 MB table, at 1/28 of the size |
| 7 | **Integer-only inference** for that network, so it decodes identically on every machine | ✅ costs only +0.2% (see below, the coolest part) |
| 8 | **C port** of the table codec, byte-identical to Python | ✅ 20-250× faster |
| 9 | **Other kinds of text** (Wikipedia, source code) | tables and models only work on text they've seen; a *mixed* table fixes most of it |
| 10 | **v2**: try every method per file and keep the smallest, plus a "match model" for repeated passages | ✅ never worse than the best single method |

Final numbers, as real files (decoded and checked byte for byte), per document:

| | 1 KB | 3 KB | 10 KB | vs gzip |
|---|---|---|---|---|
| books | 197 B | 555 B | 1,813 B | 2.4-2.9× smaller |
| Wikipedia | 226 B | 673 B | 2,179 B | 1.9-2.4× smaller |
| Python code | 219 B | 638 B | 1,980 B | 1.5-1.9× smaller |

### Postscript: a model trained on books, Wikipedia and code

The last run trained the same 20M model on a mix (60% books, 25% Wikipedia, 15% Python code), to see whether one small model can cover
several kinds of text, as the mixed table did. Its results:

<!-- mixed-results:start -->
| ideal bytes per doc, 1 / 3 / 10 KB | gzip (real) | mixed table + adaptive (real) | books-only 20M model | **mixed 20M model** |
|---|---|---|---|---|
| books | 565 / 1,480 / 4,410 | 256 / 732 / 2,382 | 197 / 554 / 1,819 | **194 / 546 / 1,795** |
| wiki | 538 / 1,411 / 4,201 | 229 / 679 / 2,195 | 278 / 811 / 2,749 | **172 / 499 / 1,598** |
| code | 421 / 1,061 / 2,936 | 220 / 638 / 1,980 | 301 / 863 / 2,796 | **137 / 390 / 1,237** |

Measured on the domains.py test set (30 docs per size, never in training). Model columns are ideal bits + 7 B container.
One small model now covers all three kinds of text, and it's even slightly better on books than the books-only model
(whole test books: 1.531 vs 1.562 bpc; int8 costs +0.3%). Caveat: the Wikipedia and code tests come from the same sources as the
training data (other articles from the same 2006 dump; other files from the same project, CPython), so those gains are optimistic,
code most of all. A fair test would use code from other projects.
<!-- mixed-results:end -->

## Why it doesn't work on paper

All the gain comes from the shared table or model. For a digital archive that's fine: publish the table once, and files only
carry its 4-byte fingerprint. But **on paper the reference has to be printed too**, or the archive can't be read in 200 years:

| reference | size | printed pages (est.) |
|---|---|---|
| our 20M neural model | 20 MB | ~70 |
| a table built from 85 MB of books | 31 MB | ~110 |
| the 3 GB table | 545 MB | ~1,900 |
| a local Qwen/GLM-class model | 0.25-4 GB | ~900-14,000 |

A table can at least be *rebuilt* from a compressed seed corpus plus the method (we did that bit for bit, and the compressed seed is
even smaller than the table). A neural network can't: training isn't reproducible. The only design that works on paper without
an external reference is an archive that **learns from itself**: every part is coded with the knowledge of the parts before it. That
needs only the method on paper, but it gains maybe 20-40% over xz or gzip. For the complexity, that isn't worth a new standard.
**So: a fun journey, but an unrealistic approach for paper.**

## The cool bit: neural inference that is bit-identical everywhere

A compressor and its decompressor must compute *exactly* the same probabilities. One rounding difference and the rest of the file
decodes to garbage. Neural networks normally use floating point, and floating-point results can differ between machines, libraries,
thread counts and versions. That's fine for a chatbot, and fatal for an archive.

What we built (`lab/round2/intmodel/`):

- **Everything in integers:** 8-bit weights with an integer multiplier and shift per channel, 64-bit accumulators, and integer versions
  of LayerNorm, softmax and GELU. An integer rule then turns the network's output into the frequencies the arithmetic coder needs.
- **Floating point only as an exact accelerator.** Fast float matrix libraries may be used, but only where every intermediate result is
  guaranteed to be exactly representable (below 2^24 for float32, below 2^53 for float64), so the answer is the same as the
  integer one.
- **A self-test in the model file:** a hash of the integer outputs for a fixed input. A machine that doesn't reproduce it exactly
  refuses to run, instead of silently producing a corrupt file.
- **Cost:** +0.1-0.2% bigger files than the floating-point ideal. Identical output across backends and thread settings.

None of the pieces is new (integer quantisation is well known), but the combination is practical, and it's useful well beyond
compression: anywhere a neural network must give the same answer on every machine, such as reproducible results, systems where many
parties must agree on one computation, or tests that must never flake.

## Others went much further

If this topic interests you, look at the people who do it properly:
- **"Language Modeling Is Compression"** (DeepMind, 2023): large language models as compressors.
- **Fabrice Bellard's `nncp` and `ts_zip`**: neural compression done seriously, including determinism.
- **The Hutter Prize and `cmix` / `paq8`**: decades of context-mixing compression, far better than our numbers.
- **zstd and Brotli dictionaries**: the production version of "a shared table for small messages".

Our numbers are modest next to theirs. The point here was to understand *why* things work, by building each step and measuring it.

## Lessons that aren't about compression

- **Measure before believing.** Several "obviously good" ideas (recursion, a second round, a smarter tokenizer for code) measured
  as dead ends, and one boring idea (learning from the document) was the biggest win.
- **Watch your test set.** Project Gutenberg has several editions of the same book; without removing them we'd have measured the
  Sloot pointer effect and called it compression.
- **Variety beats volume.** A 67 MB table built from books, Wikipedia and code beat a 545 MB books-only table everywhere outside books.
- **Watch the goal.** Bigger models kept improving the numbers while drifting away from the actual question (paper). Stopping was the right call.

## If you want to build on it

Everything is reproducible: see [README.md](README.md) for the commands and [lab/README.md](lab/README.md) for a map of every
experiment with its verdict. The most reusable parts are the integer-only neural codec (`lab/round2/intmodel/`), the byte-identical
C table codec (`lab/c/`), the v2 container that picks the best method per file (`lab/v2/`), and the test-set leakage check
(`lab/exp_table/leakage.py`).

Open directions that were never tried: an archive that learns from itself plus a real paper layer (symbology, error correction,
a self-describing header page), and lossless recompression of files that are already compressed (JPEG, MP3, zip).
