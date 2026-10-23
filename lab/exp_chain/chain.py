"""E9: the bootstrap chain T0 -> T1 -> archive.

The published table T1 (tables/en-gutenberg-v1.npz, fingerprint 1cd43ea2) is a
deterministic function of its seed corpus (the 90 training books) and the v1
build method. Instead of shipping T1, we ship the seed compressed, and anyone
rebuilds T1 bit-identically.

T0 is the empty table: vocab = [UNK] only, no contexts, spelling = Laplace.
Coding from an empty table uses E8b's adaptive codec plus one change: every
spelled (new) word also increments the order-0 count of UNK, so the chance of
a new word follows the observed rate of new words (PPM method C at order 0)
instead of vanishing.

Seed modes:
  progressive  books in training order, grouped into chunks that start at a
               cumulative 1, 2, 4, 8, ... MB of text. Chunk 0 is coded with T0; chunk k with
               a table built (v1 method) from all books of chunks 0..k-1, which
               the decoder rebuilds the same way after decoding them. Each book
               is one adaptive document. Books within a chunk are independent
               (parallel).
  perbook      every book coded alone with T0 (fully parallel, no rebuilds).

Seed container:
  magic b"STNSEED1", varint number of books, then per book:
  varint chunk index, varint name length, name (UTF-8), varint blob length, blob
  (blob = one adaptive-codec file, see lab/exp_adaptive/adaptive.py)

Usage:
  python3 lab/exp_chain/chain.py tune BOOK_ID          pick T0 weights on one book
  python3 lab/exp_chain/chain.py encode MODE OUT       compress the seed
  python3 lab/exp_chain/chain.py decode SEED OUTDIR    decode, rebuild T1, verify
  python3 lab/exp_chain/chain.py crosscheck OUTDIR     rebuilt T1 vs original: arrays + test files
"""
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
sys.path[:0] = [str(LAB), str(LAB / "exp_adaptive")]

import adaptive  # noqa: E402
from adaptive import ADAPTIVE, Config, DocState, decode_spelling, decode_token, encode_spelling, encode_token  # noqa: E402
from codec import STORED, UNK, Decoder, Encoder, FrozenTable, read_varint, varint  # noqa: E402
from steno import TOKEN, split_books  # noqa: E402

ROOT = LAB.parent
MAGIC = b"STNSEED1"
V1_FINGERPRINT = "1cd43ea2"
MB = 1_000_000


def config(w):
    return Config(f"d3-sum-w{w}", (("D", 3, 1), ("M", 2, w), ("M", 1, w)), w0=w, new_words=True, ws=1)


# T0 has no counts for the "M" levels to be weighted against, so its weight
# only trades document counts against the escape count; set by `tune`.
T0_WEIGHT = 1
TABLE_WEIGHT = 8  # best on seed books in `tune`-style checks (w=8: 2.088/1.834 bpc vs w=32: 2.223/1.855)


def empty_table():
    vocab = "<UNK>".encode()
    empty_i8 = np.empty(0, dtype=np.int64)
    empty_u4 = np.empty(0, dtype=np.uint32)
    arrays = {"vocab_blob": np.frombuffer(vocab, dtype=np.uint8), "vocab_lens": np.array([len(vocab)], np.uint32),
              "uni": np.array([1], np.uint32), "spell": np.zeros((257, 257), np.uint32)}
    for k in (1, 2):
        arrays[f"c{k}_ctx"] = empty_i8
        arrays[f"c{k}_start"] = np.zeros(1, np.uint32)
        arrays[f"c{k}_sym"] = empty_u4
        arrays[f"c{k}_cnt"] = empty_u4
    return FrozenTable(arrays)


class ChainDoc(DocState):
    """E8b document state; with count_unk, spelled words also count as UNK at order 0."""

    def __init__(self, table, cfg, count_unk):
        super().__init__(table, cfg)
        self.count_unk = count_unk

    def learn(self, tok, w, hist):
        if self.count_unk and w == UNK:
            self.doc_uni[UNK] += 1
            self.fen.add(UNK, 1)
            self.doc_total += 1
        return super().learn(tok, w, hist)


def compress_doc(table, cfg, text, count_unk):
    tokens = TOKEN.findall(text)
    st = ChainDoc(table, cfg, count_unk)
    enc = Encoder()
    hist = []
    for tok in tokens:
        w = st.lookup(tok)
        encode_token(enc, st, w, hist)
        if w == UNK:
            encode_spelling(enc, st, tok.encode())
        hist = (hist + [st.learn(tok, w, hist)])[-3:]
    return bytes([ADAPTIVE]) + table.fingerprint + varint(len(tokens)) + enc.finish()


def decompress_doc(table, cfg, blob, count_unk):
    assert blob[0] == ADAPTIVE and blob[1:5] == table.fingerprint, "wrong mode or table"
    count, pos = read_varint(blob, 5)
    dec = Decoder(blob[pos:])
    st = ChainDoc(table, cfg, count_unk)
    parts, hist = [], []
    for _ in range(count):
        w = decode_token(dec, st, hist)
        tok = decode_spelling(dec, st).decode() if w == UNK else st.token(w)
        parts.append(tok)
        hist = (hist + [st.learn(tok, w, hist)])[-3:]
    return "".join(parts)


def schedule(sizes, mode):
    """Chunk index per book: progressive = new chunk at a cumulative 1, 2, 4, ... MB; perbook = all 0."""
    if mode == "perbook":
        return [0] * len(sizes)
    chunks, chunk, done, limit = [], 0, 0, MB
    for size in sizes:
        if done >= limit:
            chunk += 1
            limit *= 2
        chunks.append(chunk)
        done += size
    return chunks


def chunk_table(texts, chunk):
    """The table a chunk is coded with: T0 for chunk 0, else built from all earlier texts."""
    return empty_table() if chunk == 0 else FrozenTable.build(texts)


# --- workers (fork start method: the table is inherited, not pickled) --------

_table = None


def _encode(job):
    name, text, chunk = job
    t0 = time.time()
    cfg = config(T0_WEIGHT if chunk == 0 else TABLE_WEIGHT)
    blob = compress_doc(_table, cfg, text, count_unk=True)
    return name, blob, time.time() - t0


def _decode(job):
    name, blob, chunk = job
    t0 = time.time()
    cfg = config(T0_WEIGHT if chunk == 0 else TABLE_WEIGHT)
    return name, decompress_doc(_table, cfg, blob, count_unk=True), time.time() - t0


def run_chunks(jobs_by_chunk, worker, texts_before):
    """Run each chunk's jobs in parallel with its table; texts_before(k) gives the texts preceding chunk k."""
    global _table
    ctx = mp.get_context("fork")
    out = {}
    for chunk in sorted(jobs_by_chunk):
        t0 = time.time()
        _table = chunk_table(texts_before(chunk), chunk)
        built = time.time() - t0
        with ctx.Pool(3) as pool:
            for name, result, secs in pool.imap_unordered(worker, jobs_by_chunk[chunk]):
                out[name] = (result, secs)
        print(f"chunk {chunk}: {len(jobs_by_chunk[chunk])} books, table {len(_table.vocab):,} words "
              f"(built {built:.0f}s), done in {time.time() - t0:.0f}s", flush=True)
    return out


# --- commands ---------------------------------------------------------------

def cmd_tune(book_id):
    path = next(p for p in split_books()[0] if p.stem == book_id)
    text = path.read_text(encoding="utf-8")
    t = empty_table()
    for w in (1, 2, 4, 32):
        for count_unk in (True, False):
            blob = compress_doc(t, config(w), text, count_unk)
            ok = decompress_doc(t, config(w), blob, count_unk) == text
            print(f"book {book_id} ({len(text.encode()):,} B) w={w:<3} count_unk={count_unk!s:<5} "
                  f"-> {len(blob):,} B {8 * len(blob) / len(text.encode()):.3f} bpc round trip {ok}", flush=True)


def cmd_encode(mode, out_path):
    train, _ = split_books()
    texts = [p.read_text(encoding="utf-8") for p in train]
    names = [p.stem for p in train]
    chunks = schedule([len(t.encode()) for t in texts], mode)
    jobs = {}
    for name, text, chunk in zip(names, texts, chunks):
        jobs.setdefault(chunk, []).append((name, text, chunk))
    first = {c: chunks.index(c) for c in set(chunks)}
    t0 = time.time()
    results = run_chunks(jobs, _encode, lambda c: texts[:first[c]])
    seed = bytearray(MAGIC + varint(len(names)))
    for name, chunk in zip(names, chunks):
        blob = results[name][0]
        seed += varint(chunk) + varint(len(name.encode())) + name.encode() + varint(len(blob)) + blob
    Path(out_path).write_bytes(seed)
    raw = sum(len(t.encode()) for t in texts)
    stats = {"mode": mode, "books": len(names), "chunks": max(chunks) + 1, "raw": raw, "seed": len(seed),
             "bpc": 8 * len(seed) / raw, "seconds": time.time() - t0,
             "cpu_seconds": sum(r[1] for r in results.values()),
             "per_book": {n: {"raw": len(t.encode()), "real": len(results[n][0]), "chunk": c}
                          for n, t, c in zip(names, texts, chunks)}}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"encode_{mode}.json").write_text(json.dumps(stats, indent=1))
    print(f"seed ({mode}): {raw:,} -> {len(seed):,} bytes ({stats['bpc']:.3f} bpc) in {stats['seconds']:.0f}s "
          f"wall, {stats['cpu_seconds']:.0f}s CPU", flush=True)


def parse_seed(data):
    assert data[:8] == MAGIC, "not a seed file"
    n, pos = read_varint(data, 8)
    books = []
    for _ in range(n):
        chunk, pos = read_varint(data, pos)
        ln, pos = read_varint(data, pos)
        name = data[pos:pos + ln].decode()
        pos += ln
        lb, pos = read_varint(data, pos)
        books.append((name, chunk, data[pos:pos + lb]))
        pos += lb
    assert pos == len(data)
    return books


def cmd_decode(seed_path, out_dir):
    """Decode the seed (books + chunk order come from the seed only), rebuild T1, verify."""
    books = parse_seed(Path(seed_path).read_bytes())
    jobs = {}
    for name, chunk, blob in books:
        jobs.setdefault(chunk, []).append((name, blob, chunk))
    decoded = {}

    def texts_before(chunk):
        return [decoded[name] for name, c, _ in books if c < chunk]

    t0 = time.time()
    global _table
    ctx = mp.get_context("fork")
    for chunk in sorted(jobs):
        _table = chunk_table(texts_before(chunk), chunk)
        with ctx.Pool(3) as pool:
            for name, text, _ in pool.imap_unordered(_decode, jobs[chunk]):
                decoded[name] = text
        print(f"decoded chunk {chunk} ({len(jobs[chunk])} books), {time.time() - t0:.0f}s", flush=True)
    dec_s = time.time() - t0

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, _, _ in books:
        (out / f"{name}.txt").write_text(decoded[name], encoding="utf-8")
    originals = {p.stem: p.read_text(encoding="utf-8") for p in split_books()[0]}
    identical = sum(decoded[n] == originals.get(n) for n, _, _ in books)
    t1 = time.time()
    rebuilt = FrozenTable.build([decoded[name] for name, _, _ in books])
    build_s = time.time() - t1
    rebuilt.save(out / "rebuilt-T1.npz")
    print(f"books identical to corpus: {identical}/{len(books)}; decode {dec_s:.0f}s, T1 rebuild {build_s:.0f}s; "
          f"rebuilt fingerprint {rebuilt.fingerprint.hex()} (expected {V1_FINGERPRINT})", flush=True)
    return rebuilt


def cmd_crosscheck(out_dir):
    """Rebuilt T1 must equal the original array for array; files made with the original must decode with it."""
    import codec
    from short_docs import snippets
    original = FrozenTable.load()
    rebuilt = FrozenTable.load(Path(out_dir) / "rebuilt-T1.npz")
    same = all(np.array_equal(original.arrays[n], rebuilt.arrays[n]) for n in FrozenTable.ARRAYS)
    print(f"fingerprints: original {original.fingerprint.hex()}, rebuilt {rebuilt.fingerprint.hex()}; "
          f"all {len(FrozenTable.ARRAYS)} arrays identical: {same}")
    e8b = config(32)
    ok = n = 0
    for path in split_books()[1]:
        text = path.read_text(encoding="utf-8")
        for snip in list(snippets(text, 3000))[:2]:
            data = snip.encode()
            v1 = codec.compress(original, data)
            ad = adaptive.compress(original, e8b, data)
            ok += codec.decompress(rebuilt, v1) == data
            ok += adaptive.decompress(rebuilt, e8b, ad) == data
            n += 2
    small = next(p for p in split_books()[1] if p.stem == "1952").read_bytes()
    ok += codec.decompress(rebuilt, codec.compress(original, small)) == small
    n += 1
    print(f"test files compressed with the original table, decoded with the rebuilt one: {ok}/{n} identical "
          f"(20 snippets x static v1 + adaptive E8b, plus book 1952 static)")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "tune":
        cmd_tune(sys.argv[2])
    elif cmd == "encode":
        cmd_encode(sys.argv[2], sys.argv[3])
    elif cmd == "decode":
        cmd_decode(sys.argv[2], sys.argv[3])
    elif cmd == "crosscheck":
        cmd_crosscheck(sys.argv[2])
    else:
        sys.exit(__doc__)
