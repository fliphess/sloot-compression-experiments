"""Steno codec v1: a real encoder/decoder on top of a frozen reference table.

Everything is integer arithmetic, so any language reproduces it bit for bit.

Table (built once from the training books, then frozen and saved to disk):
  vocab     every token seen >= 2 times, sorted by count desc then bytes asc;
            id 0 = UNK, a token not in the table
  order 0   count of every token id
  order 1/2 for every context of 1 or 2 previous token ids: the token ids
            that followed it, with counts
  spell     order-1 byte counts over the rare training tokens, used to spell
            UNK tokens byte by byte (byte 256 = end of token)

Model per token, PPM method C with full exclusion:
  start at the order-2 context, then order 1, then order 0. In a context the
  candidates are the tokens seen there that weren't already excluded by a
  higher order, each with its count, plus an escape with count = number of
  candidates. If the token is a candidate it's coded; otherwise the escape is
  coded and all candidates are excluded below. Order 0 has every vocab token
  (minus exclusions) and no escape. UNK is then followed by its spelling.

Coder: Witten-Neal-Cleary arithmetic coding, 32-bit, frequency totals < 2^30.

File:
  byte 0      mode: 0 = stored (raw bytes follow), 1 = steno v1
  bytes 1-4   table fingerprint (first 4 bytes of its SHA-256)
  varint      number of tokens
  rest        arithmetic code; missing trailing bytes read as zero

Usage:
  python3 lab/codec.py build                 build and save the table
  python3 lab/codec.py compress IN OUT
  python3 lab/codec.py decompress IN OUT
  python3 lab/codec.py test                  round trip + size report
"""
import hashlib
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from steno import TOKEN, split_books

ROOT = Path(__file__).resolve().parent.parent
TABLE_PATH = ROOT / "tables" / "en-gutenberg-v1.npz"
STORED, STENO_V1 = 0, 1
UNK = 0
SPELL_END = 256

BITS = 32
TOP = (1 << BITS) - 1
HALF = 1 << (BITS - 1)
QUARTER = 1 << (BITS - 2)
MAX_TOTAL = QUARTER


# --- arithmetic coder ---------------------------------------------------------

class Encoder:
    def __init__(self):
        self.low, self.high, self.pending = 0, TOP, 0
        self.bits = []
        self.ideal = 0.0  # sum of -log2(p): what a perfect coder would spend

    def _emit(self, bit):
        self.bits.append(bit)
        self.bits.extend([1 - bit] * self.pending)
        self.pending = 0

    def encode(self, cum_low, cum_high, total):
        assert 0 <= cum_low < cum_high <= total <= MAX_TOTAL
        self.ideal -= math.log2((cum_high - cum_low) / total)
        span = self.high - self.low + 1
        self.high = self.low + span * cum_high // total - 1
        self.low = self.low + span * cum_low // total
        while True:
            if self.high < HALF:
                self._emit(0)
            elif self.low >= HALF:
                self._emit(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high < HALF + QUARTER:
                self.pending += 1
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                break
            self.low = 2 * self.low
            self.high = 2 * self.high + 1

    def finish(self):
        self.pending += 1
        self._emit(0 if self.low < QUARTER else 1)
        bits = self.bits + [0] * (-len(self.bits) % 8)
        out = bytearray(np.packbits(np.array(bits, dtype=np.uint8)).tobytes()) if bits else bytearray()
        while out and out[-1] == 0:  # the decoder reads missing bytes as zero
            out.pop()
        return bytes(out)


class Decoder:
    def __init__(self, data):
        self.bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8)).tolist()
        self.pos = 0
        self.low, self.high = 0, TOP
        self.value = 0
        for _ in range(BITS):
            self.value = 2 * self.value + self._bit()

    def _bit(self):
        bit = self.bits[self.pos] if self.pos < len(self.bits) else 0
        self.pos += 1
        return bit

    def target(self, total):
        span = self.high - self.low + 1
        return ((self.value - self.low + 1) * total - 1) // span

    def consume(self, cum_low, cum_high, total):
        span = self.high - self.low + 1
        self.high = self.low + span * cum_high // total - 1
        self.low = self.low + span * cum_low // total
        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low -= HALF
                self.high -= HALF
                self.value -= HALF
            elif self.low >= QUARTER and self.high < HALF + QUARTER:
                self.low -= QUARTER
                self.high -= QUARTER
                self.value -= QUARTER
            else:
                break
            self.low = 2 * self.low
            self.high = 2 * self.high + 1
            self.value = 2 * self.value + self._bit()


# --- frozen table -------------------------------------------------------------

class FrozenTable:
    ARRAYS = ["vocab_blob", "vocab_lens", "uni", "spell",
              "c1_ctx", "c1_start", "c1_sym", "c1_cnt",
              "c2_ctx", "c2_start", "c2_sym", "c2_cnt"]

    @classmethod
    def build(cls, texts):
        tokens = [t for text in texts for t in TOKEN.findall(text)]
        freq = Counter(tokens)
        kept = sorted((t for t, c in freq.items() if c >= 2), key=lambda t: (-freq[t], t.encode()))
        vocab = ["<UNK>"] + kept
        index = {t: i for i, t in enumerate(vocab)}
        ids = np.array([index.get(t, UNK) for t in tokens], dtype=np.int64)
        b = len(vocab).bit_length()
        arrays = {
            "vocab_blob": np.frombuffer(b"".join(t.encode() for t in vocab), dtype=np.uint8),
            "vocab_lens": np.array([len(t.encode()) for t in vocab], dtype=np.uint32),
            "uni": np.bincount(ids, minlength=len(vocab)).astype(np.uint32),
        }
        for k in (1, 2):
            key = ids[k:].copy()
            for j in range(1, k + 1):
                key |= ids[k - j:len(ids) - j] << (b * j)
            uniq, counts = np.unique(key, return_counts=True)
            ctx, start = np.unique(uniq >> b, return_index=True)
            arrays[f"c{k}_ctx"] = ctx
            arrays[f"c{k}_start"] = np.append(start, len(uniq)).astype(np.uint32)
            arrays[f"c{k}_sym"] = (uniq & ((1 << b) - 1)).astype(np.uint32)
            arrays[f"c{k}_cnt"] = counts.astype(np.uint32)
        spell = np.zeros((257, 257), dtype=np.uint32)
        for tok, c in freq.items():
            if c == 1:
                seq = [SPELL_END] + list(tok.encode()) + [SPELL_END]
                np.add.at(spell, (seq[:-1], seq[1:]), 1)
        arrays["spell"] = spell
        return cls(arrays)

    def __init__(self, arrays):
        self.arrays = arrays
        blob, lens = arrays["vocab_blob"].tobytes(), arrays["vocab_lens"]
        offsets = [0] + np.cumsum(lens, dtype=np.int64).tolist()
        self.vocab = [blob[offsets[i]:offsets[i + 1]].decode() for i in range(len(lens))]
        self.vocab[UNK] = None
        self.index = {t: i for i, t in enumerate(self.vocab) if t is not None}
        self.b = len(self.vocab).bit_length()
        self.uni = arrays["uni"].astype(np.int64)
        self.uni_cum = np.cumsum(self.uni)  # inclusive
        self.total0 = int(self.uni_cum[-1])
        self.spell_cum = np.cumsum(arrays["spell"].astype(np.int64) + 1, axis=1)  # Laplace +1
        self.ctx = {k: arrays[f"c{k}_ctx"] for k in (1, 2)}
        self.start = {k: arrays[f"c{k}_start"] for k in (1, 2)}
        self.sym = {k: arrays[f"c{k}_sym"].astype(np.int64) for k in (1, 2)}
        self.cnt = {k: arrays[f"c{k}_cnt"].astype(np.int64) for k in (1, 2)}
        assert self.total0 < MAX_TOTAL
        digest = hashlib.sha256()
        for name in self.ARRAYS:
            digest.update(name.encode())
            digest.update(np.ascontiguousarray(arrays[name]).astype("<i8").tobytes())
        self.fingerprint = digest.digest()[:4]

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.arrays)

    @classmethod
    def load(cls, path=TABLE_PATH):
        with np.load(path) as data:
            return cls({name: data[name] for name in cls.ARRAYS})

    def group(self, k, history):
        ctx = 0
        for j in range(1, k + 1):
            ctx |= history[-j] << (self.b * (j - 1))
        i = int(np.searchsorted(self.ctx[k], ctx))
        if i == len(self.ctx[k]) or self.ctx[k][i] != ctx:
            return None
        s, e = int(self.start[k][i]), int(self.start[k][i + 1])
        return self.sym[k][s:e], self.cnt[k][s:e]


# --- model: shared walk over the contexts, used by encoder and decoder --------

def candidates(table, history):
    """Yield (symbols, counts) per active context, highest order first, with exclusions applied."""
    excluded = np.empty(0, dtype=np.int64)
    for k in (2, 1):
        if len(history) < k:
            continue
        group = table.group(k, history)
        if group is None:
            continue
        syms, cnts = group
        if excluded.size:
            keep = ~np.isin(syms, excluded, assume_unique=True)
            syms, cnts = syms[keep], cnts[keep]
        if syms.size == 0:
            continue
        yield syms, cnts
        excluded = np.union1d(excluded, syms)
    yield excluded, None  # order 0 marker: everything except the excluded


def order0_cum(table, excluded):
    """Inclusive cumulative counts over the vocab with excluded tokens given zero width."""
    if excluded.size == 0:
        return table.uni_cum
    removed = np.zeros(len(table.uni), dtype=np.int64)
    removed[excluded] = table.uni[excluded]
    return table.uni_cum - np.cumsum(removed)


def encode_token(enc, table, w, history):
    for syms, cnts in candidates(table, history):
        if cnts is None:
            cum = order0_cum(table, syms)
            low = int(cum[w - 1]) if w else 0
            enc.encode(low, int(cum[w]), int(cum[-1]))
            return
        esc = int(syms.size)
        total = int(cnts.sum()) + esc
        i = int(np.searchsorted(syms, w))
        if i < syms.size and syms[i] == w:
            low = int(cnts[:i].sum())
            enc.encode(low, low + int(cnts[i]), total)
            return
        enc.encode(total - esc, total, total)


def decode_token(dec, table, history):
    for syms, cnts in candidates(table, history):
        if cnts is None:
            cum = order0_cum(table, syms)
            total = int(cum[-1])
            w = int(np.searchsorted(cum, dec.target(total), side="right"))
            dec.consume(int(cum[w - 1]) if w else 0, int(cum[w]), total)
            return w
        esc = int(syms.size)
        total = int(cnts.sum()) + esc
        target = dec.target(total)
        if target >= total - esc:
            dec.consume(total - esc, total, total)
            continue
        cum = np.cumsum(cnts)
        i = int(np.searchsorted(cum, target, side="right"))
        dec.consume(int(cum[i] - cnts[i]), int(cum[i]), total)
        return int(syms[i])
    raise AssertionError("unreachable")


def encode_spelling(enc, table, raw):
    prev = SPELL_END
    for b in list(raw) + [SPELL_END]:
        cum = table.spell_cum[prev]
        enc.encode(int(cum[b - 1]) if b else 0, int(cum[b]), int(cum[-1]))
        prev = b


def decode_spelling(dec, table):
    out, prev = bytearray(), SPELL_END
    while True:
        cum = table.spell_cum[prev]
        total = int(cum[-1])
        b = int(np.searchsorted(cum, dec.target(total), side="right"))
        dec.consume(int(cum[b - 1]) if b else 0, int(cum[b]), total)
        if b == SPELL_END:
            return bytes(out)
        out.append(b)
        prev = b


# --- container ----------------------------------------------------------------

def varint(n):
    out = bytearray()
    while True:
        out.append((n & 0x7F) | (0x80 if n > 0x7F else 0))
        n >>= 7
        if not n:
            return bytes(out)


def read_varint(data, pos):
    n = shift = 0
    while True:
        byte = data[pos]
        n |= (byte & 0x7F) << shift
        pos += 1
        shift += 7
        if not byte & 0x80:
            return n, pos


def compress(table, data, stats=None):
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return bytes([STORED]) + data
    tokens = TOKEN.findall(text)
    enc = Encoder()
    history = []
    for tok in tokens:
        w = table.index.get(tok, UNK)
        encode_token(enc, table, w, history)
        if w == UNK:
            encode_spelling(enc, table, tok.encode())
        history = (history + [w])[-2:]
    out = bytes([STENO_V1]) + table.fingerprint + varint(len(tokens)) + enc.finish()
    if stats is not None:
        stats["ideal_bits"] = enc.ideal
    if len(out) >= 1 + len(data):
        return bytes([STORED]) + data
    return out


def decompress(table, blob):
    if blob[0] == STORED:
        return blob[1:]
    if blob[0] != STENO_V1:
        raise ValueError(f"unknown mode {blob[0]}")
    if blob[1:5] != table.fingerprint:
        raise ValueError(f"made with table {blob[1:5].hex()}, this table is {table.fingerprint.hex()}")
    count, pos = read_varint(blob, 5)
    dec = Decoder(blob[pos:])
    parts, history = [], []
    for _ in range(count):
        w = decode_token(dec, table, history)
        parts.append(decode_spelling(dec, table).decode() if w == UNK else table.vocab[w])
        history = (history + [w])[-2:]
    return "".join(parts).encode()


# --- commands -----------------------------------------------------------------

def cmd_build():
    train, _ = split_books()
    t0 = time.time()
    table = FrozenTable.build([p.read_text(encoding="utf-8") for p in train])
    table.save(TABLE_PATH)
    print(f"built {TABLE_PATH.relative_to(ROOT)} in {time.time() - t0:.0f}s: "
          f"{len(table.vocab):,} words, {len(table.sym[1]):,} + {len(table.sym[2]):,} context entries, "
          f"{TABLE_PATH.stat().st_size / 1e6:.1f} MB on disk, fingerprint {table.fingerprint.hex()}")


def cmd_test():
    from short_docs import SIZES, snippets
    from steno import Table, encode_cost

    table = FrozenTable.load()
    _, test = split_books()
    texts = [p.read_text(encoding="utf-8") for p in test]
    estimate_table = Table([p.read_text(encoding="utf-8") for p in split_books()[0]])

    edge = {"empty": b"", "one char": b"a", "unicode": "Zoë’s café — naïve «quote» 日本\r\n".encode(),
            "binary": bytes(range(256)) * 4, "random": np.random.default_rng(1).bytes(3000),
            "unknown words": b"Xqzzt flurbograph vorpalisms zzyzx."}
    for name, data in edge.items():
        blob = compress(table, data)
        assert decompress(table, blob) == data, name
        print(f"edge {name:<14} {len(data):>6} -> {len(blob):>6} bytes (mode {blob[0] if blob else '-'})  OK")

    print(f"\n{'size':>7}{'docs':>6}{'estimate v0':>13}{'PPM ideal':>11}{'real file':>11}{'real bpc':>10}{'enc+dec s':>11}")
    for size in SIZES:
        n = est = ideal = real = raw = 0
        t0 = time.time()
        for text in texts:
            for snip in snippets(text, size):
                data = snip.encode()
                stats = {}
                blob = compress(table, data, stats)
                assert decompress(table, blob) == data
                est += math.ceil(encode_cost(estimate_table, snip, adaptive=False)[0][-1] / 8) + 4
                ideal += math.ceil(stats["ideal_bits"] / 8)
                real += len(blob)
                raw += len(data)
                n += 1
        print(f"{size:>7,}{n:>6}{est / n:>11.0f} B{ideal / n:>9.0f} B{real / n:>9.0f} B"
              f"{8 * real / raw:>10.3f}{(time.time() - t0) / n:>11.2f}")
    print("\nestimate v0 = lab/steno.py static calculation (interpolated model, +4 B)")
    print("PPM ideal   = -log2(p) of the codec's own model, no container")
    print("real file   = actual compressed file incl. mode, fingerprint and token count; all round trips OK")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"
    if cmd == "build":
        cmd_build()
    elif cmd == "test":
        cmd_test()
    elif cmd in ("compress", "decompress"):
        table = FrozenTable.load()
        data = Path(sys.argv[2]).read_bytes()
        fn = compress if cmd == "compress" else decompress
        Path(sys.argv[3]).write_bytes(fn(table, data))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
