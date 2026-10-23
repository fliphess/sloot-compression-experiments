"""v2 method 5: the integer neural codec (E12) + an integer match model (trick 2 of the tricks round).

Everything in the coding path is integer. Per byte:
  1. The neural model gives cumulative integer frequencies over 257 symbols (nncodec.freqs, unchanged).
  2. If the match model predicts a byte b* (it follows the earlier occurrence of the last MINLEN bytes),
     a binary "next byte == b*?" is coded first with probability q / 2^12, where
       q = clamp(floor((2·hits + 1) · 2^12 / (2·total + 2)), 1, 2^12 - 1)
     from an adaptive table indexed by (match length // 4, floor(f(b*) · 16 / total_freq)).
     On a miss, the byte is coded with the model's frequencies with b*'s frequency removed.
  3. Without a prediction, the byte is coded with the model's frequencies as in E12.
Windows, context reset and the model are exactly E12's (nncodec.py), so on a document without any
match this produces the same bits as method 4.
File (inside the v2 container, after the method byte): model fingerprint (4) + varint(length) + code.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "round2" / "intmodel"), str(HERE.parent)]

from codec import Decoder, Encoder  # noqa: E402
from nncodec import BOS, CTX, freqs, window_starts  # noqa: E402

MINLEN, MAXLEN = 6, 64
LBUCKETS = MAXLEN // 4 + 1
PBINS = 16
QBITS = 12
QTOTAL = 1 << QBITS


class Match:
    def __init__(self):
        self.data = bytearray()
        self.table = {}
        self.ptr = 0
        self.length = 0
        self.hits = np.zeros((LBUCKETS, PBINS), dtype=np.int64)
        self.total = np.zeros((LBUCKETS, PBINS), dtype=np.int64)

    def predict(self):
        return self.data[self.ptr] if self.length > 0 else -1

    def context(self, cum, b):
        total = int(cum[-1])
        fb = int(cum[b]) - (int(cum[b - 1]) if b else 0)
        lb, pbin = self.length // 4, fb * PBINS // total
        q = ((2 * int(self.hits[lb, pbin]) + 1) << QBITS) // (2 * int(self.total[lb, pbin]) + 2)
        return lb, pbin, min(max(q, 1), QTOTAL - 1), fb, total

    def update(self, sym):
        t = len(self.data)
        self.data.append(sym)
        if self.length > 0:
            if self.data[self.ptr] == sym:
                self.length, self.ptr = min(self.length + 1, MAXLEN), self.ptr + 1
            else:
                self.length = 0
        if t + 1 >= MINLEN:
            key = bytes(self.data[t + 1 - MINLEN:t + 1])
            if self.length == 0 and key in self.table:
                ptr, back = self.table[key], 0
                while back < MAXLEN and ptr - 1 - back >= 0 and self.data[ptr - 1 - back] == self.data[t - back]:
                    back += 1
                self.ptr, self.length = ptr, max(back, 1)
            self.table[key] = t + 1


def encode_symbol(enc, cum, sym, m):
    lo, hi = (int(cum[sym - 1]) if sym else 0), int(cum[sym])
    b = m.predict()
    if b < 0:
        enc.encode(lo, hi, int(cum[-1]))
    else:
        lb, pbin, q, fb, total = m.context(cum, b)
        hit = sym == b
        enc.encode(0, q, QTOTAL) if hit else enc.encode(q, QTOTAL, QTOTAL)
        m.hits[lb, pbin] += hit
        m.total[lb, pbin] += 1
        if not hit:
            if sym > b:
                lo, hi = lo - fb, hi - fb
            enc.encode(lo, hi, total - fb)
    m.update(sym)


def decode_symbol(dec, cum, m):
    b = m.predict()
    if b < 0:
        total = int(cum[-1])
        sym = int(np.searchsorted(cum, dec.target(total), side="right"))
        dec.consume(int(cum[sym - 1]) if sym else 0, int(cum[sym]), total)
    else:
        lb, pbin, q, fb, total = m.context(cum, b)
        hit = dec.target(QTOTAL) < q
        dec.consume(0, q, QTOTAL) if hit else dec.consume(q, QTOTAL, QTOTAL)
        m.hits[lb, pbin] += hit
        m.total[lb, pbin] += 1
        if hit:
            sym = b
        else:
            adj = cum.astype(np.int64).copy()
            adj[b:] -= fb  # b gets zero width, so it can't be selected
            sym = int(np.searchsorted(adj, dec.target(total - fb), side="right"))
            dec.consume(int(adj[sym - 1]) if sym else 0, int(adj[sym]), total - fb)
    m.update(sym)
    return sym


def encode_bytes(model, data):
    x = np.concatenate(([BOS], np.frombuffer(data, dtype=np.uint8).astype(np.int64)))
    enc, m = Encoder(), Match()
    for s, frm in window_starts(len(data)):
        inp = x[s:s + CTX][:len(x[s + 1:s + 1 + CTX])]
        cum = freqs(model, model.forward(inp, 0, model.new_caches()))
        tgt = x[s + 1:s + 1 + len(inp)]
        for j in range(frm, len(inp)):
            encode_symbol(enc, cum[j], int(tgt[j]), m)
    return enc.finish(), enc.ideal


def decode_bytes(model, code, n):
    dec, m = Decoder(code), Match()
    x = [BOS]
    for s, frm in window_starts(n):
        caches = model.new_caches()
        stop = min(s + CTX, n)
        logits = model.forward(x[s:s + frm + 1], 0, caches)
        x.append(decode_symbol(dec, freqs(model, logits[-1:])[0], m))
        while len(x) - 1 < stop:
            t0 = len(x) - 1 - s
            x.append(decode_symbol(dec, freqs(model, model.forward(x[-1:], t0, caches))[0], m))
    assert len(x) - 1 == n
    return bytes(x[1:])
