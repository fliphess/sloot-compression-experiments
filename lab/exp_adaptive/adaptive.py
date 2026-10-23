"""E8b: steno codec with a per-document adaptive layer on top of the frozen table.

The frozen table (tables/en-gutenberg-v1.npz) is never modified. While coding,
a document layer counts what the document has shown so far; the decoder
rebuilds exactly the same layer from the tokens it has decoded, so nothing
extra is stored. All arithmetic is integer.

Document layer:
  doc_uni[id]           order-0 counts (Fenwick tree for prefix sums)
  doc_ctx[k][ctx][id]   counts after the last k actual token ids (k = 1..3)
  doc_spell[prev][b]    letter-pair counts of spelled (new) words
  new words             a token not in the table vocab is coded as UNK + spelling
                        the first time; it then gets id V + n (n = 0, 1, ...)
                        and from then on is a normal symbol. For table lookups
                        the history maps new ids back to UNK (the table knows
                        what follows rare words); document contexts use the
                        actual ids.

Model: a PPM walk over a configurable list of levels, highest first, with full
exclusion, escape count = number of remaining candidates (method C):
  ("T", k)      table-only order-k context
  ("M", k, W)   table counts + W * document counts, same order-k context
  ("D", k, W)   document-only order-k context, counts * W
then a final order-0 level over all vocab + new ids: table count + W0 * doc
count, no escape. Candidate order inside a level: table symbols (ascending id)
followed by document-only symbols (ascending id).

Spelling: table letter counts + 1 (Laplace) + WS * document letter counts.

Container: byte 0 = mode (0 stored, 2 adaptive), bytes 1-4 table fingerprint,
varint token count, arithmetic code (same coder as lab/codec.py).
"""
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from codec import (STORED, SPELL_END, UNK, Decoder, Encoder, FrozenTable,  # noqa: E402
                   read_varint, varint)
from steno import TOKEN  # noqa: E402

ADAPTIVE = 2
MAX_NEW = 1 << 19  # new words per document


@dataclass(frozen=True)
class Config:
    name: str
    levels: tuple  # e.g. (("M", 2, 4), ("M", 1, 4))
    w0: int = 0  # weight of document counts at order 0
    new_words: bool = False  # give new words their own id after the first spelling
    ws: int = 0  # weight of document letter counts in the spelling model

    def __post_init__(self):
        # a new word id has no table count, so it needs a document count at order 0
        assert self.w0 >= 1 or not self.new_words, self.name

    def doc_orders(self):
        return sorted({lv[1] for lv in self.levels if lv[0] in "MD"})


class Fenwick:
    def __init__(self, n):
        self.n = n
        self.tree = [0] * (n + 1)

    def add(self, i, delta):
        i += 1
        while i <= self.n:
            self.tree[i] += delta
            i += i & -i

    def prefix(self, i):
        """Sum of entries 0..i-1."""
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & -i
        return s


class DocState:
    def __init__(self, table, cfg):
        self.t = table
        self.cfg = cfg
        self.V = len(table.vocab)
        size = self.V + MAX_NEW
        self.uni_ext = np.zeros(size, dtype=np.int64)
        self.uni_ext[:self.V] = table.uni
        self.doc_uni = np.zeros(size, dtype=np.int64)
        self.fen = Fenwick(size)
        self.doc_total = 0
        self.doc_ctx = {k: defaultdict(dict) for k in (1, 2, 3)}
        self.doc_spell = np.zeros((257, 257), dtype=np.int64)
        self.new_ids = {}
        self.new_tokens = []

    # --- symbol ids ---
    def lookup(self, tok):
        w = self.t.index.get(tok)
        if w is not None:
            return w
        return self.new_ids.get(tok, UNK)

    def n_symbols(self):
        return self.V + len(self.new_tokens)

    def token(self, w):
        return self.t.vocab[w] if w < self.V else self.new_tokens[w - self.V]

    # --- levels ---
    def level(self, spec, hist, tab_hist):
        kind, k = spec[0], spec[1]
        if len(hist) < k:
            return None
        if kind == "T" or kind == "M":
            group = self.t.group(k, tab_hist)
            tsyms, tcnts = group if group is not None else (np.empty(0, np.int64), np.empty(0, np.int64))
        if kind == "T":
            return (tsyms, tcnts) if tsyms.size else None
        doc = self.doc_ctx[k].get(tuple(hist[-k:]))
        weight = spec[2]
        if kind == "D":
            if not doc:
                return None
            ds = np.fromiter(doc.keys(), dtype=np.int64, count=len(doc))
            dc = np.fromiter(doc.values(), dtype=np.int64, count=len(doc)) * weight
            order = np.argsort(ds)
            return ds[order], dc[order]
        # "M": table + weighted doc counts
        if not doc:
            return (tsyms, tcnts) if tsyms.size else None
        ds = np.fromiter(doc.keys(), dtype=np.int64, count=len(doc))
        dc = np.fromiter(doc.values(), dtype=np.int64, count=len(doc)) * weight
        cnts = tcnts.copy()
        if tsyms.size:
            pos = np.searchsorted(tsyms, ds)
            pos_c = np.minimum(pos, tsyms.size - 1)
            found = (pos < tsyms.size) & (tsyms[pos_c] == ds)
            cnts[pos[found]] += dc[found]
            ds, dc = ds[~found], dc[~found]
        order = np.argsort(ds)
        return np.concatenate((tsyms, ds[order])), np.concatenate((cnts, dc[order]))

    def candidates(self, hist):
        """Yield (syms, cnts) per non-empty level with exclusions applied; then (excluded, None)."""
        tab_hist = [w if w < self.V else UNK for w in hist]
        excluded = np.empty(0, dtype=np.int64)
        for spec in self.cfg.levels:
            lv = self.level(spec, hist, tab_hist)
            if lv is None:
                continue
            syms, cnts = lv
            if excluded.size:
                keep = ~np.isin(syms, excluded, assume_unique=True)
                syms, cnts = syms[keep], cnts[keep]
            if syms.size == 0:
                continue
            yield syms, cnts
            excluded = np.union1d(excluded, syms)
        yield excluded, None

    # --- order 0 ---
    def order0(self, excluded):
        w0 = self.cfg.w0
        if excluded.size:
            ecnt = self.uni_ext[excluded] + w0 * self.doc_uni[excluded]
            eprefix = np.cumsum(ecnt)
            etotal = int(eprefix[-1])
        else:
            eprefix, etotal = None, 0
        total = self.t.total0 + w0 * self.doc_total - etotal
        return eprefix, total

    def cum_before(self, w, excluded, eprefix):
        tab = int(self.t.uni_cum[w - 1]) if 0 < w <= self.V else (self.t.total0 if w > self.V else 0)
        doc = self.cfg.w0 * self.fen.prefix(w) if self.cfg.w0 else 0
        excl = 0
        if eprefix is not None:
            i = int(np.searchsorted(excluded, w, side="left"))
            excl = int(eprefix[i - 1]) if i else 0
        return tab + doc - excl

    def width0(self, w):
        return int(self.uni_ext[w]) + self.cfg.w0 * int(self.doc_uni[w])

    # --- spelling ---
    def spell_row(self, prev):
        row = self.t.spell_cum[prev]
        if self.cfg.ws:
            row = row + self.cfg.ws * np.cumsum(self.doc_spell[prev])
        return row

    # --- learning (identical in encoder and decoder) ---
    def learn(self, tok, w, hist):
        spelled = w == UNK
        if spelled and self.cfg.new_words:
            w = self.V + len(self.new_tokens)
            self.new_ids[tok] = w
            self.new_tokens.append(tok)
        if self.cfg.ws and spelled:
            seq = [SPELL_END] + list(tok.encode()) + [SPELL_END]
            np.add.at(self.doc_spell, (seq[:-1], seq[1:]), 1)
        self.doc_uni[w] += 1
        self.fen.add(w, 1)
        self.doc_total += 1
        for k in self.cfg.doc_orders():
            if len(hist) >= k:
                ctx = self.doc_ctx[k][tuple(hist[-k:])]
                ctx[w] = ctx.get(w, 0) + 1
        return w  # id used in the history


def encode_token(enc, st, w, hist):
    for syms, cnts in st.candidates(hist):
        if cnts is None:
            eprefix, total = st.order0(syms)
            low = st.cum_before(w, syms, eprefix)
            enc.encode(low, low + st.width0(w), total)
            return
        esc = int(syms.size)
        total = int(cnts.sum()) + esc
        hit = np.flatnonzero(syms == w)
        if hit.size:
            i = int(hit[0])
            low = int(cnts[:i].sum())
            enc.encode(low, low + int(cnts[i]), total)
            return
        enc.encode(total - esc, total, total)


def decode_token(dec, st, hist):
    for syms, cnts in st.candidates(hist):
        if cnts is None:
            eprefix, total = st.order0(syms)
            target = dec.target(total)
            lo, hi = 0, st.n_symbols() - 1  # smallest w with cum_before(w + 1) > target
            while lo < hi:
                mid = (lo + hi) // 2
                if st.cum_before(mid + 1, syms, eprefix) > target:
                    hi = mid
                else:
                    lo = mid + 1
            low = st.cum_before(lo, syms, eprefix)
            dec.consume(low, low + st.width0(lo), total)
            return lo
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


def encode_spelling(enc, st, raw):
    prev = SPELL_END
    for b in list(raw) + [SPELL_END]:
        row = st.spell_row(prev)
        enc.encode(int(row[b - 1]) if b else 0, int(row[b]), int(row[-1]))
        prev = b


def decode_spelling(dec, st):
    out, prev = bytearray(), SPELL_END
    while True:
        row = st.spell_row(prev)
        total = int(row[-1])
        b = int(np.searchsorted(row, dec.target(total), side="right"))
        dec.consume(int(row[b - 1]) if b else 0, int(row[b]), total)
        if b == SPELL_END:
            return bytes(out)
        out.append(b)
        prev = b


def compress(table, cfg, data):
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return bytes([STORED]) + data
    tokens = TOKEN.findall(text)
    st = DocState(table, cfg)
    enc = Encoder()
    hist = []
    for tok in tokens:
        w = st.lookup(tok)
        encode_token(enc, st, w, hist)
        if w == UNK:
            encode_spelling(enc, st, tok.encode())
        hid = st.learn(tok, w, hist)
        hist = (hist + [hid])[-3:]
    out = bytes([ADAPTIVE]) + table.fingerprint + varint(len(tokens)) + enc.finish()
    if len(out) >= 1 + len(data):
        return bytes([STORED]) + data
    return out


def decompress(table, cfg, blob):
    if blob[0] == STORED:
        return blob[1:]
    if blob[0] != ADAPTIVE:
        raise ValueError(f"unknown mode {blob[0]}")
    if blob[1:5] != table.fingerprint:
        raise ValueError(f"made with table {blob[1:5].hex()}, this table is {table.fingerprint.hex()}")
    count, pos = read_varint(blob, 5)
    dec = Decoder(blob[pos:])
    st = DocState(table, cfg)
    parts, hist = [], []
    for _ in range(count):
        w = decode_token(dec, st, hist)
        if w == UNK:
            tok = decode_spelling(dec, st).decode()
        else:
            tok = st.token(w)
        parts.append(tok)
        hid = st.learn(tok, w, hist)
        hist = (hist + [hid])[-3:]
    return "".join(parts).encode()
