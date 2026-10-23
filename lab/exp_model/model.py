"""E8a: generalized steno codec with optional model improvements.

Same container and coder as lab/codec.py (v1); the model is configurable:

  case    words are coded lowercased + a case class (0 lower, 1 Capitalized,
          2 ALLCAPS, 3 mixed = token kept verbatim), predicted from the word
          and a sentence-position flag derived from the text decoded so far
  case2   as case, plus the previous word's case class (4 = none yet) in the context
  wrap    a single "\n" between a token and a following word is folded into
          that word's leading space (" word" + wrap flag); the flag is
          predicted from the column the word would end at and the document's
          observed line width so far
  order   max PPM context length in tokens (2 = v1, 3, 4)
  escape  C (v1: esc = distinct), D (counts 2c-1, esc = distinct),
          X (esc = singletons + 1, Good-Turing style),
          S (SEE: escape rate per bucket measured on held-out training books)
  sp2     spell UNK tokens with an order-2 byte model (Witten-Bell blend of
          order-2 counts over the order-1 Laplace model), integer-only

All counts are integers; encoder and decoder run the same state machine.

Usage (from project root):
  python3 lab/exp_model/model.py build NAME     NAME like "case+wrap+o3+D" or "base"
"""
import hashlib
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from codec import Decoder, Encoder, MAX_TOTAL, read_varint, varint  # noqa: E402
from steno import TOKEN, split_books  # noqa: E402

ROOT = HERE.parent.parent
TABLES = ROOT / "tables"
STORED, MODE = 0, 2
UNK = 0
SPELL_END = 256
WORD = re.compile(r" ?[A-Za-z]+")
SKIP = set(" \t\"'()[]_*“”‘’")
MAX_COL = 127
NO_INFO = 1 << 30


@dataclass(frozen=True)
class Config:
    case: bool = False
    wrap: bool = False
    order: int = 2
    escape: str = "C"
    sp2: bool = False
    min_ctx: int = 0  # orders >= 3 only use contexts whose total count is at least this
    case_prev: bool = False  # case2: case class context also includes the previous word's class

    @classmethod
    def parse(cls, name):
        parts = set(name.split("+")) - {"base"}
        order = 2
        escape = "C"
        min_ctx = 0
        for p in list(parts):
            if re.fullmatch(r"m\d+", p):
                min_ctx = int(p[1:])
                parts.discard(p)
            elif p in ("o2", "o3", "o4"):
                order = int(p[1])
                parts.discard(p)
            elif p in ("C", "D", "X", "S"):
                escape = p
                parts.discard(p)
        case2 = "case2" in parts
        cfg = cls("case" in parts or case2, "wrap" in parts, order, escape, "sp2" in parts, min_ctx, case2)
        unknown = parts - {"case", "case2", "wrap", "sp2"}
        assert not unknown, unknown
        return cfg

    @property
    def table_name(self):
        # escape method doesn't change the table
        case = ["case2" if self.case_prev else "case"] * self.case
        return "+".join(case + ["wrap"] * self.wrap + [f"o{self.order}"])


# --- text <-> events ----------------------------------------------------------

def case_class(tok):
    s = tok.lstrip(" ")
    if s.islower():
        return 0
    if s[0].isupper() and (len(s) == 1 or s[1:].islower()):
        return 1
    if s.isupper():
        return 2
    return 3


def apply_case(tok, cls):
    if cls == 1:
        i = 1 if tok.startswith(" ") else 0
        return tok[:i] + tok[i].upper() + tok[i + 1:]
    if cls == 2:
        return tok.upper()
    return tok


def events(text, cfg):
    """Split text into (vocab token, case class or None, wrap flag or None), losslessly."""
    toks = TOKEN.findall(text)
    out = []
    i = 0
    while i < len(toks):
        tok, wrapped = toks[i], None
        if (cfg.wrap and tok == "\n" and i > 0 and toks[i - 1] != "\n" and i + 1 < len(toks)
                and toks[i + 1][0] != " " and WORD.fullmatch(toks[i + 1])):
            tok, wrapped = " " + toks[i + 1], 1
            i += 1
        elif cfg.wrap and tok[0] == " " and WORD.fullmatch(tok):
            wrapped = 0
        cls = None
        if cfg.case and WORD.fullmatch(tok):
            cls = case_class(tok)
            if cls != 3:
                tok = tok.lower()
        out.append((tok, cls, wrapped))
        i += 1
    return out


def render(tok, cls, wrapped):
    if cls is not None:
        tok = apply_case(tok, cls)
    if wrapped:
        tok = "\n" + tok[1:]
    return tok


class State:
    """Deterministic context derived from the text produced so far."""

    def __init__(self):
        self.col = 0
        self.last_sig = None
        self.newline = False
        self.max_nowrap = -1
        self.min_wrap = NO_INFO
        self.last_cls = 4

    def case_ctx(self):
        if self.last_sig is None or self.last_sig in ".!?":
            return 1
        if self.newline:
            return 2
        if self.last_sig in ":;":
            return 3
        return 0

    def wrap_ctx(self, tok):
        end = min(self.col + len(tok), MAX_COL)
        if end <= self.max_nowrap:
            rel = 0
        elif end >= self.min_wrap:
            rel = 1
        else:
            rel = 2
        return end, rel

    def wrap_seen(self, end, flag):
        if flag:
            self.min_wrap = min(self.min_wrap, end)
        else:
            self.max_nowrap = max(self.max_nowrap, end)

    def emit(self, s):
        nl = s.rfind("\n")
        self.col = len(s) - nl - 1 if nl >= 0 else self.col + len(s)
        for ch in s:
            if ch == "\n":
                self.newline = True
            elif ch not in SKIP:
                self.last_sig = ch
                self.newline = False


# --- table --------------------------------------------------------------------

class Table:
    def __init__(self, arrays, cfg):
        self.cfg = cfg
        self.arrays = arrays
        blob, lens = arrays["vocab_blob"].tobytes(), arrays["vocab_lens"]
        offsets = [0] + np.cumsum(lens, dtype=np.int64).tolist()
        self.vocab = [blob[offsets[i]:offsets[i + 1]].decode() for i in range(len(lens))]
        self.vocab[UNK] = None
        self.index = {t: i for i, t in enumerate(self.vocab) if t is not None}
        self.b = len(self.vocab).bit_length()
        self.mask = (1 << self.b) - 1
        self.uni = arrays["uni"].astype(np.int64)
        self.uni_cum = np.cumsum(self.uni)
        self.total0 = int(self.uni_cum[-1])
        assert self.total0 < MAX_TOTAL
        self.spell_cum = np.cumsum(arrays["spell"].astype(np.int64) + 1, axis=1)
        K = range(1, cfg.order + 1)
        self.keys = {k: arrays[f"k{k}_key"] for k in K}  # sorted (ctx << b | sym)
        self.cnt = {k: arrays[f"k{k}_cnt"].astype(np.int64) for k in K}
        self.ctx = {k: arrays[f"k{k}_ctx"] for k in K}
        self.start = {k: arrays[f"k{k}_start"] for k in K}
        self.sym = {k: (self.keys[k] & self.mask).astype(np.int64) for k in K}
        if cfg.case:
            case = arrays["case"].astype(np.int64)  # (V, 4 flags, [5 previous classes,] 4 classes)
            prior = case.sum(axis=0)
            prior = 1 + (prior * 60) // np.maximum(prior.sum(axis=-1, keepdims=True), 1)
            self.case_cum = np.cumsum(16 * case + prior[None], axis=-1)
        if cfg.wrap:
            self.wrap_freq = arrays["wrap"].astype(np.int64) + 1  # (MAX_COL+1, 3 rel, 2)
        digest = hashlib.sha256(repr(cfg).encode())
        for name in sorted(arrays):
            digest.update(name.encode())
            digest.update(np.ascontiguousarray(arrays[name]).astype("<i8").tobytes())
        self.fingerprint = digest.digest()[:4]

    @classmethod
    def build(cls, texts, cfg):
        freq = Counter()
        for text in texts:  # events are recomputed in the second pass to save memory
            freq.update(t for t, _, _ in events(text, cfg))
        kept = sorted((t for t, c in freq.items() if c >= 2), key=lambda t: (-freq[t], t.encode()))
        vocab = ["<UNK>"] + kept
        index = {t: i for i, t in enumerate(vocab)}
        b = len(vocab).bit_length()
        V = len(vocab)
        case = np.zeros((V, 4, 5, 4) if cfg.case_prev else (V, 4, 4), dtype=np.int64)
        wrap = np.zeros((MAX_COL + 1, 3, 2), dtype=np.int64)
        id_arrays = []
        for text in texts:
            ev = events(text, cfg)
            ids = np.fromiter((index.get(t, UNK) for t, _, _ in ev), dtype=np.int64, count=len(ev))
            id_arrays.append(ids)
            if cfg.case or cfg.wrap:
                st = State()
                for (t, c, wr), w in zip(ev, ids.tolist()):
                    if c is not None:
                        if cfg.case_prev:
                            case[w, st.case_ctx(), st.last_cls, c] += 1
                        else:
                            case[w, st.case_ctx(), c] += 1
                        st.last_cls = c
                    if wr is not None:
                        end, rel = st.wrap_ctx(t)
                        wrap[end, rel, wr] += 1
                        st.wrap_seen(end, wr)
                    st.emit(render(t, c, wr))
        ids = np.concatenate(id_arrays)
        arrays = {
            "vocab_blob": np.frombuffer(b"".join(t.encode() for t in vocab), dtype=np.uint8),
            "vocab_lens": np.array([len(t.encode()) for t in vocab], dtype=np.uint32),
            "uni": np.bincount(ids, minlength=V).astype(np.uint32),
        }
        # order k context id: order 1 = previous token id; order k+1 = index of the
        # (k+1)-gram "context_k + token" in the sorted order-k key array
        n = len(ids)
        ctx_at = np.full(n, -1, dtype=np.int64)  # context id for predicting position i
        ctx_at[1:] = ids[:-1]
        for k in range(1, cfg.order + 1):
            valid = np.flatnonzero(ctx_at >= 0)
            key_at = np.full(n, -1, dtype=np.int64)
            key_at[valid] = (ctx_at[valid] << b) | ids[valid]
            uniq, counts = np.unique(key_at[valid], return_counts=True)
            ctx, start = np.unique(uniq >> b, return_index=True)
            arrays[f"k{k}_key"] = uniq
            arrays[f"k{k}_cnt"] = counts.astype(np.uint32)
            arrays[f"k{k}_ctx"] = ctx
            arrays[f"k{k}_start"] = np.append(start, len(uniq)).astype(np.uint32)
            if k < cfg.order:
                nxt = np.full(n, -1, dtype=np.int64)
                prev = key_at[:-1]
                ok = np.flatnonzero(prev >= 0)
                nxt[ok + 1] = np.searchsorted(uniq, prev[ok])
                ctx_at = nxt
        spell = np.zeros((257, 257), dtype=np.uint32)
        for tok, c in freq.items():
            if c == 1:
                seq = [SPELL_END] + list(tok.encode()) + [SPELL_END]
                np.add.at(spell, (seq[:-1], seq[1:]), 1)
        arrays["spell"] = spell
        if cfg.case:
            arrays["case"] = case.astype(np.uint32)
        if cfg.wrap:
            arrays["wrap"] = wrap.astype(np.uint32)
        return cls(arrays, cfg)

    def save(self, path):
        np.savez_compressed(path, **self.arrays)

    @classmethod
    def load(cls, path, cfg):
        with np.load(path) as data:
            return cls({k: data[k] for k in data.files}, cfg)

    def case_cum_for(self, w, st):
        if self.cfg.case_prev:
            return self.case_cum[w, st.case_ctx(), st.last_cls]
        return self.case_cum[w, st.case_ctx()]

    def spell_cum2(self, p2, p1):
        """Cumulative frequencies for the next spelled byte after bytes p2, p1."""
        if not self.cfg.sp2:
            return self.spell_cum[p1]
        f1 = self.spell_cum[p1]
        f1 = np.diff(f1, prepend=0)
        f1s = (f1 * 4096) // int(f1.sum()) + 1
        key = p2 * 257 + p1
        i = int(np.searchsorted(self.sp2_ctx, key))
        if i == len(self.sp2_ctx) or self.sp2_ctx[i] != key:
            return np.cumsum(f1s)
        c2 = self.sp2_rows[i]
        types = int((c2 > 0).sum())
        return np.cumsum(c2 * int(f1s.sum()) + types * f1s)

    def group(self, k, ctx):
        arr = self.ctx[k]
        i = int(np.searchsorted(arr, ctx))
        if i == len(arr) or arr[i] != ctx:
            return None
        s, e = int(self.start[k][i]), int(self.start[k][i + 1])
        return self.sym[k][s:e], self.cnt[k][s:e]

    def advance(self, nodes, w):
        """Context ids after token w, given the ids before it (nodes[k] for k=1..order)."""
        new = {1: w}
        for k in range(2, self.cfg.order + 1):
            prev = nodes.get(k - 1)
            if prev is None:
                break
            key = (prev << self.b) | w
            arr = self.keys[k - 1]
            i = int(np.searchsorted(arr, key))
            if i == len(arr) or arr[i] != key:
                break
            new[k] = i
        return new


def table_for(cfg):
    path = TABLES / f"exp_model_{cfg.table_name}.npz"
    if path.exists():
        table = Table.load(path, cfg)
    else:
        train, _ = split_books()
        table = Table.build([p.read_text(encoding="utf-8") for p in train], cfg)
        table.save(path)
    if cfg.sp2:
        sp_path = TABLES / f"exp_model_{cfg.table_name}_sp2.npz"
        if not sp_path.exists():
            np.savez_compressed(sp_path, **build_spell2(cfg))
        with np.load(sp_path) as data:
            table.sp2_ctx, table.sp2_rows = data["ctx"], data["rows"].astype(np.int64)
        assert table.sp2_rows.sum(axis=1).max() * 4353 * 2 < MAX_TOTAL
        table.fingerprint = hashlib.sha256(table.fingerprint + b"sp2" + table.sp2_rows.astype("<i8").tobytes()).digest()[:4]
    if cfg.escape == "S":
        see_path = TABLES / f"exp_model_{cfg.table_name}{f'_m{cfg.min_ctx}' if cfg.min_ctx else ''}_see.npy"
        if not see_path.exists():
            np.save(see_path, estimate_see(cfg))
        table.see = np.load(see_path)
        table.fingerprint = hashlib.sha256(table.fingerprint + table.see.astype("<i8").tobytes()).digest()[:4]
    return table


def build_spell2(cfg):
    """Order-2 byte counts over the rare (count 1) training tokens, the same tokens the order-1 model uses."""
    train, _ = split_books()
    freq = Counter()
    for p in train:
        freq.update(t for t, _, _ in events(p.read_text(encoding="utf-8"), cfg))
    keys = []
    for tok, c in freq.items():
        if c == 1:
            seq = [SPELL_END, SPELL_END] + list(tok.encode()) + [SPELL_END]
            keys.extend((seq[i - 2] * 257 + seq[i - 1]) * 257 + seq[i] for i in range(2, len(seq)))
    uniq, counts = np.unique(np.array(keys, dtype=np.int64), return_counts=True)
    ctx, inverse = np.unique(uniq // 257, return_inverse=True)
    rows = np.zeros((len(ctx), 257), dtype=np.uint32)
    rows[inverse, uniq % 257] = counts
    return {"ctx": ctx, "rows": rows}


SEE_D, SEE_N = 11, 21
SEE_HELDOUT_CHARS = 400_000
SEE_MIN_VISITS = 20


def estimate_see(cfg):
    """Escape rates per (order, first, distinct bucket, count bucket), as 16.16 fixed-point odds.

    Built on 80 training books, measured by walking the PPM contexts over the
    first SEE_HELDOUT_CHARS of each of the other 10 training books (never the test books).
    Sparse buckets fall back to coarser pools.
    """
    train, _ = split_books()
    held = train[::9]
    rest = [p for p in train if p not in held]
    table = Table.build([p.read_text(encoding="utf-8") for p in rest], cfg)
    shape = (cfg.order, 2, SEE_D, SEE_N)
    visits = np.zeros(shape, dtype=np.int64)
    escapes = np.zeros(shape, dtype=np.int64)
    for p in held:
        nodes = {}
        for tok, _, _ in events(p.read_text(encoding="utf-8")[:SEE_HELDOUT_CHARS], cfg):
            w = table.index.get(tok, UNK)
            excluded = np.empty(0, dtype=np.int64)
            for k in range(cfg.order, 0, -1):
                ctx = nodes.get(k)
                group = None if ctx is None else table.group(k, ctx)
                if group is None:
                    continue
                syms, cnts = group
                if k >= 3 and int(cnts.sum()) < cfg.min_ctx:
                    continue
                if excluded.size:
                    keep = ~np.isin(syms, excluded, assume_unique=True)
                    syms, cnts = syms[keep], cnts[keep]
                if syms.size == 0:
                    continue
                key = see_key(k, excluded.size == 0, int(syms.size), int(cnts.sum()))
                visits[key] += 1
                i = int(np.searchsorted(syms, w))
                if i < syms.size and syms[i] == w:
                    break
                escapes[key] += 1
                excluded = np.union1d(excluded, syms)
            nodes = table.advance(nodes, w)
    v, e = visits.astype(float), escapes.astype(float)
    pools = [(v, e), (v.sum(axis=3, keepdims=True), e.sum(axis=3, keepdims=True)),
             (v.sum(axis=(2, 3), keepdims=True), e.sum(axis=(2, 3), keepdims=True))]
    rate = None
    for pv, pe in reversed(pools):  # coarsest first, refined where there is enough data
        r = np.broadcast_to((pe + 0.5) / (pv + 1.0), shape)
        ok = np.broadcast_to(pv >= SEE_MIN_VISITS, shape)
        rate = np.where(ok, r, 0.5 if rate is None else rate)
    rate = np.clip(rate, 1e-4, 0.94)
    return np.round(65536 * rate / (1 - rate)).astype(np.int64)


# --- PPM ----------------------------------------------------------------------

def see_key(k, first, d, n):
    return k - 1, int(first), min(d.bit_length(), SEE_D - 1), min(n.bit_length(), SEE_N - 1)


def scaled(cnts, escape, table=None, k=0, first=True):
    """(symbol frequencies, escape frequency) for one context."""
    d = int(cnts.size)
    if escape == "S":
        n = int(cnts.sum())
        shift = min(8, max(0, 25 - n.bit_length()))
        ratio = int(table.see[see_key(k, first, d, n)])
        return cnts << shift, max(1, ((n << shift) * ratio) >> 16)
    if escape == "C":
        return cnts, d
    if escape == "D":
        return 2 * cnts - 1, d
    if escape == "X":
        return cnts, int((cnts == 1).sum()) + 1
    raise ValueError(escape)


def contexts(table, nodes):
    excluded = np.empty(0, dtype=np.int64)
    for k in range(table.cfg.order, 0, -1):
        ctx = nodes.get(k)
        if ctx is None:
            continue
        group = table.group(k, ctx)
        if group is None:
            continue
        syms, cnts = group
        if k >= 3 and int(cnts.sum()) < table.cfg.min_ctx:
            continue
        if excluded.size:
            keep = ~np.isin(syms, excluded, assume_unique=True)
            syms, cnts = syms[keep], cnts[keep]
        if syms.size == 0:
            continue
        freqs, esc = scaled(cnts, table.cfg.escape, table, k, excluded.size == 0)
        yield syms, freqs, esc
        excluded = np.union1d(excluded, syms)
    yield excluded, None, None


def order0_cum(table, excluded):
    if excluded.size == 0:
        return table.uni_cum
    removed = np.zeros(len(table.uni), dtype=np.int64)
    removed[excluded] = table.uni[excluded]
    return table.uni_cum - np.cumsum(removed)


def encode_token(enc, table, w, nodes):
    for syms, freqs, esc in contexts(table, nodes):
        if freqs is None:
            ex = syms
            if ex.size:
                total = table.total0 - int(table.uni[ex].sum())
                below = int(table.uni[ex[ex < w]].sum())
            else:
                total, below = int(table.uni_cum[-1]), 0
            low = (int(table.uni_cum[w - 1]) if w else 0) - below
            enc.encode(low, low + int(table.uni[w]), total)
            return
        total = int(freqs.sum()) + esc
        assert total < MAX_TOTAL
        i = int(np.searchsorted(syms, w))
        if i < syms.size and syms[i] == w:
            low = int(freqs[:i].sum())
            enc.encode(low, low + int(freqs[i]), total)
            return
        enc.encode(total - esc, total, total)


def decode_token(dec, table, nodes):
    for syms, freqs, esc in contexts(table, nodes):
        if freqs is None:
            cum = order0_cum(table, syms)
            total = int(cum[-1])
            w = int(np.searchsorted(cum, dec.target(total), side="right"))
            dec.consume(int(cum[w - 1]) if w else 0, int(cum[w]), total)
            return w
        total = int(freqs.sum()) + esc
        target = dec.target(total)
        if target >= total - esc:
            dec.consume(total - esc, total, total)
            continue
        cum = np.cumsum(freqs)
        i = int(np.searchsorted(cum, target, side="right"))
        dec.consume(int(cum[i] - freqs[i]), int(cum[i]), total)
        return int(syms[i])
    raise AssertionError("unreachable")


def encode_cum(enc, cum, sym):
    enc.encode(int(cum[sym - 1]) if sym else 0, int(cum[sym]), int(cum[-1]))


def decode_cum(dec, cum):
    total = int(cum[-1])
    sym = int(np.searchsorted(cum, dec.target(total), side="right"))
    dec.consume(int(cum[sym - 1]) if sym else 0, int(cum[sym]), total)
    return sym


def binary_cum(freq):
    return (int(freq[0]), int(freq[0] + freq[1]))


# --- container ----------------------------------------------------------------

def compress(table, data):
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return bytes([STORED]) + data
    cfg = table.cfg
    ev = events(text, cfg)
    enc = Encoder()
    nodes, st = {}, State()
    for tok, cls, wrapped in ev:
        w = table.index.get(tok, UNK)
        encode_token(enc, table, w, nodes)
        if w == UNK:
            p2 = p1 = SPELL_END
            for byte in list(tok.encode()) + [SPELL_END]:
                encode_cum(enc, table.spell_cum2(p2, p1), byte)
                p2, p1 = p1, byte
        if cls is not None:
            encode_cum(enc, table.case_cum_for(w, st), cls)
            st.last_cls = cls
        if wrapped is not None:
            end, rel = st.wrap_ctx(tok)
            encode_cum(enc, np.cumsum(table.wrap_freq[end, rel]), wrapped)
            st.wrap_seen(end, wrapped)
        st.emit(render(tok, cls, wrapped))
        nodes = table.advance(nodes, w)
    out = bytes([MODE]) + table.fingerprint + varint(len(ev)) + enc.finish()
    if len(out) >= 1 + len(data):
        return bytes([STORED]) + data
    return out


def decompress(table, blob):
    if blob[0] == STORED:
        return blob[1:]
    assert blob[0] == MODE and blob[1:5] == table.fingerprint, "wrong mode or table"
    cfg = table.cfg
    count, pos = read_varint(blob, 5)
    dec = Decoder(blob[pos:])
    nodes, st, parts = {}, State(), []
    for _ in range(count):
        w = decode_token(dec, table, nodes)
        if w == UNK:
            raw, p2, p1 = bytearray(), SPELL_END, SPELL_END
            while (byte := decode_cum(dec, table.spell_cum2(p2, p1))) != SPELL_END:
                raw.append(byte)
                p2, p1 = p1, byte
            tok = raw.decode()
        else:
            tok = table.vocab[w]
        cls = wrapped = None
        if cfg.case and WORD.fullmatch(tok):
            cls = decode_cum(dec, table.case_cum_for(w, st))
            st.last_cls = cls
        if cfg.wrap and tok[0] == " " and WORD.fullmatch(tok):
            end, rel = st.wrap_ctx(tok)
            wrapped = decode_cum(dec, np.cumsum(table.wrap_freq[end, rel]))
            st.wrap_seen(end, wrapped)
        s = render(tok, cls, wrapped)
        st.emit(s)
        parts.append(s)
        nodes = table.advance(nodes, w)
    return "".join(parts).encode()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "build":
        cfg = Config.parse(sys.argv[2])
        t = table_for(cfg)
        print(cfg.table_name, len(t.vocab), t.fingerprint.hex())
    else:
        sys.exit(__doc__)
