"""E12: real compressed files from the integer byte transformer (intlm.py) + codec v1's arithmetic coder.

Frequency rule (integer logits l, Q10, 257 symbols; BOS is never coded but keeps freq >= 1):
  e_i    = EXP[min(max(l) - l_i, 32768)]           (Q30 table of exp(-x/1024), from the model file)
  f_i    = 1 + floor(e_i * 2^24 / sum(e))           so every f_i >= 1, total <= 257 + 2^24 < 2^30
  cum    = prefix sums in symbol order 0..256

Context semantics = E10's evaluation exactly: the context is reset per document; [BOS] + data is
cut into windows of 512 inputs starting every 256 positions, positions restart at 0 per window;
window 0 codes all its targets, later windows only their last 256 (each byte coded exactly once,
with >= 256 bytes of context once past the first window). The encoder runs each window as one
block; the decoder prefills the 256 known inputs of a window, then decodes token by token with a
KV cache. Both produce identical integers because every integer op is row-wise.

File:
  byte 0      mode: 0 = stored (raw bytes follow), 2 = integer byte transformer v1
  bytes 1-4   model fingerprint (first 4 bytes of the SHA-256 of the model file)
  varint      number of bytes
  rest        arithmetic code (WNC 32-bit, lab/codec.py); missing trailing bytes read as zero

Usage:
  .venv/bin/python lab/round2/intmodel/nncodec.py compress MODEL IN OUT [--backend int|f64]
  .venv/bin/python lab/round2/intmodel/nncodec.py decompress MODEL IN OUT [--backend int|f64]
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]

from codec import Decoder, Encoder, read_varint, varint  # noqa: E402
from intlm import BOS, CTX, EXP_LEN, STRIDE, IntLM  # noqa: E402

STORED, NEURAL_V1 = 0, 2
FREQ_BITS = 24


def freqs(model, logits):
    """(rows, 257) int logits -> (rows, 257) cumulative frequencies (int64)."""
    d = np.minimum(logits.max(axis=-1, keepdims=True) - logits, EXP_LEN)
    e = model.exp30[d]
    f = 1 + (e << FREQ_BITS) // e.sum(axis=-1, keepdims=True)
    return np.cumsum(f, axis=-1)


def window_starts(n):
    """Window starts s and the first target index coded in it (targets are data[0..n-1])."""
    s = 0
    while True:
        yield s, (0 if s == 0 else CTX - STRIDE)
        if s + CTX >= n:
            return
        s += STRIDE


def encode_bytes(model, data, stats=None):
    x = np.concatenate(([BOS], np.frombuffer(data, dtype=np.uint8).astype(np.int64)))
    n = len(data)
    enc = Encoder()
    for s, frm in window_starts(n):
        inp = x[s:s + CTX][:len(x[s + 1:s + 1 + CTX])]
        cum = freqs(model, model.forward(inp, 0, model.new_caches()))
        tgt = x[s + 1:s + 1 + len(inp)]
        for j in range(frm, len(inp)):
            c, sym = cum[j], int(tgt[j])
            enc.encode(int(c[sym - 1]) if sym else 0, int(c[sym]), int(c[-1]))
    if stats is not None:
        stats["ideal_bits"] = enc.ideal
    return enc.finish()


def decode_bytes(model, code, n):
    dec = Decoder(code)
    x = [BOS]

    def step(cum):
        total = int(cum[-1])
        sym = int(np.searchsorted(cum, dec.target(total), side="right"))
        dec.consume(int(cum[sym - 1]) if sym else 0, int(cum[sym]), total)
        x.append(sym)

    for s, frm in window_starts(n):
        caches = model.new_caches()
        stop = min(s + CTX, n)  # last input position + 1 in this window (inputs x[s:stop])
        logits = model.forward(x[s:s + frm + 1], 0, caches)  # prefill the known context
        step(freqs(model, logits[-1:])[0])
        while len(x) - 1 < stop:  # targets x[s+1 .. stop]; next input is x[-1] at position len(x)-1-s
            t0 = len(x) - 1 - s
            step(freqs(model, model.forward(x[-1:], t0, caches))[0])
    assert len(x) - 1 == n
    return bytes(x[1:])


def window_exists(s, n):
    return s == 0 or s + (CTX - STRIDE) < n


def encode_many(model, docs, chunk=8):
    """Batched encode_bytes: same codes, windows of different documents share forward passes."""
    xs = [np.concatenate(([BOS], np.frombuffer(d, dtype=np.uint8).astype(np.int64))) for d in docs]
    encs = [Encoder() for _ in docs]
    s = 0
    while True:
        active = [i for i, d in enumerate(docs) if d and window_exists(s, len(d))]
        if not active:
            break
        frm = 0 if s == 0 else CTX - STRIDE
        by_len = {}
        for i in active:
            by_len.setdefault(min(CTX, len(docs[i]) - s), []).append(i)
        for length, ids in by_len.items():
            for c in range(0, len(ids), chunk):
                part = ids[c:c + chunk]
                inp = np.stack([xs[i][s:s + length] for i in part])
                cum = freqs(model, model.forward(inp, 0, model.new_caches()))
                for row, i in enumerate(part):
                    tgt = xs[i][s + 1:s + 1 + length]
                    for j in range(frm, length):
                        cr, sym = cum[row, j], int(tgt[j])
                        encs[i].encode(int(cr[sym - 1]) if sym else 0, int(cr[sym]), int(cr[-1]))
        s += STRIDE
    return [e.finish() for e in encs], [e.ideal for e in encs]


def decode_many(model, items):
    """Batched decode_bytes for [(code, n), ...]: all documents advance in lockstep (same window, same position)."""
    decs = [Decoder(code) for code, _ in items]
    ns = [n for _, n in items]
    xs = [[BOS] for _ in items]

    def step(i, cum):
        total = int(cum[-1])
        sym = int(np.searchsorted(cum, decs[i].target(total), side="right"))
        decs[i].consume(int(cum[sym - 1]) if sym else 0, int(cum[sym]), total)
        xs[i].append(sym)

    s = 0
    while True:
        active = [i for i, n in enumerate(ns) if n and window_exists(s, n)]
        if not active:
            break
        frm = 0 if s == 0 else CTX - STRIDE
        stop = {i: min(s + CTX, ns[i]) for i in active}
        caches = model.new_caches()
        cum = freqs(model, model.forward(np.array([xs[i][s:s + frm + 1] for i in active]), 0, caches)[:, -1])
        for row, i in enumerate(active):
            step(i, cum[row])
        t0 = frm + 1
        while any(len(xs[i]) - 1 < stop[i] for i in active):
            # unfinished documents feed their last byte (position t0); finished ones a dummy 0, output ignored
            inp = np.array([[xs[i][-1] if len(xs[i]) - 1 < stop[i] else 0] for i in active])
            cum = freqs(model, model.forward(inp, t0, caches)[:, 0])
            for row, i in enumerate(active):
                if len(xs[i]) - 1 < stop[i]:
                    step(i, cum[row])
            t0 += 1
        s += STRIDE
    return [bytes(x[1:]) for x in xs]


def compress_many(model, docs):
    codes, ideals = encode_many(model, docs)
    out = []
    for data, code in zip(docs, codes):
        blob = bytes([NEURAL_V1]) + model.fingerprint + varint(len(data)) + code if data else b""
        out.append(blob if data and len(blob) < 1 + len(data) else bytes([STORED]) + data)
    return out, ideals


def decompress_many(model, blobs):
    out = [None] * len(blobs)
    todo = []
    for k, blob in enumerate(blobs):
        if blob[0] == STORED:
            out[k] = blob[1:]
        else:
            if blob[0] != NEURAL_V1 or blob[1:5] != model.fingerprint:
                raise ValueError(f"blob {k}: mode {blob[0]} / model {blob[1:5].hex()} not readable with {model.fingerprint.hex()}")
            n, pos = read_varint(blob, 5)
            todo.append((k, blob[pos:], n))
    for (k, _, _), data in zip(todo, decode_many(model, [(c, n) for _, c, n in todo])):
        out[k] = data
    return out


def compress(model, data, stats=None):
    if not data:
        return bytes([STORED])
    out = bytes([NEURAL_V1]) + model.fingerprint + varint(len(data)) + encode_bytes(model, data, stats)
    if len(out) >= 1 + len(data):
        return bytes([STORED]) + data
    return out


def decompress(model, blob):
    if blob[0] == STORED:
        return blob[1:]
    if blob[0] != NEURAL_V1:
        raise ValueError(f"unknown mode {blob[0]}")
    if blob[1:5] != model.fingerprint:
        raise ValueError(f"made with model {blob[1:5].hex()}, this model is {model.fingerprint.hex()}")
    n, pos = read_varint(blob, 5)
    return decode_bytes(model, blob[pos:], n)


def main():
    args = sys.argv[1:]
    backend = "int"
    if "--backend" in args:
        i = args.index("--backend")
        backend = args[i + 1]
        del args[i:i + 2]
    cmd, model_path, src, dst = args
    model = IntLM(model_path, backend)
    model.selftest()
    data = Path(src).read_bytes()
    Path(dst).write_bytes(compress(model, data) if cmd == "compress" else decompress(model, data))


if __name__ == "__main__":
    main()
