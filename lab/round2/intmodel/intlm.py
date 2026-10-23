"""E12: integer-only inference for the E10 byte transformer (numpy, no floats).

The model file (tables/intmodel_*.bin, written by export.py) is the artifact:

  magic   b"JSINTLM1"
  u32 LE  header length H
  H bytes header: JSON (utf-8, sorted keys): config, fixed-point formats, array list
  arrays  raw little-endian, in header order, each (name, dtype, shape)

Fingerprint = SHA-256 of the whole file; the codec container stores its first 4 bytes.

Number formats (QF = integer with F fractional bits):
  residual stream  Q{f_res}, clamped to |x| < 2^24
  LayerNorm out    Q12 (gamma Q16, beta Q12)
  q, k, v          Q{f_qkv[l]} per layer, clamped to |x| < 2^20
  scores           Q10 (after 1/sqrt(dh)), exp via table EXP (Q30) over [-32, 0]
  fc out / GELU    Q10, GELU by table on [-8, 8), identity above, 0 below
  logits           Q10

Every Linear: the input row X (QF_in, |X| < 2^21) is quantised to int8 with its own
row max m (q = round(127 X / m)); acc = q . Wq (int8 x int8, exact int);
y = floor((acc * m * M[c] + 2^(S[c]-1)) / 2^S[c]) + bias[c]   (QF_out)
where M[c], S[c] encode s_w[c] * 2^(F_out - F_in) / 127 with M in [2^14, 2^15].
All shifts are floor divisions by powers of two (well defined for negatives).

Overflow: every int64 intermediate is bounded below 2^62 by construction (see
RESULTS.md, "bounds"); the clamps make the bounds hold for any input.

Matmul backends (the result is the same integer either way):
  "int"  numpy integer matmul (int32 for int8 x int8, int64 for attention); no BLAS, no floats
  "f32"  like f64 for attention; the int8 x int8 linears use fp32 BLAS in K chunks of <= 1024,
         exact because 1024 * 127^2 < 2^24; chunk results are added in int64. Half the memory
         traffic of f64, the default for the measurements.
  "f64"  float64 BLAS on integer-valued operands. Exact because every product and every
         partial sum is an integer of magnitude < 2^53, so IEEE-754 binary64 represents all
         of them exactly, whatever the summation order, blocking, FMA use or thread count.
         An accelerator only; the reference definition is the integer one.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

MAGIC = b"JSINTLM1"
BOS = 256
VOCAB = 257
CTX = 512
STRIDE = 256

SELFTEST = [BOS] + list(b"Integer self-test: The quick brown fox jumps over the lazy dog. 0123456789\n")
RES_LIM = (1 << 24) - 1
LIN_LIM = (1 << 21) - 1
QKV_LIM = (1 << 20) - 1
FC_LIM = (1 << 20) - 1
F_LN = 12
F_G = 16  # LayerNorm gamma
F_S = 10  # scores, GELU, logits
EXP_LEN = 32 * 1024  # EXP[i] = round(2^30 exp(-i / 1024)), i = 0..EXP_LEN
GELU_HALF = 8 * 1024  # GELU table covers [-8, 8) in Q10


def rshift(v, s):
    """Round-half-up shift right: floor((v + 2^(s-1)) / 2^s), s >= 1 (scalar or array)."""
    d = np.left_shift(np.int64(1), np.asarray(s, dtype=np.int64))
    return (v + (d >> 1)) // d


def round_div(a, b):
    """Round-half-up a / b for b > 0."""
    return (2 * a + b) // (2 * b)


def isqrt(v):
    """Exact floor(sqrt(v)) per element of an int64 array (Python's integer math.isqrt)."""
    return np.array([math.isqrt(int(a)) for a in v.ravel()], dtype=np.int64).reshape(v.shape)


def quant_rows(x):
    """Per-row symmetric int8: returns (q int8-valued int64, m = row max |x| >= 1)."""
    x = np.clip(x, -LIN_LIM, LIN_LIM)
    a = np.abs(x)
    m = np.maximum(a.max(axis=-1, keepdims=True), 1)
    q = (a * 254 + m) // (2 * m)
    return np.where(x < 0, -q, q), m


def load(path):
    blob = Path(path).read_bytes()
    assert blob[:8] == MAGIC, "not an integer model file"
    hlen = int.from_bytes(blob[8:12], "little")
    header = json.loads(blob[12:12 + hlen].decode())
    pos = 12 + hlen
    arrays = {}
    for name, dtype, shape in header["arrays"]:
        dt = np.dtype(dtype).newbyteorder("<")
        n = int(np.prod(shape)) * dt.itemsize
        arrays[name] = np.frombuffer(blob, dtype=dt, count=int(np.prod(shape)), offset=pos).reshape(shape).astype(dt.newbyteorder("="))
        pos += n
    assert pos == len(blob)
    return header, arrays, hashlib.sha256(blob).digest()


F32_CHUNK = 1024  # 1024 * 127 * 127 = 16,516,096 < 2^24: an fp32 dot product of this length is exact


def int8_matmul(q, wt, backend):
    """Exact q @ wt for int8-valued q (..., K) and prepared weights wt (see prep_weights)."""
    lead = q.shape[:-1]
    q = q.reshape(-1, q.shape[-1])  # one 2-D product (a 3-D one would loop over the batch, re-reading wt)
    if backend == "int":
        acc = (q.astype(np.int32) @ wt).astype(np.int64)
    elif backend == "f64":
        acc = (q.astype(np.float64) @ wt).astype(np.int64)
    else:
        qf = q.astype(np.float32)
        acc = None
        for k0, w in wt:  # fp32, one exact partial product per K chunk, summed in int64
            part = (qf[:, k0:k0 + w.shape[0]] @ w).astype(np.int64)
            acc = part if acc is None else acc + part
    return acc.reshape(*lead, acc.shape[-1])


def prep_weights(w, backend):
    """int8 (out, in) -> transposed operand for int8_matmul."""
    wt = w.T
    if backend == "int":
        return wt.astype(np.int32).copy()
    if backend == "f64":
        return wt.astype(np.float64).copy()
    return [(k0, wt[k0:k0 + F32_CHUNK].astype(np.float32).copy()) for k0 in range(0, wt.shape[0], F32_CHUNK)]


class Linear:
    def __init__(self, a, prefix, backend):
        self.w = a[prefix + ".w"]  # int8 (out, in)
        self.mul = a[prefix + ".m"].astype(np.int64)
        self.div = np.left_shift(np.int64(1), a[prefix + ".s"].astype(np.int64))
        self.half = self.div >> 1
        self.bias = a[prefix + ".b"].astype(np.int64)
        self.backend = backend
        self.wt = prep_weights(self.w, backend)

    def __call__(self, x):
        q, m = quant_rows(x)
        acc = int8_matmul(q, self.wt, self.backend)
        return (acc * m * self.mul + self.half) // self.div + self.bias


class IntLM:
    def __init__(self, path, backend="int"):
        self.header, a, self.digest = load(path)
        self.fingerprint = self.digest[:4]
        h = self.header
        self.d, self.layers, self.heads = h["d"], h["layers"], h["heads"]
        self.dh = self.d // self.heads
        self.backend = backend
        self.f_res = h["f_res"]
        self.exp30 = a["exp"].astype(np.int64)  # Q30, length EXP_LEN + 1
        self.exp16 = self.exp30 >> 14
        self.gelu = a["gelu"].astype(np.int64)  # Q10 for x in [-8, 8)
        # embedding tables in the residual format, computed in integers at load time
        self.emb_q = a["emb.w"].astype(np.int64)
        self.emb_res = rshift(self.emb_q * a["emb.mr"].astype(np.int64)[:, None], a["emb.sr"].astype(np.int64)[:, None])
        pos_q = a["pos.w"].astype(np.int64)
        self.pos_res = rshift(pos_q * a["pos.mr"].astype(np.int64)[:, None], a["pos.sr"].astype(np.int64)[:, None])
        self.head_mul = a["emb.mh"].astype(np.int64)
        self.head_div = np.left_shift(np.int64(1), a["emb.sh"].astype(np.int64))
        self.head_wt = prep_weights(self.emb_q, backend)
        self.lnf = (a["lnf.g"].astype(np.int64), a["lnf.b"].astype(np.int64))
        self.blocks = []
        for i in range(self.layers):
            p = f"b{i}."
            self.blocks.append({
                "ln1": (a[p + "ln1.g"].astype(np.int64), a[p + "ln1.b"].astype(np.int64)),
                "ln2": (a[p + "ln2.g"].astype(np.int64), a[p + "ln2.b"].astype(np.int64)),
                "qkv": Linear(a, p + "qkv", backend),
                "proj": Linear(a, p + "proj", backend),
                "fc": Linear(a, p + "fc", backend),
                "out": Linear(a, p + "out", backend),
                "att_m": int(h["att_m"][i]), "att_div": 1 << int(h["att_s"][i]),
            })
        self.eps = h["ln_eps"]

    # --- pieces -------------------------------------------------------------

    def layernorm(self, x, gb):
        g, b = gb
        d = x.shape[-1]
        mean = round_div(x.sum(axis=-1, keepdims=True), d)
        xc = x - mean
        var = (xc * xc).sum(axis=-1, keepdims=True) // d + self.eps
        std = isqrt(var)
        n = round_div(xc * (1 << F_LN), std)
        return rshift(n * g, F_G) + b

    def gelu_q10(self, x):
        x = np.clip(x, -FC_LIM, FC_LIM)
        idx = np.clip(x + GELU_HALF, 0, 2 * GELU_HALF - 1)
        return np.where(x >= GELU_HALF, x, np.where(x < -GELU_HALF, 0, self.gelu[idx]))

    def attention(self, blk, qkv, cache, t0):
        """qkv: (B, t, 3d) for new positions t0..t0+t-1; cache: k, v of positions < t0 (per window)."""
        b, t = qkv.shape[:2]
        H, dh, d = self.heads, self.dh, self.d
        qkv = np.clip(qkv, -QKV_LIM, QKV_LIM)
        q, k, v = (qkv[..., i * d:(i + 1) * d].reshape(b, t, H, dh).transpose(0, 2, 1, 3) for i in range(3))
        fl = self.backend != "int"
        if "k" not in cache:  # preallocated per window; K stored transposed (B, H, dh, CTX), V as (B, H, CTX, dh)
            dt = np.float64 if fl else np.int64  # f64 for the float backends
            cache["k"] = np.zeros((b, H, dh, CTX), dtype=dt)
            cache["v"] = np.zeros((b, H, CTX, dh), dtype=dt)
        n = t0 + t
        cache["k"][..., t0:n] = k.transpose(0, 1, 3, 2)
        cache["v"][:, :, t0:n] = v
        kc, vc = cache["k"][..., :n], cache["v"][:, :, :n]  # strided views BLAS can use without copying
        if fl:  # all terms < 2^46: exact in f64
            raw = (q.astype(np.float64) @ kc).astype(np.int64)
        else:
            raw = q @ kc  # (B, H, t, n)
        s = (raw * blk["att_m"] + (blk["att_div"] >> 1)) // blk["att_div"]
        valid = np.arange(n)[None, :] <= (t0 + np.arange(t))[:, None]  # causal
        s = np.where(valid, s, np.iinfo(np.int64).min // 4)
        mx = s.max(axis=-1, keepdims=True)
        idx = np.minimum(mx - s, EXP_LEN)
        e = np.where(valid, self.exp16[idx], 0)  # Q16; the max key has e = 2^16
        if fl:  # terms < 2^45
            acc = (e.astype(np.float64) @ vc).astype(np.int64)
        else:
            acc = e @ vc  # (B, H, t, dh)
        y = round_div(acc, e.sum(axis=-1, keepdims=True))
        return y.transpose(0, 2, 1, 3).reshape(b, t, d)

    # --- forward ------------------------------------------------------------

    def forward(self, tokens, t0, caches):
        """Integer logits (Q10) for tokens at positions t0..t0+t-1: (t,) -> (t, 257), (B, t) -> (B, t, 257).

        caches: list (one dict per layer), extended in place; a batch of B independent sequences
        shares t0. Every operation is per row (and attention per sequence), so a block forward,
        a token-by-token forward and any batching give identical integers.
        """
        tokens = np.asarray(tokens, dtype=np.int64)
        single = tokens.ndim == 1
        if single:
            tokens = tokens[None]
        t = tokens.shape[1]
        x = self.emb_res[tokens] + self.pos_res[t0:t0 + t]
        for blk, cache in zip(self.blocks, caches):
            x = np.clip(x, -RES_LIM, RES_LIM)
            a = self.attention(blk, blk["qkv"](self.layernorm(x, blk["ln1"])), cache, t0)
            x = np.clip(x + blk["proj"](a), -RES_LIM, RES_LIM)
            x = x + blk["out"](self.gelu_q10(blk["fc"](self.layernorm(x, blk["ln2"]))))
        x = np.clip(x, -RES_LIM, RES_LIM)
        q, m = quant_rows(self.layernorm(x, self.lnf))
        acc = int8_matmul(q, self.head_wt, self.backend)
        logits = (acc * m * self.head_mul + (self.head_div >> 1)) // self.head_div
        return logits[0] if single else logits

    def selftest_digest(self):
        logits = self.forward(SELFTEST, 0, self.new_caches())
        return hashlib.sha256(logits.astype("<i8").tobytes()).hexdigest()

    def selftest(self):
        """Raise unless this backend reproduces the integer logits recorded at export time."""
        if self.selftest_digest() != self.header["selftest"]:
            raise RuntimeError(f"self-test failed for backend {self.backend}: inference is not bit-exact here")

    def new_caches(self):
        return [{} for _ in range(self.layers)]
