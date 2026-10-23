"""E12: export an fp32 E10 checkpoint to an integer-only model file (build time, floats allowed).

Usage: .venv/bin/python lab/round2/intmodel/export.py CKPT NAME
  e.g. .venv/bin/python lab/round2/intmodel/export.py tables/exp_tinymodel_20m.pt 20m
writes tables/intmodel_NAME.bin (the artifact) and tables/intmodel_NAME.build.json (calibration
stats, fingerprint). Architecture (d, layers) is read from the state dict; heads from E10's CONFIGS.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT / "lab" / "exp_tinymodel"), str(ROOT / "lab")]

import intlm  # noqa: E402
from intlm import EXP_LEN, F_G, F_LN, F_S, GELU_HALF, MAGIC, QKV_LIM, RES_LIM  # noqa: E402

CALIB_WINDOWS = 64
SEED = 0


def mult_shift(r):
    """Integer M in [2^14, 2^15] and shift S with M / 2^S ~= r (r >= 0, array)."""
    r = np.asarray(r, dtype=np.float64)
    safe = np.where(r > 0, r, 1.0)
    s = 14 - np.floor(np.log2(safe)).astype(np.int64)
    m = np.round(safe * np.exp2(s)).astype(np.int64)
    m = np.where(r > 0, m, 0)
    s = np.where(r > 0, s, 1)
    assert (s >= 1).all() and (s <= 62).all() and (m < 1 << 16).all(), (s.min(), s.max())
    return m.astype(np.int32), s.astype(np.int8)


def quant_w(w):
    """Symmetric int8 per output row, as E10's fake_quant(dim=1)."""
    scale = np.maximum(np.abs(w).max(axis=1), 1e-8) / 127
    q = np.clip(np.round(w / scale[:, None]), -127, 127).astype(np.int8)
    return q, scale


@torch.no_grad()
def calibrate(model, d, layers):
    """Max |activation| per site on CALIB_WINDOWS random 512-byte windows of the training stream."""
    from common import train_bytes
    data = train_bytes()
    rng = np.random.default_rng(SEED)
    starts = rng.integers(0, len(data) - intlm.CTX, CALIB_WINDOWS)
    x = torch.from_numpy(np.stack([data[s:s + intlm.CTX] for s in starts]).astype(np.int64))
    x[:, 0] = intlm.BOS  # documents start with BOS
    stats = {}

    def rec(key, t):
        stats[key] = max(stats.get(key, 0.0), float(t.abs().max()))

    def rec2(k1, t1, k2, t2):
        rec(k1, t1)
        rec(k2, t2)

    hooks = []
    for i, blk in enumerate(model.blocks):
        hooks.append(blk.ln1.register_forward_hook(lambda m, inp, out, i=i: rec2("res", inp[0], f"ln{i}a", out)))
        hooks.append(blk.ln2.register_forward_hook(lambda m, inp, out, i=i: rec2("res", inp[0], f"ln{i}b", out)))
        hooks.append(blk.qkv.register_forward_hook(lambda m, inp, out, i=i: rec(f"qkv{i}", out)))
        hooks.append(blk.fc.register_forward_hook(lambda m, inp, out, i=i: rec(f"fc{i}", out)))
    hooks.append(model.ln.register_forward_hook(lambda m, inp, out: rec2("res", inp[0], "lnf", out)))
    for start in range(0, CALIB_WINDOWS, 16):
        logits = model(x[start:start + 16])
        rec("logits", logits)
    for h in hooks:
        h.remove()
    return stats


def export(ckpt, name):
    from common import CONFIGS, ByteLM
    sd = torch.load(ckpt, map_location="cpu")
    vocab, d = sd["emb.weight"].shape
    layers = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
    heads = next(h for (dd, ll, h) in CONFIGS.values() if (dd, ll) == (d, layers))
    torch.set_num_threads(3)
    model = ByteLM(d, layers, heads)
    model.load_state_dict(sd)
    model.eval()
    stats = calibrate(model, d, layers)
    W = {k: v.numpy().astype(np.float64) for k, v in sd.items()}

    f_res = int(math.floor(math.log2(RES_LIM / (8 * stats["res"]))))
    f_qkv = [int(math.floor(math.log2(QKV_LIM / (4 * stats[f"qkv{i}"])))) for i in range(layers)]
    for i in range(layers):  # LN out Q12 must stay < 2^21 / 4 and fc out (Q10) < 2^20 / 4
        assert stats[f"ln{i}a"] * 4 < (1 << 21) >> F_LN and stats[f"ln{i}b"] * 4 < (1 << 21) >> F_LN
        assert stats[f"fc{i}"] * 4 < (1 << 20) >> F_S
    arrays = []

    def add(key, arr):
        arrays.append((key, np.ascontiguousarray(arr)))

    def linear(key, w, b, f_in, f_out):
        q, s = quant_w(w)
        m, sh = mult_shift(s * 2.0 ** (f_out - f_in) / 127)
        add(key + ".w", q)
        add(key + ".m", m)
        add(key + ".s", sh)
        add(key + ".b", np.round(b * 2.0 ** f_out).astype(np.int32))

    def ln(key, g, b):
        add(key + ".g", np.round(g * 2.0 ** F_G).astype(np.int32))
        add(key + ".b", np.round(b * 2.0 ** F_LN).astype(np.int32))

    i = np.arange(EXP_LEN + 1)
    add("exp", np.round(2.0 ** 30 * np.exp(-i / 1024)).astype(np.int64))
    xs = (np.arange(2 * GELU_HALF) - GELU_HALF) / 1024
    add("gelu", np.round(1024 * xs * 0.5 * (1 + np.array([math.erf(v / math.sqrt(2)) for v in xs]))).astype(np.int32))
    eq, es = quant_w(W["emb.weight"])
    add("emb.w", eq)
    m, s = mult_shift(es * 2.0 ** f_res)
    add("emb.mr", m)
    add("emb.sr", s)
    m, s = mult_shift(es * 2.0 ** (F_S - F_LN) / 127)
    add("emb.mh", m)
    add("emb.sh", s)
    pq, ps = quant_w(W["pos.weight"])
    add("pos.w", pq)
    m, s = mult_shift(ps * 2.0 ** f_res)
    add("pos.mr", m)
    add("pos.sr", s)
    att_m, att_s = [], []
    dh = d // heads
    for l in range(layers):
        p, k = f"blocks.{l}.", f"b{l}."
        ln(k + "ln1", W[p + "ln1.weight"], W[p + "ln1.bias"])
        linear(k + "qkv", W[p + "qkv.weight"], W[p + "qkv.bias"], F_LN, f_qkv[l])
        linear(k + "proj", W[p + "proj.weight"], W[p + "proj.bias"], f_qkv[l], f_res)
        ln(k + "ln2", W[p + "ln2.weight"], W[p + "ln2.bias"])
        linear(k + "fc", W[p + "fc.weight"], W[p + "fc.bias"], F_LN, F_S)
        linear(k + "out", W[p + "out.weight"], W[p + "out.bias"], F_S, f_res)
        m, s = mult_shift(np.array([2.0 ** (F_S - 2 * f_qkv[l]) / math.sqrt(dh)]))
        att_m.append(int(m[0]))
        att_s.append(int(s[0]))
    ln("lnf", W["ln.weight"], W["ln.bias"])

    header = {
        "format": "JSINTLM1", "vocab": int(vocab), "ctx": intlm.CTX, "stride": intlm.STRIDE, "d": int(d),
        "layers": int(layers), "heads": int(heads), "f_res": f_res, "f_qkv": f_qkv, "att_m": att_m, "att_s": att_s,
        "ln_eps": max(1, int(round(1e-5 * 2.0 ** (2 * f_res)))),
        "source": Path(ckpt).name, "source_sha256": hashlib.sha256(Path(ckpt).read_bytes()).hexdigest(),
        "arrays": [[k, a.dtype.newbyteorder("<").str, list(a.shape)] for k, a in arrays],
    }
    assert d <= 2048 and 4 * d <= 2048 and d // heads <= 64  # the overflow bounds in intlm.py assume these
    for k, a in arrays:
        if k.endswith((".g", ".b")) and a.dtype == np.int32:
            assert np.abs(a.astype(np.int64)).max() < 1 << 24, k

    def pack(hdr):
        hb = json.dumps(hdr, sort_keys=True, separators=(",", ":")).encode()
        return MAGIC + len(hb).to_bytes(4, "little") + hb + b"".join(a.astype(a.dtype.newbyteorder("<")).tobytes() for _, a in arrays)

    # self-test vector: SHA-256 of the integer logits (int64 LE) for a fixed input, computed with the
    # pure-integer backend. IntLM.selftest() recomputes it with whatever backend is in use.
    tmp = ROOT / "tables" / f".intmodel_{name}.tmp"
    tmp.write_bytes(pack(header))
    header["selftest"] = intlm.IntLM(tmp, "int").selftest_digest()
    tmp.unlink()
    blob = pack(header)
    out = ROOT / "tables" / f"intmodel_{name}.bin"
    out.write_bytes(blob)
    digest = hashlib.sha256(blob).hexdigest()
    build = {"file": out.name, "bytes": len(blob), "sha256": digest, "fingerprint": digest[:8],
             "source": header["source"], "source_sha256": header["source_sha256"], "f_res": f_res,
             "f_qkv": f_qkv, "calibration_max_abs": stats, "calib_windows": CALIB_WINDOWS, "seed": SEED}
    (ROOT / "tables" / f"intmodel_{name}.build.json").write_text(json.dumps(build, indent=1) + "\n")
    print(json.dumps({k: build[k] for k in ("file", "bytes", "fingerprint", "f_res", "f_qkv")}))
    print("calib:", {k: round(v, 2) for k, v in stats.items() if not k.startswith("ln")})


if __name__ == "__main__":
    export(sys.argv[1], sys.argv[2])
