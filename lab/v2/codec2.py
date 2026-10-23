"""v2 container: the encoder tries every method with every available table/model and keeps the smallest.

Byte 0 is the method id. Each method has its own id, which fixes the E11 finding that mode byte 2 meant
three different things (table adaptive, the E9 chain variant, the neural codec):
  0  stored                         raw bytes follow (worst case: input + 1 byte)
  1  table, static (codec v1)       fingerprint(4) + varint(tokens) + code       (= codec.py mode 1 without its mode byte)
  2  table, adaptive d3-sum-w32     fingerprint(4) + varint(tokens) + code       (= exp_adaptive mode 2)
  3  reserved: E9 chain T0 variant (count_unk); not produced by this encoder
  4  neural, integer (E12)          fingerprint(4) + varint(bytes) + code        (= nncodec mode 2)
  5  neural, integer + match model  fingerprint(4) + varint(bytes) + code        (neuralmatch.py)
The fingerprint names the table or model, so routing between resources costs no extra bytes (tricks round, trick 1).
The decoder needs the named resource; files made with a resource it doesn't have are refused.

Usage (Python reference; slow for the neural methods):
  .venv/bin/python lab/v2/codec2.py compress IN OUT     tries v1 / 3 GB / mixed tables + the 20m_big model
  .venv/bin/python lab/v2/codec2.py decompress IN OUT
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
sys.path[:0] = [str(HERE), str(LAB / "round2" / "intmodel"), str(LAB), str(LAB / "exp_adaptive")]

import adaptive  # noqa: E402
import codec  # noqa: E402
import neuralmatch  # noqa: E402
import nncodec  # noqa: E402
import run as e8b  # noqa: E402
from codec import read_varint, varint  # noqa: E402
from intlm import IntLM  # noqa: E402

ROOT = LAB.parent
STORED, TABLE_STATIC, TABLE_ADAPTIVE, CHAIN_RESERVED, NEURAL, NEURAL_MATCH = range(6)
NAMES = {0: "stored", 1: "table static", 2: "table adaptive", 4: "neural", 5: "neural + match"}
DEFAULT_TABLES = ["en-gutenberg-v1.npz", "exp_table_3gb.npz", "round2_mixed.npz"]
DEFAULT_MODELS = ["intmodel_20m_big.bin"]
CFG = e8b.make_configs()["d3-sum-w32"]


class Registry:
    """The published resources this coder knows, by fingerprint."""

    def __init__(self, tables=DEFAULT_TABLES, models=DEFAULT_MODELS, backend="f32"):
        self.tables = {}
        for name in tables:
            t = codec.FrozenTable.load(ROOT / "tables" / name)
            self.tables[t.fingerprint] = t
        self.models = {}
        for name in models:
            m = IntLM(ROOT / "tables" / name, backend)
            m.selftest()
            self.models[m.fingerprint] = m


def candidates(reg, data):
    """Yield (method, resource name, v2 file) for every method × resource."""
    yield STORED, "-", bytes([STORED]) + data
    for t in reg.tables.values():
        for method, inner in ((TABLE_STATIC, codec.compress(t, data)), (TABLE_ADAPTIVE, adaptive.compress(t, CFG, data))):
            if inner[0] != codec.STORED:
                yield method, t.fingerprint.hex(), bytes([method]) + inner[1:]
    for m in reg.models.values():
        if data:
            inner = nncodec.compress(m, data)
            if inner[0] != nncodec.STORED:
                yield NEURAL, m.fingerprint.hex(), bytes([NEURAL]) + inner[1:]
            code, _ = neuralmatch.encode_bytes(m, data)
            yield NEURAL_MATCH, m.fingerprint.hex(), bytes([NEURAL_MATCH]) + m.fingerprint + varint(len(data)) + code


def compress(reg, data):
    return min(candidates(reg, data), key=lambda c: len(c[2]))[2]


def decompress(reg, blob):
    method, body = blob[0], blob[1:]
    if method == STORED:
        return body
    fp = body[:4]
    if method in (TABLE_STATIC, TABLE_ADAPTIVE):
        if fp not in reg.tables:
            raise ValueError(f"file needs table {fp.hex()}, not available")
        t = reg.tables[fp]
        if method == TABLE_STATIC:
            return codec.decompress(t, bytes([codec.STENO_V1]) + body)
        return adaptive.decompress(t, CFG, bytes([adaptive.ADAPTIVE]) + body)
    if method in (NEURAL, NEURAL_MATCH):
        if fp not in reg.models:
            raise ValueError(f"file needs model {fp.hex()}, not available")
        m = reg.models[fp]
        if method == NEURAL:
            return nncodec.decompress(m, bytes([nncodec.NEURAL_V1]) + body)
        n, pos = read_varint(body, 4)
        return neuralmatch.decode_bytes(m, body[pos:], n)
    raise ValueError(f"unknown method {method}")


def main():
    cmd, src, dst = sys.argv[1:4]
    reg = Registry()
    data = Path(src).read_bytes()
    Path(dst).write_bytes(compress(reg, data) if cmd == "compress" else decompress(reg, data))


if __name__ == "__main__":
    main()
