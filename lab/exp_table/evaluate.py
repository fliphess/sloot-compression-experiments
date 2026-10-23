"""E8c: evaluate a codec-v1 table on the fixed test set with real bytes.

Short docs: the 200 snippets of lab/short_docs.py (1/3/10/30 KB), each
compressed and decompressed with the unchanged codec.decompress, compared byte
for byte. Full books: the 10 test books are encoded for real bytes; books up
to --verify-max bytes are also decoded and compared.

The encoder is codec v1's algorithm with a faster order-0 step (O(exclusions)
instead of O(vocab)); --check-identical also runs codec.compress and asserts
both produce the same bytes. LeanTable keeps the big arrays as uint32 instead
of upcasting them, to save memory; coding is unchanged.

Usage: python3 lab/exp_table/evaluate.py tables/exp_table_300mb.npz [--check-identical]
"""
import argparse
import hashlib
import json
import math
import resource
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "lab"))
import codec  # noqa: E402
from codec import STENO_V1, STORED, UNK, Encoder, FrozenTable, candidates, encode_spelling, varint  # noqa: E402
from short_docs import SIZES, snippets  # noqa: E402
from steno import TOKEN, split_books  # noqa: E402


class LeanTable(FrozenTable):
    def __init__(self, arrays):
        self.arrays = arrays
        blob, lens = arrays["vocab_blob"].tobytes(), arrays["vocab_lens"]
        offsets = [0] + np.cumsum(lens, dtype=np.int64).tolist()
        self.vocab = [blob[offsets[i]:offsets[i + 1]].decode() for i in range(len(lens))]
        self.vocab[UNK] = None
        self.index = {t: i for i, t in enumerate(self.vocab) if t is not None}
        self.b = len(self.vocab).bit_length()
        self.uni = arrays["uni"].astype(np.int64)
        self.uni_cum = np.cumsum(self.uni)
        self.total0 = int(self.uni_cum[-1])
        self.spell_cum = np.cumsum(arrays["spell"].astype(np.int64) + 1, axis=1)
        self.ctx = {k: arrays[f"c{k}_ctx"] for k in (1, 2)}
        self.start = {k: arrays[f"c{k}_start"] for k in (1, 2)}
        self.sym = {k: arrays[f"c{k}_sym"] for k in (1, 2)}
        self.cnt = {k: arrays[f"c{k}_cnt"] for k in (1, 2)}
        assert self.total0 < codec.MAX_TOTAL
        digest = hashlib.sha256()
        for name in self.ARRAYS:
            digest.update(name.encode())
            digest.update(np.ascontiguousarray(arrays[name]).astype("<i8").tobytes())
        self.fingerprint = digest.digest()[:4]

    def group(self, k, history):
        found = FrozenTable.group(self, k, history)
        if found is None:
            return None
        syms, cnts = found
        return syms.astype(np.int64), cnts.astype(np.int64)


def encode_token_fast(enc, table, w, history):
    for syms, cnts in candidates(table, history):
        if cnts is None:
            # identical to codec.order0_cum, computed from the exclusions only
            below = int(table.uni[syms[syms < w]].sum()) if syms.size else 0
            removed = int(table.uni[syms].sum()) if syms.size else 0
            low = (int(table.uni_cum[w - 1]) if w else 0) - below
            enc.encode(low, low + int(table.uni[w]), table.total0 - removed)
            return
        esc = int(syms.size)
        total = int(cnts.sum()) + esc
        i = int(np.searchsorted(syms, w))
        if i < syms.size and syms[i] == w:
            low = int(cnts[:i].sum())
            enc.encode(low, low + int(cnts[i]), total)
            return
        enc.encode(total - esc, total, total)


def compress_fast(table, data):
    """Same bytes as codec.compress, faster order-0 encoding."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return bytes([STORED]) + data
    tokens = TOKEN.findall(text)
    enc = Encoder()
    history = []
    for tok in tokens:
        w = table.index.get(tok, UNK)
        encode_token_fast(enc, table, w, history)
        if w == UNK:
            encode_spelling(enc, table, tok.encode())
        history = (history + [w])[-2:]
    out = bytes([STENO_V1]) + table.fingerprint + varint(len(tokens)) + enc.finish()
    if len(out) >= 1 + len(data):
        return bytes([STORED]) + data
    return out


_table = None
_args = None


def run_book(path):
    data = path.read_bytes()
    t0 = time.time()
    blob = compress_fast(_table, data)
    enc_s = time.time() - t0
    verified = False
    if len(data) <= _args.verify_max:
        assert codec.decompress(_table, blob) == data, path.name
        verified = True
    return path.stem, len(data), len(blob), verified, round(enc_s, 1)


def main():
    global _table, _args
    ap = argparse.ArgumentParser()
    ap.add_argument("table")
    ap.add_argument("--verify-max", type=int, default=500_000)
    ap.add_argument("--check-identical", action="store_true")
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--skip-books", action="store_true")
    _args = args = ap.parse_args()
    t0 = time.time()
    _table = table = LeanTable.load(ROOT / args.table)
    load_s = time.time() - t0
    print(f"{args.table}: {len(table.vocab):,} words, fingerprint {table.fingerprint.hex()}, load {load_s:.0f}s", flush=True)
    _, test = split_books()
    texts = [p.read_text(encoding="utf-8") for p in test]

    result = {"table": args.table, "fingerprint": table.fingerprint.hex(), "load_s": round(load_s), "short": {}, "books": []}
    for size in SIZES:
        t1 = time.time()
        n = real = raw = 0
        for text in texts:
            for snip in snippets(text, size):
                data = snip.encode()
                blob = compress_fast(table, data)
                if args.check_identical:
                    assert blob == codec.compress(table, data), "fast encoder differs from codec.compress"
                assert codec.decompress(table, blob) == data, "round trip failed"
                n += 1
                real += len(blob)
                raw += len(data)
        result["short"][size] = {"docs": n, "avg_bytes": round(real / n, 1), "bpc": round(8 * real / raw, 4),
                                 "secs_per_doc": round((time.time() - t1) / n, 2)}
        print(f"  {size:>6}: {real / n:8.1f} B  {8 * real / raw:.4f} bpc  (all {n} round trips OK)", flush=True)

    if not args.skip_books:
        with get_context("fork").Pool(args.procs) as pool:
            for stem, raw_len, real_len, verified, enc_s in pool.imap(run_book, test):
                result["books"].append({"book": stem, "bytes": raw_len, "real": real_len, "verified": verified, "enc_s": enc_s})
                print(f"  book {stem:>6}: {raw_len:>10,} -> {real_len:>9,}  {8 * real_len / raw_len:.4f} bpc"
                      f"  verified={verified}  {enc_s}s", flush=True)
        tot_raw = sum(b["bytes"] for b in result["books"])
        tot_real = sum(b["real"] for b in result["books"])
        result["books_total_bpc"] = round(8 * tot_real / tot_raw, 4)
        print(f"  books total: {8 * tot_real / tot_raw:.4f} bpc", flush=True)
    result["peak_rss_gb_main"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
    result["total_s"] = round(time.time() - t0)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / (Path(args.table).stem + ".eval.json")).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
