"""Evaluate one E8a model variant on real bytes, with round-trip checks.

Short docs: the 200 snippets of lab/short_docs.py, all compressed + decompressed + compared.
Books: all 10 test books compressed; those <= --verify-max bytes also decompressed + compared.
Results go to lab/exp_model/results/VARIANT.json.

Usage (from project root):
  python3 lab/exp_model/run.py VARIANT [--no-books] [--verify-max BYTES] [--quick]
  VARIANT examples: base, case, wrap, o3, o4, D, X, case+wrap+o3+D
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from model import TABLES, Config, compress, decompress, table_for  # noqa: E402
from short_docs import SIZES, snippets  # noqa: E402
from steno import split_books  # noqa: E402

EDGE = {
    "empty": b"", "one char": b"a", "unicode": "Zoë’s café — naïve «quote» 日本\r\n".encode(),
    "binary": bytes(range(256)) * 4, "random": np.random.default_rng(1).bytes(3000),
    "unknown": b"Xqzzt flurbograph vorpalisms zzyzx.",
    "wrap+case": b"THE END\nOf the Story. McDonald said:\nyes, IT is\n\nNew paragraph here\nand there.\n",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant")
    ap.add_argument("--no-books", action="store_true")
    ap.add_argument("--verify-max", type=int, default=500_000)
    ap.add_argument("--quick", action="store_true", help="1 KB snippets only, no books")
    args = ap.parse_args()
    cfg = Config.parse(args.variant)
    t0 = time.time()
    table = table_for(cfg)
    path = TABLES / f"exp_model_{cfg.table_name}.npz"
    res = {"variant": args.variant, "config": repr(cfg), "table_mb": round(path.stat().st_size / 1e6, 1),
           "vocab": len(table.vocab), "fingerprint": table.fingerprint.hex(),
           "table_load_s": round(time.time() - t0, 1)}
    print(f"{args.variant}: table {res['table_mb']} MB, {res['vocab']:,} words", flush=True)

    for name, data in EDGE.items():
        assert decompress(table, compress(table, data)) == data, f"edge case {name}"

    _, test = split_books()
    texts = [p.read_text(encoding="utf-8") for p in test]
    res["short"] = {}
    for size in SIZES[:1] if args.quick else SIZES:
        t0 = time.time()
        docs = list(s for text in texts for s in snippets(text, size))
        if args.quick:
            docs = docs[:15]
        real = raw = 0
        for snip in docs:
            data = snip.encode()
            blob = compress(table, data)
            assert decompress(table, blob) == data, f"round trip failed at size {size}"
            real += len(blob)
            raw += len(data)
        res["short"][size] = {"docs": len(docs), "avg_bytes": round(real / len(docs), 1),
                              "bpc": round(8 * real / raw, 4), "secs": round(time.time() - t0)}
        print(f"  {size:>6,}: {real / len(docs):8.1f} B  {8 * real / raw:.3f} bpc  (all {len(docs)} round trips OK)", flush=True)

    if not (args.no_books or args.quick):
        res["books"] = {}
        total_raw = total_real = 0
        for p in test:
            data = p.read_bytes()
            t0 = time.time()
            blob = compress(table, data)
            verified = len(data) <= args.verify_max
            if verified:
                assert decompress(table, blob) == data, f"book {p.stem}"
            res["books"][p.stem] = {"bytes": len(data), "real": len(blob), "verified": verified}
            total_raw += len(data)
            total_real += len(blob)
            print(f"  book {p.stem:<6} {len(data):>10,} -> {len(blob):>9,}  {8 * len(blob) / len(data):.3f} bpc"
                  f"  {'verified' if verified else 'encoded'}  {time.time() - t0:.0f}s", flush=True)
        res["books_bpc"] = round(8 * total_real / total_raw, 4)
        print(f"  books total {8 * total_real / total_raw:.3f} bpc", flush=True)

    if not args.quick:
        out = HERE / "results" / f"{args.variant}.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
