"""Codec v1 on the 10 full unseen test books: real compressed bytes.

Encodes every book; decodes and compares the ones up to --verify-max bytes
(decoding big books in Python is slow).

Usage: python3 lab/eval_books.py [--verify-max BYTES]
"""
import argparse
import time

from codec import FrozenTable, compress, decompress
from steno import split_books


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-max", type=int, default=500_000)
    args = ap.parse_args()
    table = FrozenTable.load()
    _, test = split_books()
    total_raw = total_real = 0
    print(f"{'book':<8}{'bytes':>11}{'real':>11}{'bpc':>8}{'verified':>10}{'secs':>7}")
    for path in test:
        data = path.read_bytes()
        t0 = time.time()
        blob = compress(table, data)
        verified = "-"
        if len(data) <= args.verify_max:
            assert decompress(table, blob) == data, path.name
            verified = "yes"
        total_raw += len(data)
        total_real += len(blob)
        print(f"{path.stem:<8}{len(data):>11,}{len(blob):>11,}{8 * len(blob) / len(data):>8.3f}"
              f"{verified:>10}{time.time() - t0:>7.0f}", flush=True)
    print(f"{'TOTAL':<8}{total_raw:>11,}{total_real:>11,}{8 * total_real / total_raw:>8.3f}")


if __name__ == "__main__":
    main()
