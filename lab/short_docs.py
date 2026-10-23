"""Short documents (1-30 KB, roughly one paper page): what does a reference table add?

Snippets are cut from the 10 unseen test books, 5 per book per size, each
starting at a paragraph. Every method codes each snippet on its own, as one
independent archive item. Sizes are real compressed bytes, headers included;
the steno sizes are ideal arithmetic coding plus 4 bytes for length and coder
flush.

Methods that use reference knowledge:
  brotli       built-in ~120 KB dictionary (web text, several languages)
  zstd+dict    zstd with a dictionary trained on the 90 training books
  steno        the frozen n-gram table from lab/steno.py

Usage: python3 lab/short_docs.py
"""
import math
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from steno import Table, compressed_size, encode_cost, split_books

SIZES = [1_000, 3_000, 10_000, 30_000]
PER_BOOK = 5
STENO_OVERHEAD = 4


def snippets(text, size):
    for i in range(PER_BOOK):
        pos = int(len(text) * (i + 0.5) / PER_BOOK)
        para = text.find("\n\n", pos)
        start = para + 2 if 0 <= para < len(text) - size else max(0, len(text) - size)
        yield text[start:start + size]


def train_dict(train_paths, size, out):
    subprocess.run(["zstd", "--train-fastcover", "-q", "-f", "-B16384", f"--maxdict={size}",
                    *map(str, train_paths), "-o", str(out)], check=True)


def main():
    train, test = split_books()
    train_texts = [p.read_text(encoding="utf-8") for p in train]
    table = Table(train_texts)
    tests = [p.read_text(encoding="utf-8") for p in test]

    with tempfile.TemporaryDirectory() as tmp:
        small, big = Path(tmp) / "dict110k", Path(tmp) / "dict4m"
        train_dict(train, 112_640, small)
        train_dict(train, 4_000_000, big)
        methods = {
            "gzip -9": ["gzip", "-9", "-c"],
            "xz raw": ["xz", "--format=raw", "--lzma2=preset=9e", "-c"],
            "zstd -19": ["zstd", "-19", "-c", "-q"],
            "brotli": ["brotli", "-q", "11", "-c"],
            "zstd+110K": ["zstd", "-19", "-D", str(small), "-c", "-q"],
            "zstd+4M": ["zstd", "-19", "-D", str(big), "-c", "-q"],
        }
        cols = list(methods) + ["steno fix", "steno ada"]
        print(f"{'size':>7}{'docs':>6}" + "".join(f"{c:>11}" for c in cols) + f"{'ada wins':>10}")
        for size in SIZES:
            totals, wins, docs, raw = defaultdict(int), 0, 0, 0
            for text in tests:
                for snip in snippets(text, size):
                    data = snip.encode()
                    sizes = {name: compressed_size(cmd, data) for name, cmd in methods.items()}
                    for name, adaptive in (("steno fix", False), ("steno ada", True)):
                        bits, _, _ = encode_cost(table, snip, adaptive)
                        sizes[name] = math.ceil(bits[-1] / 8) + STENO_OVERHEAD
                    wins += sizes["steno ada"] < min(sizes[m] for m in methods)
                    for name, n in sizes.items():
                        totals[name] += n
                    docs += 1
                    raw += len(data)
            print(f"{size:>7,}{docs:>6}" + "".join(f"{8 * totals[c] / raw:>7.3f} bpc" for c in cols)
                  + f"{wins:>6}/{docs}", flush=True)
            print(f"{'':>13}" + "".join(f"{totals[c] / docs:>9.0f} B" for c in cols))
    print("\nfirst line: bits per input byte; second line: average compressed size per document")
    print("ada wins: documents where adaptive steno beats every other method")


if __name__ == "__main__":
    main()
