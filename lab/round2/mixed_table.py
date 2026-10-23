"""E13b: a mixed-domain table. Does broader coverage fix the domain gap found by domains.py?

Same codec v1 method (FrozenTable.build), different training mix:
  books  the 90 v1 training books (84.7 MB), identical to tables/en-gutenberg-v1.npz
  wiki   enwik8[0:85 MB] (the domains.py wiki test snippets come from enwik8[90 MB:])
  code   CPython 3.15.0 Lib/<subdir>/**/*.py, up to 40 MB (the code test snippets come from the
         top-level Lib/*.py files, which are excluded here)
Output: tables/round2_mixed.npz (+ the C version via lab/c/convert_table.py).

Usage: python3 lab/round2/mixed_table.py
"""
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent))

from codec import FrozenTable  # noqa: E402
from steno import split_books  # noqa: E402

MB = 1_000_000
OUT = ROOT / "tables" / "round2_mixed.npz"
CODE_LIMIT = 40 * MB


def code_text():
    tgz = ROOT / "corpus" / "downloads" / "Python-3.15.0.tgz"
    parts, total = [], 0
    with tarfile.open(tgz) as tar:  # read in memory, nothing extracted
        members = sorted((m for m in tar.getmembers()
                          if m.isfile() and m.name.endswith(".py") and "/Lib/" in m.name and m.name.count("/") >= 3),
                         key=lambda m: m.name.encode())
        for m in members:
            raw = tar.extractfile(m).read()
            parts.append(raw.decode("utf-8", errors="replace"))
            total += len(raw)
            if total >= CODE_LIMIT:
                break
    return "".join(parts)


def main():
    t0 = time.time()
    books = [p.read_text(encoding="utf-8") for p in split_books()[0]]
    wiki = (ROOT / "corpus" / "enwik8").read_bytes()[:85 * MB].decode("utf-8", errors="replace")
    code = code_text()
    print(f"books {sum(map(len, books)) / 1e6:.1f}M chars, wiki {len(wiki) / 1e6:.1f}M, code {len(code) / 1e6:.1f}M", flush=True)
    table = FrozenTable.build(books + [wiki, code])
    table.save(OUT)
    print(f"{OUT.name}: {len(table.vocab):,} words, {len(table.sym[1]):,} + {len(table.sym[2]):,} entries, "
          f"{OUT.stat().st_size / 1e6:.1f} MB, fingerprint {table.fingerprint.hex()}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
