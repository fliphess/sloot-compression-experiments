"""Overnight combination run: E8b's adaptive layer (d3-sum-w32) on E8c's larger tables.

Question: do the gains of a bigger table (E8c) and per-document learning (E8b) stack?
Per table: all 200 short snippets (encode + decode, every round trip verified) and the
10 full test books (encode for real bytes; decode-verified up to 500 KB). Results go to
lab/overnight/results/<table>.json; `summary` prints the comparison.

Second use, weight tuning: `sweep` runs several adaptive configs on one table, on the
test set or on a fresh validation set (10 books from corpus/gutenberg_big that are neither
in the 100 original books nor flagged in E8c's leakage list). The validation set is only
valid for tables that weren't trained on gutenberg_big (en-gutenberg-v1). Sweeps measure
short snippets plus the full books up to 500 KB.

Usage:
  python3 lab/overnight/combo.py run TABLE [TABLE ...]     e.g. en-gutenberg-v1 exp_table_1gb
  python3 lab/overnight/combo.py sweep TABLE test|val CONFIG [CONFIG ...]
  python3 lab/overnight/combo.py summary
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
sys.path[:0] = [str(LAB), str(LAB / "exp_adaptive")]

import adaptive  # noqa: E402
import run as adaptive_run  # noqa: E402
from codec import FrozenTable  # noqa: E402
from short_docs import SIZES, snippets  # noqa: E402
from steno import split_books  # noqa: E402

ROOT = LAB.parent
RESULTS = HERE / "results"
CONFIG = "d3-sum-w32"
VERIFY_MAX = 500_000


def validation_books():
    """10 fresh books: not among the original 100, not flagged as leakage, 100 KB - 1.5 MB, spread over the IDs."""
    original = {p.stem for p in (ROOT / "corpus" / "gutenberg").glob("*.txt")}
    flagged = set()
    for line in (LAB / "exp_table" / "leakage.tsv").read_text().splitlines()[1:]:
        cols = line.split("\t")
        if cols[6] == "1":
            flagged.add(cols[0])
    candidates = sorted((p for p in (ROOT / "corpus" / "gutenberg_big").glob("*.txt")
                         if p.stem not in original and p.stem not in flagged
                         and 100_000 <= p.stat().st_size <= 1_500_000), key=lambda p: int(p.stem))
    step = len(candidates) // 10
    return [candidates[i * step + step // 2] for i in range(10)]


def run_table(name, config=CONFIG, dataset="test", max_book=None, tag=None):
    cfg = adaptive_run.make_configs()[config]
    path = ROOT / "tables" / f"{name}.npz"
    table = FrozenTable.load(path)
    tag = tag or name
    res = {"table": name, "config": config, "set": dataset, "table_mb": round(path.stat().st_size / 1e6, 1),
           "vocab": len(table.vocab), "fingerprint": table.fingerprint.hex(), "short": {}, "books": {}}
    test = split_books()[1] if dataset == "test" else validation_books()
    res["book_ids"] = [p.stem for p in test]
    texts = [p.read_text(encoding="utf-8") for p in test]
    for size in SIZES:
        t0 = time.time()
        total = raw = 0
        for text in texts:
            for snip in snippets(text, size):
                data = snip.encode()
                blob = adaptive.compress(table, cfg, data)
                assert adaptive.decompress(table, cfg, blob) == data, f"{name} snippet {size}"
                total += len(blob)
                raw += len(data)
        res["short"][size] = {"avg_bytes": total / 50, "bpc": 8 * total / raw, "secs": round(time.time() - t0)}
        print(f"{tag} short {size}: {total / 50:.0f} B avg, verified", flush=True)
        save(tag, res)
    for path_book in test:
        data = path_book.read_bytes()
        if max_book and len(data) > max_book:
            continue
        t0 = time.time()
        blob = adaptive.compress(table, cfg, data)
        verified = len(data) <= VERIFY_MAX
        if verified:
            assert adaptive.decompress(table, cfg, blob) == data, f"{name} book {path_book.stem}"
        res["books"][path_book.stem] = {"bytes": len(data), "real": len(blob), "verified": verified,
                                        "secs": round(time.time() - t0)}
        print(f"{tag} book {path_book.stem}: {8 * len(blob) / len(data):.3f} bpc", flush=True)
        save(tag, res)
    res["done"] = True
    save(tag, res)


def save(name, res):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(res, indent=1))


def summary():
    print(f"{'table':<22}{'config':<13}{'set':<5}{'MB':>6}{'words':>9}" + "".join(f"{f'{s // 1000}K':>8}" for s in SIZES)
          + f"{'books bpc':>11}{'books done':>12}")
    for path in sorted(RESULTS.glob("*.json")):
        r = json.loads(path.read_text())
        books = r["books"].values()
        bpc = 8 * sum(b["real"] for b in books) / max(1, sum(b["bytes"] for b in books))
        row = f"{r['table']:<22}{r['config']:<13}{r.get('set', 'test'):<5}{r['table_mb']:>6}{r['vocab']:>9,}"
        row += "".join(f"{r['short'][str(s)]['avg_bytes']:>6.0f} B" if str(s) in r["short"] else f"{'-':>8}" for s in SIZES)
        print(row + f"{bpc:>11.3f}{len(r['books']):>9}" + ("" if r.get("done") else "  (running)"))
    print("\nshort = average real bytes per snippet (50 per size), all round trips verified; "
          "books = real bytes, verified up to 500 KB")
    print("reference: v1 static 285/837/2756/8207 B, books 2.420; E8b adaptive on v1 276/793/2552/7484 B, books 1.911")


if __name__ == "__main__":
    if sys.argv[1:2] == ["summary"]:
        summary()
    elif sys.argv[1:2] == ["run"]:
        for name in sys.argv[2:]:
            run_table(name)
    elif sys.argv[1:2] == ["sweep"]:
        name, dataset, *configs = sys.argv[2:]
        for config in configs:
            run_table(name, config, dataset, max_book=VERIFY_MAX, tag=f"{name}__{config}__{dataset}")
    else:
        sys.exit(__doc__)
