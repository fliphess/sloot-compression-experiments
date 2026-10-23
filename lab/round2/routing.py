"""Trick 1: encoder-side table routing. Compress each document with every published table, keep the smallest.

The 4-byte table fingerprint is already in every file header, so routing costs zero extra bytes: the
decoder just loads the table the fingerprint names. Never worse than any single table, by construction.
Uses the domains.py test documents (books / wiki / code × 1/3/10 KB, 30 each) and the C codec in
adaptive mode (real bytes, round trips checked).

Usage: python3 lab/round2/routing.py
"""
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from domains import ROOT, domains, steno_sizes  # noqa: E402

TABLES = {"v1": "c_en-gutenberg-v1.bin", "3gb": "c_exp_table_3gb.bin", "mixed": "c_round2_mixed.bin"}
SETS = {"books-only tables": ["v1", "3gb"], "3gb + mixed": ["3gb", "mixed"], "all three": ["v1", "3gb", "mixed"]}


def main():
    docs = domains()
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for (dom, size), group in docs.items():
            sizes = {t: steno_sizes(group, ROOT / "tables" / f, "-a", Path(tmp)) for t, f in TABLES.items()}
            row = {t: sum(v) / len(v) for t, v in sizes.items()}
            for label, names in SETS.items():
                row[f"route {label}"] = sum(min(sizes[t][i] for t in names) for i in range(len(group))) / len(group)
            row["picked (all three)"] = {t: sum(1 for i in range(len(group)) if min(sizes, key=lambda n: sizes[n][i]) == t)
                                         for t in TABLES}
            out[f"{dom} {size}"] = row
            print(f"{dom:<6}{size:>6}", "  ".join(f"{k} {v:.0f}" for k, v in row.items() if not isinstance(v, dict)),
                  row["picked (all three)"], flush=True)
    (HERE / "results" / "routing.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
