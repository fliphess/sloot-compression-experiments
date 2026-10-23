"""Fill the mixed-domain model results into WRITEUP.md and RESEARCH.md (between the mixed-results markers).

Reads lab/round2/results/domains_mixed_model.json (domain test of the books/wiki/code model), domains.json (the books-only
20m_big model and gzip) and domains_mixed.json (the mixed table). Without the mixed-model results it writes placeholders,
so it's safe to run any time.

Usage: python3 lab/round2/fill_mixed_results.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RES = HERE / "results"
START, END = "<!-- mixed-results:start -->", "<!-- mixed-results:end -->"


def load(name):
    path = RES / name
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    base, tables, mixed = load("domains.json"), load("domains_mixed.json"), load("domains_mixed_model.json")
    done = bool(mixed)
    rows = ["| ideal bytes per doc, 1 / 3 / 10 KB | gzip (real) | mixed table + adaptive (real) | books-only 20M model | **mixed 20M model** |",
            "|---|---|---|---|---|"]
    for dom in ("books", "wiki", "code"):
        def cell(src, col):
            vals = [src.get(f"{dom} {s}", {}).get(col) for s in (1000, 3000, 10000)]
            return " / ".join(f"{v:,.0f}" if v is not None else "…" for v in vals)
        rows.append(f"| {dom} | {cell(base, 'gzip')} | {cell(tables, 'mixed adaptive')} | {cell(base, '20m_big (3 GB)')} "
                    f"| **{cell(mixed, 'mixed_model')}** |")
    note = ("Measured on the domains.py test set (30 docs per size, never in training). Model columns are ideal bits + 7 B container."
            if done else "*Pending: the run finishes ~2026-10-24 18:30; this table fills itself in afterwards "
                         "(`lab/round2/fill_mixed_results.py`).*")
    block = f"{START}\n" + "\n".join(rows) + f"\n\n{note}\n{END}"
    for name in ("WRITEUP.md", "RESEARCH.md"):
        path = ROOT / name
        text = path.read_text()
        if START in text:
            head, rest = text.split(START, 1)
            text = head + block + rest.split(END, 1)[1]
            path.write_text(text)
            print(f"updated {name} ({'results' if done else 'placeholders'})")


if __name__ == "__main__":
    main()
