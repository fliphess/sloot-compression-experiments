"""Fill the results table in lab/exp_tinymodel/RESULTS.md from results/*.json.

Usage: python3 lab/exp_tinymodel/summarize.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
START, END = "<!-- results:start -->", "<!-- results:end -->"
SIZES = ["1000", "3000", "10000", "30000"]
BASELINES = [  # name, avg bytes per short-doc size, full-books bpc (real bytes, from RESEARCH.md)
    ("v1 static n-gram (E7)", [285, 837, 2756, 8207], 2.420),
    ("E8a static combo", [270, None, None, None], 2.289),
    ("E8b adaptive n-gram", [276, 793, 2552, 7484], 1.911),
]


def table():
    rows = ["| model | params | int8 size | epochs | train min | 1 KB | 3 KB | 10 KB | 30 KB | books bpc | int8 books bpc | MFLOP/byte |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, sizes, books in BASELINES:
        cells = " | ".join(f"{s:,} B" if s else "-" for s in sizes)
        rows.append(f"| {name} | - | 31 MB table | - | - | {cells} | {books:.3f} | - | - |")
    for name in ("1m", "5m", "20m"):
        path = HERE / "results" / f"{name}.json"
        if not path.exists():
            rows.append(f"| byte transformer {name} | pending | | | | | | | | | | |")
            continue
        r = json.loads(path.read_text())
        f, q = r["fp32"], r["int8"]
        cells = " | ".join(f"{f[s]['avg_bytes']:,.0f} B ({f[s]['bpc']:.3f})" for s in SIZES)
        rows.append(f"| byte transformer {name} | {r['params'] / 1e6:.2f}M | {r['int8_mb']:.1f} MB | {r['epochs']:.2f} | "
                    f"{r['train_minutes']:.0f} | {cells} | **{f['books']['bpc']:.3f}** | {q['books']['bpc']:.3f} | "
                    f"{r['flops_per_byte'] / 1e6:.1f} |")
    rows.append("")
    rows.append("Short docs: average bytes per document = ceil(ideal bits / 8) + 7 B container, with bpc in brackets. "
                "Model rows are **ideal code lengths, not real files**; baselines are real files.")
    int8_rows = []
    for name in ("1m", "5m", "20m"):
        path = HERE / "results" / f"{name}.json"
        if path.exists():
            r = json.loads(path.read_text())
            deltas = ", ".join(f"{k}: {100 * (r['int8'][k]['bpc'] / r['fp32'][k]['bpc'] - 1):+.2f}%" for k in SIZES + ["books"])
            int8_rows.append(f"- {name}: fake int8 vs fp32: {deltas}")
    return "\n".join(rows + [""] + int8_rows)


def main():
    path = HERE / "RESULTS.md"
    text = path.read_text()
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    path.write_text(f"{head}{START}\n{table()}\n{END}{tail}")
    print(table())


if __name__ == "__main__":
    main()
