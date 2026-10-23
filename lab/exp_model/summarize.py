"""Print the E8a results table (markdown) from lab/exp_model/results/*.json.

Usage: python3 lab/exp_model/summarize.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORDER = ["base", "case", "case2", "wrap", "o3", "o4", "D", "X", "S", "sp2", "wrap+case2+sp2+S"]


def main():
    results = {p.stem: json.loads(p.read_text()) for p in (HERE / "results").glob("*.json")}
    names = [n for n in ORDER if n in results] + sorted(set(results) - set(ORDER))
    base = results.get("base")
    print("| variant | 1 KB | 3 KB | 10 KB | 30 KB | short docs vs base (mean of sizes) | full books bpc | books vs base | table on disk |")
    print("|---|---|---|---|---|---|---|---|---|")
    for n in names:
        r = results[n]
        short = r["short"]
        cells = [f"{short[s]['avg_bytes']:.0f} B ({short[s]['bpc']:.3f})" for s in ("1000", "3000", "10000", "30000")]
        rel = books_rel = ""
        if base:
            change = [short[s]["avg_bytes"] / base["short"][s]["avg_bytes"] - 1 for s in short]
            rel = f"{100 * sum(change) / len(change):+.1f}%"
            if "books_bpc" in r:
                books_rel = f"{100 * (r['books_bpc'] / base['books_bpc'] - 1):+.1f}%"
        books = f"{r['books_bpc']:.3f}" if "books_bpc" in r else "-"
        print(f"| {n} | " + " | ".join(cells) + f" | {rel} | {books} | {books_rel} | {r['table_mb']} MB |")


if __name__ == "__main__":
    main()
