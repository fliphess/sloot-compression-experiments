"""E8c: collect build/eval results into markdown tables and fit the scaling curve.

Fit per metric: value(MB) = A + B * MB^-alpha (least squares over alpha on a
grid, A and B linear), using the unpruned tiers. A is the asymptote: what
this model (orders 2/1/0) would reach with unlimited training text.

Usage: python3 lab/exp_table/summarize.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SIZES = ["1000", "3000", "10000", "30000"]


def load():
    rows = []
    for build in sorted(RESULTS.glob("*.build.json")):
        name = build.name.removesuffix(".build.json")
        ev = RESULTS / f"{name}.eval.json"
        if not ev.exists():
            continue
        rows.append((json.loads(build.read_text()), json.loads(ev.read_text())))
    rows.sort(key=lambda r: (r[0]["prune"], r[0]["train_mb"]))
    return rows


def fit(mb, values):
    best = None
    for alpha in np.linspace(0.02, 1.5, 1480):
        x = np.asarray(mb, dtype=float) ** -alpha
        design = np.stack([np.ones_like(x), x], axis=1)
        coef, *_ = np.linalg.lstsq(design, np.asarray(values, dtype=float), rcond=None)
        err = float(((design @ coef - values) ** 2).sum())
        if best is None or err < best[0]:
            best = (err, alpha, coef)
    _, alpha, (a, b) = best
    return a, b, alpha


def main():
    rows = load()
    print("| table | train MB | books | tokens | vocab | o1 + o2 entries | prune | disk MB | build s | build peak GB "
          "| 1 KB | 3 KB | 10 KB | 30 KB | books bpc | eval peak GB |")
    print("|" + "---|" * 16)
    for build, ev in rows:
        short = ev["short"]
        cells = [f"{short[s]['avg_bytes']:.1f} B ({short[s]['bpc']:.3f})" for s in SIZES]
        print(f"| {Path(build['table']).stem} | {build['train_mb']:.0f} | {build['books']} | {build['tokens'] / 1e6:.0f}M "
              f"| {build['vocab']:,} | {build['entries_o1'] / 1e6:.1f}M + {build['entries_o2'] / 1e6:.1f}M "
              f"| {build['prune'] or '-'} | {build['disk_mb']:.0f} | {build['build_s']} | {build['peak_rss_gb_main']:.1f} | "
              + " | ".join(cells) + f" | {ev.get('books_total_bpc', float('nan')):.3f} | {ev['peak_rss_gb_main']:.1f} |")

    full = [(b, e) for b, e in rows if not b["prune"]]
    if len(full) >= 3:
        mb = [b["train_mb"] for b, _ in full]
        print("\nScaling fit, unpruned tiers: value = A + B * MB^-alpha")
        print("| metric | A (asymptote) | alpha | fit at 10 GB text | fit at 30 GB text | measured |")
        print("|---|---|---|---|---|---|")
        metrics = {f"{s} B doc (bytes)": [e["short"][s]["avg_bytes"] for _, e in full] for s in SIZES}
        metrics["books bpc"] = [e["books_total_bpc"] for _, e in full]
        for name, values in metrics.items():
            a, b, alpha = fit(mb, values)
            at = lambda x: a + b * x ** -alpha  # noqa: E731
            measured = ", ".join(f"{v:g}" for v in values)
            print(f"| {name} | {a:.3f} | {alpha:.2f} | {at(10_000):.3f} | {at(30_000):.3f} | {measured} |")
        disk = [b["disk_mb"] for b, _ in full]
        slope = np.polyfit(np.log(mb), np.log(disk), 1)[0]
        print(f"\nTable disk size grows as MB^{slope:.2f} (unpruned); "
              f"a 10 GB table ≈ {np.exp(np.log(10_000 / disk[-1]) / slope) * mb[-1] / 1000:.0f} GB of training text at this rate")


if __name__ == "__main__":
    main()
