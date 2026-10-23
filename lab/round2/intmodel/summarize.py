"""E12: summarise results/*.jsonl against E10's fp32 / fake-int8 ideal numbers.

Usage: python3 lab/round2/intmodel/summarize.py [BACKEND=f32]
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFS = [HERE.parents[1] / "exp_tinymodel" / "results", HERE.parent / "results"]  # E10, round 2 (train_big.py)
SIZES = [1000, 3000, 10000, 30000]


def fp32_ref(name):
    """The torch evaluation JSON (fp32 + fake-int8 per-doc ideal bits) for this model, or None."""
    for d in REFS:
        if (d / f"{name}.json").exists():
            return json.loads((d / f"{name}.json").read_text())
    return None


def load(name, backend):
    recs = {}
    for p in sorted((HERE / "results").glob(f"{name}_{backend}_s*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                recs[r["i"]] = r
    return recs


def main():
    backend = sys.argv[1] if len(sys.argv) > 1 else "f32"
    print("| model | size | docs | verified | **real B (avg)** | int-model ideal + 7 B | fp32 ideal + 7 B (E10) | fake-int8 ideal + 7 B | real vs fp32 | 1 stream enc / dec B/s | batched enc / dec B/s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    names = sorted({p.name.split("_" + backend)[0] for p in (HERE / "results").glob(f"*_{backend}_s*.jsonl")})
    for name in sorted(names, key=lambda n: (len(n), n)):
        recs = load(name, backend)
        if not recs:
            continue
        e10 = fp32_ref(name)
        by = defaultdict(list)
        for r in recs.values():
            by[r["size"]].append(r)
        for size in SIZES:
            rs = sorted(by.get(size, []), key=lambda r: r["j"])
            if not rs:
                continue
            idx = [r["j"] for r in rs]
            ref = lambda key: [math.ceil(e10[key][str(size)]["per_doc_bits"][j] / 8) + 7 for j in idx] if e10 else [float("nan")]  # noqa: E731
            fp, fq = ref("fp32"), ref("int8")
            real = [r["bytes_out"] for r in rs]
            ideal = [math.ceil(r["ideal_bits"] / 8) + 7 for r in rs]
            ok = sum(1 for r in rs if r.get("ok"))
            dec = [r for r in rs if "dec_s" in r]
            avg = lambda v: sum(v) / len(v)  # noqa: E731

            def speed(sel):
                if not sel:
                    return "-"
                e = sum(r["bytes_in"] for r in sel) / sum(r["enc_s"] for r in sel)
                dd = [r for r in sel if "dec_s" in r]
                d = sum(r["bytes_in"] for r in dd) / sum(r["dec_s"] for r in dd) if dd else float("nan")
                return f"{e:,.0f} / {d:,.0f} ({len(sel)})"
            print(f"| {name} | {size // 1000} KB | {len(rs)} | {ok}/{len(dec)} | **{avg(real):,.1f}** | {avg(ideal):,.1f} | "
                  f"{avg(fp):,.1f} | {avg(fq):,.1f} | {100 * (avg(real) / avg(fp) - 1):+.2f}% | "
                  f"{speed([r for r in rs if r.get('batch', 1) == 1])} | {speed([r for r in rs if r.get('batch', 1) > 1])} |")


def det_report():
    """Every extra run (tagged files) vs the main f32 run with 1 BLAS thread: identical files?"""
    print()
    print("| run | docs | identical to main f32 run | round trips ok |")
    print("|---|---|---|---|")
    for p in sorted((HERE / "results").glob("*_det*.jsonl")):
        name = p.name.split("_det")[0].rsplit("_", 1)[0]
        main_recs = load(name, "f32")
        recs = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        both = [r for r in recs if r["i"] in main_recs]
        same = sum(1 for r in both if r["sha256"] == main_recs[r["i"]]["sha256"])
        dec = [r for r in recs if "ok" in r]
        print(f"| {p.stem} (threads {recs[0]['threads'] if recs else '-'}) | {len(recs)} | {same}/{len(both)} | "
              f"{sum(r['ok'] for r in dec)}/{len(dec)} |")


if __name__ == "__main__":
    main()
    det_report()
