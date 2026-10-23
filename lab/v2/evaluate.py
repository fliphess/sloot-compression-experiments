"""Evaluate the v2 router on the domains.py documents (books / wiki / code × 1/3/10 KB, 30 each): real bytes.

Per document every candidate is produced; the v2 file is the smallest. Table candidates use the C codec
(byte-identical to Python, E11) with v1 / 3 GB / mixed tables, static and adaptive; every C file is round-trip
checked. Neural candidates use the integer 20m_big model (f32 exact backend): method 4 (E12) and method 5
(+ match model); method 5 is decoded and checked for every document, method 4 when it's the winner.

Usage: .venv/bin/python lab/v2/evaluate.py run SHARD NSHARDS     (writes results/v2_SHARD.jsonl)
       .venv/bin/python lab/v2/evaluate.py summary
"""
import json
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "round2"), str(HERE.parent / "round2" / "intmodel")]

RESULTS = HERE / "results"
C_TABLES = {"v1": "c_en-gutenberg-v1.bin", "3gb": "c_exp_table_3gb.bin", "mixed": "c_round2_mixed.bin"}


def run(shard, nshards):
    import neuralmatch as nm
    import nncodec
    from codec import varint
    from domains import ROOT, domains, steno_sizes
    from intlm import IntLM

    model = IntLM(ROOT / "tables" / "intmodel_20m_big.bin", "f32")
    model.selftest()
    docs = [(f"{dom} {size}", d) for (dom, size), group in domains().items() for d in group]
    mine = [(i, k, d) for i, (k, d) in enumerate(docs) if i % nshards == shard]
    RESULTS.mkdir(exist_ok=True)
    out = open(RESULTS / f"v2_{shard}.jsonl", "a")
    done = {json.loads(line)["i"] for line in open(RESULTS / f"v2_{shard}.jsonl")} if (RESULTS / f"v2_{shard}.jsonl").exists() else set()
    with tempfile.TemporaryDirectory() as tmp:
        table_sizes = {}
        for tname, f in C_TABLES.items():
            for mode, flag in (("static", "-s"), ("adaptive", "-a")):
                sizes = steno_sizes([d for _, _, d in mine], ROOT / "tables" / f, flag, Path(tmp))  # C round trips checked
                for (i, _, _), n in zip(mine, sizes):
                    table_sizes.setdefault(i, {})[f"{tname} {mode}"] = n
    for i, key, d in mine:
        if i in done:
            continue
        sizes = dict(table_sizes[i])
        sizes["stored"] = 1 + len(d)
        plain = nncodec.compress(model, d)
        sizes["neural"] = len(plain)  # v2 method byte replaces nncodec's mode byte: same length
        code, _ = nm.encode_bytes(model, d)
        sizes["neural+match"] = 1 + 4 + len(varint(len(d))) + len(code)
        assert nm.decode_bytes(model, code, len(d)) == d, f"neural+match round trip failed on doc {i}"
        best = min(sizes, key=sizes.get)
        if best == "neural":
            assert nncodec.decompress(model, plain) == d, f"neural round trip failed on doc {i}"
        out.write(json.dumps({"i": i, "set": key, "raw": len(d), "sizes": sizes, "best": best}) + "\n")
        out.flush()
        print(i, key, best, sizes[best], flush=True)


def summary():
    rows = defaultdict(list)
    for path in sorted(RESULTS.glob("v2_*.jsonl")):
        for line in open(path):
            r = json.loads(line)
            rows[r["set"]].append(r)
    methods = ["3gb adaptive", "mixed adaptive", "neural", "neural+match"]
    print(f"{'set':<12}{'docs':>5}" + "".join(f"{m:>16}" for m in methods) + f"{'v2 (best)':>12}  chosen")
    for key in sorted(rows, key=lambda k: (k.split()[0], int(k.split()[1]))):
        rs = rows[key]
        avg = {m: sum(r["sizes"][m] for r in rs) / len(rs) for m in methods}
        best = sum(r["sizes"][r["best"]] for r in rs) / len(rs)
        chosen = Counter(r["best"] for r in rs).most_common()
        print(f"{key:<12}{len(rs):>5}" + "".join(f"{avg[m]:>14.0f} B" for m in methods) + f"{best:>10.0f} B  "
              + ", ".join(f"{m} {n}" for m, n in chosen))
    print("\nAverage real bytes per document; v2 = smallest candidate per document (all chosen files decode-verified).")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(int(sys.argv[2]), int(sys.argv[3]))
    else:
        summary()
