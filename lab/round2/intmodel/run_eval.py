"""E12: real files for the 200 short test snippets: compress, decompress, verify, time.

Usage: .venv/bin/python lab/round2/intmodel/run_eval.py NAME BACKEND SHARD NSHARDS [--encode-only] [--sizes 1000,3000] [--limit K]
writes lab/round2/intmodel/results/NAME_BACKEND_sSHARDofNSHARDS.jsonl (one record per document, appended
as it goes; documents already in the file are skipped, so a killed run resumes).
Documents are numbered over sizes 1000, 3000, 10000, 30000 (50 each, in that order); shard k takes
indices i with i % NSHARDS == k.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT / "lab")]

import nncodec  # noqa: E402
from intlm import IntLM  # noqa: E402
from short_docs import SIZES, snippets  # noqa: E402
from steno import split_books  # noqa: E402


def short_docs():
    texts = [p.read_text(encoding="utf-8") for p in split_books()[1]]
    return [(size, j, s.encode()) for size in SIZES for j, s in enumerate(s for t in texts for s in snippets(t, size))]


def main():
    args = sys.argv[1:]
    encode_only = "--encode-only" in args
    args = [a for a in args if a != "--encode-only"]
    sizes, limit, batch = None, None, 1
    if "--batch" in args:
        i = args.index("--batch")
        batch = int(args[i + 1])
        del args[i:i + 2]
    if "--sizes" in args:
        i = args.index("--sizes")
        sizes = {int(v) for v in args[i + 1].split(",")}
        del args[i:i + 2]
    if "--limit" in args:
        i = args.index("--limit")
        limit = int(args[i + 1])
        del args[i:i + 2]
    name, backend, shard, nshards = args[0], args[1], int(args[2]), int(args[3])
    tag = os.environ.get("E12_TAG", "")
    out = HERE / "results" / f"{name}_{backend}{tag}_s{shard}of{nshards}.jsonl"
    out.parent.mkdir(exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["i"] for line in out.read_text().splitlines() if line.strip()}
    t = time.time()
    model = IntLM(ROOT / "tables" / f"intmodel_{name}.bin", backend)
    model.selftest()
    print(f"model {name} {model.fingerprint.hex()} backend {backend} loaded in {time.time() - t:.1f}s", flush=True)
    todo = [(i, d) for i, d in enumerate(short_docs()) if i % nshards == shard and (sizes is None or d[0] in sizes)]
    if limit:
        todo = todo[:limit]
    todo = [(i, d) for i, d in todo if i not in done]
    if batch > 1:
        return run_batched(model, todo, batch, out, encode_only)
    for i, (size, j, doc) in todo:
        stats = {}
        t0 = time.perf_counter()
        blob = nncodec.compress(model, doc, stats)
        t1 = time.perf_counter()
        rec = {"i": i, "size": size, "j": j, "bytes_in": len(doc), "bytes_out": len(blob),
               "ideal_bits": stats.get("ideal_bits"), "sha256": hashlib.sha256(blob).hexdigest(),
               "enc_s": t1 - t0, "threads": os.environ.get("VECLIB_MAXIMUM_THREADS", "default")}
        if not encode_only:
            back = nncodec.decompress(model, blob)
            rec["dec_s"] = time.perf_counter() - t1
            rec["ok"] = back == doc
            if not rec["ok"]:
                print(f"ROUND TRIP FAILED doc {i}", flush=True)
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in rec.items() if k != "sha256"}), flush=True)
    print("done", flush=True)


def run_batched(model, todo, batch, out, encode_only):
    """Documents of one size in groups of `batch`, coded with nncodec.compress_many/decompress_many.
    Per-document times are the group time shared in proportion to the bytes."""
    groups = []
    for size in SIZES:
        same = [x for x in todo if x[1][0] == size]
        groups += [same[k:k + batch] for k in range(0, len(same), batch)]
    for g in groups:
        docs = [d for _, (_, _, d) in g]
        t0 = time.perf_counter()
        blobs, ideals = nncodec.compress_many(model, docs)
        t1 = time.perf_counter()
        backs = None if encode_only else nncodec.decompress_many(model, blobs)
        t2 = time.perf_counter()
        total = sum(len(d) for d in docs)
        with out.open("a") as f:
            for (i, (size, j, doc)), blob, ideal, k in zip(g, blobs, ideals, range(len(g))):
                rec = {"i": i, "size": size, "j": j, "bytes_in": len(doc), "bytes_out": len(blob), "ideal_bits": ideal,
                       "sha256": hashlib.sha256(blob).hexdigest(), "enc_s": (t1 - t0) * len(doc) / total,
                       "threads": os.environ.get("VECLIB_MAXIMUM_THREADS", "default"), "batch": len(g)}
                if backs is not None:
                    rec["dec_s"] = (t2 - t1) * len(doc) / total
                    rec["ok"] = backs[k] == doc
                    if not rec["ok"]:
                        print(f"ROUND TRIP FAILED doc {i}", flush=True)
                f.write(json.dumps(rec) + "\n")
        print(f"group size {g[0][1][0]} x{len(g)}: enc {total / (t1 - t0):,.0f} B/s"
              + ("" if encode_only else f", dec {total / (t2 - t1):,.0f} B/s, ok {sum(b == d for b, d in zip(backs, docs))}/{len(g)}"), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
