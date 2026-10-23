"""E12: real files for (a subset of) the 10 full test books, vs E10's fp32 ideal bits for the same book.

Usage: .venv/bin/python lab/round2/intmodel/run_books.py NAME BACKEND BOOK[,BOOK...] [--encode-only]
appends one JSON record per book to results/books_NAME_BACKEND.jsonl
"""
import hashlib
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT / "lab")]

import nncodec  # noqa: E402
from intlm import IntLM  # noqa: E402
from steno import split_books  # noqa: E402


def main():
    args = [a for a in sys.argv[1:] if a != "--encode-only"]
    encode_only = len(args) < len(sys.argv) - 1
    name, backend, books = args[0], args[1], args[2].split(",")
    model = IntLM(ROOT / "tables" / f"intmodel_{name}.bin", backend)
    model.selftest()
    from summarize import fp32_ref
    e10 = fp32_ref(name)
    test = split_books()[1]
    for book in books:
        k = [p.stem for p in test].index(book)
        data = test[k].read_bytes()
        stats = {}
        t0 = time.perf_counter()
        blob = nncodec.compress(model, data, stats)
        t1 = time.perf_counter()
        fp_bits = e10["fp32"]["books"]["per_doc_bits"][k] if e10 else float("nan")
        rec = {"book": book, "bytes_in": len(data), "bytes_out": len(blob), "bpc": 8 * len(blob) / len(data),
               "fp32_ideal_bytes": math.ceil(fp_bits / 8) + 7 if e10 else None, "fp32_bpc": fp_bits / len(data),
               "ideal_bits": stats["ideal_bits"], "sha256": hashlib.sha256(blob).hexdigest(), "enc_s": t1 - t0}
        if not encode_only:
            rec["ok"] = nncodec.decompress(model, blob) == data
            rec["dec_s"] = time.perf_counter() - t1
        with (HERE / "results" / f"books_{name}_{backend}.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
