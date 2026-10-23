"""E13: does it generalise beyond Gutenberg? Short documents from domains no table or model was trained on.

Domains (30 snippets per size, 1/3/10 KB, each compressed on its own, starting at a line boundary):
  wiki     raw Wikipedia XML dump with markup, from enwik8[90 MB:100 MB]
  code     Python source (corpus/samples/pysrc_1M.txt, CPython 3.15.0 Lib/*.py)
  books    the usual Gutenberg test snippets (first 30 per size), for reference

Methods:
  gzip -9, xz raw, zstd -19, brotli -q11           real bytes
  C codec (lab/c/steno), v1 and 3 GB table, static (-s) and adaptive (-a)   real bytes, round trips checked
  byte transformers (exp_tinymodel 20m, round2 20m_big), fp32 on CPU      ideal bits + 7 B container

Usage: .venv/bin/python lab/round2/domains.py                      all methods → results/domains.json
       .venv/bin/python lab/round2/domains.py --models-only NAME=CKPT ...   only these model checkpoints (tables/CKPT)
       .venv/bin/python lab/round2/domains.py --tables-only NAME=FILE ...   only the C codec with these tables
                                                   (FILE under tables/), e.g. mixed=c_round2_mixed.bin
"""
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
ROOT = LAB.parent
sys.path[:0] = [str(LAB / "exp_tinymodel"), str(LAB)]

SIZES = [1_000, 3_000, 10_000]
PER_SIZE = 30
STENO = LAB / "c" / "steno"
TABLES = {"v1": ROOT / "tables" / "c_en-gutenberg-v1.bin", "3gb": ROOT / "tables" / "c_exp_table_3gb.bin"}
MODELS = {"20m (85 MB)": "exp_tinymodel_20m.pt", "20m_big (3 GB)": "round2_20m_big.pt"}
TOOLS = {
    "gzip": ["gzip", "-9", "-c"],
    "xz": ["xz", "--format=raw", "--lzma2=preset=9e", "-c"],
    "zstd": ["zstd", "-19", "-c", "-q"],
    "brotli": ["brotli", "-q", "11", "-c"],
}


def cut(text, size, n):
    """n snippets of `size` bytes, evenly spaced, each starting at a line boundary."""
    out = []
    for i in range(n):
        pos = int(len(text) * (i + 0.5) / n)
        start = text.find(b"\n", pos) + 1
        start = start if 0 < start < len(text) - size else len(text) - size
        piece = text[start:start + size]
        while piece:  # don't split a UTF-8 character at the end
            try:
                piece.decode("utf-8")
                break
            except UnicodeDecodeError:
                piece = piece[:-1]
        out.append(piece)
    return out


def domains():
    from short_docs import snippets
    from steno import split_books

    enwik = (ROOT / "corpus" / "enwik8").read_bytes()[90_000_000:]
    code = (ROOT / "corpus" / "samples" / "pysrc_1M.txt").read_bytes()
    books = [p.read_text(encoding="utf-8") for p in split_books()[1]]
    res = {}
    for size in SIZES:
        res[("wiki", size)] = cut(enwik, size, PER_SIZE)
        res[("code", size)] = cut(code, size, PER_SIZE)
        res[("books", size)] = [s.encode() for text in books for s in snippets(text, size)][:PER_SIZE]
    return res


def steno_sizes(docs, table, mode, tmp):
    ins = [tmp / f"d{i}.txt" for i in range(len(docs))]
    outs = [tmp / f"d{i}.stn" for i in range(len(docs))]
    decs = [tmp / f"d{i}.dec" for i in range(len(docs))]
    for path, doc in zip(ins, docs):
        path.write_bytes(doc)
    subprocess.run([str(STENO), "compress", mode, str(table)] + [str(x) for pair in zip(ins, outs) for x in pair], check=True)
    subprocess.run([str(STENO), "decompress", str(table)] + [str(x) for pair in zip(outs, decs) for x in pair], check=True)
    for doc, dec in zip(docs, decs):
        assert dec.read_bytes() == doc, "C round trip failed"
    return [p.stat().st_size for p in outs]


def tables_only(specs):
    docs = domains()
    rows = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        for (dom, size), group in docs.items():
            key = f"{dom} {size}"
            rows[key] = {}
            for spec in specs:
                name, file = spec.split("=")
                for mode, flag in (("static", "-s"), ("adaptive", "-a")):
                    rows[key][f"{name} {mode}"] = sum(steno_sizes(group, ROOT / "tables" / file, flag, Path(tmpdir))) / len(group)
            print(key, {k: round(v) for k, v in rows[key].items()}, flush=True)
    names = "_".join(s.split("=")[0] for s in specs)
    (HERE / "results" / f"domains_{names}.json").write_text(json.dumps(rows, indent=1))


def models_only(specs):
    import torch
    from common import CKPT, CONTAINER, build, byte_bits

    torch.set_num_threads(4)
    docs = domains()
    rows = {f"{dom} {size}": {} for dom, size in docs}
    for spec in specs:
        name, ckpt = spec.split("=")
        model = build("20m")
        model.load_state_dict(torch.load(CKPT / ckpt, map_location="cpu"))
        model.eval()
        for (dom, size), group in docs.items():
            bits = [float(pb.sum()) for pb in byte_bits(model, group, "cpu")]
            rows[f"{dom} {size}"][name] = sum(math.ceil(b / 8) + CONTAINER for b in bits) / len(group)
            print(name, dom, size, round(rows[f"{dom} {size}"][name]), flush=True)
    names = "_".join(s.split("=")[0] for s in specs)
    (HERE / "results" / f"domains_{names}.json").write_text(json.dumps(rows, indent=1))


def main():
    if sys.argv[1:2] == ["--models-only"]:
        return models_only(sys.argv[2:])
    if sys.argv[1:2] == ["--tables-only"]:
        return tables_only(sys.argv[2:])
    import torch
    from common import CKPT, CONTAINER, build, byte_bits

    torch.set_num_threads(4)
    docs = domains()
    rows = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for (dom, size), group in docs.items():
            key = f"{dom} {size}"
            rows[key] = {"raw": sum(map(len, group)) / len(group)}
            for name, cmd in TOOLS.items():
                rows[key][name] = sum(len(subprocess.run(cmd, input=d, capture_output=True, check=True).stdout)
                                      for d in group) / len(group)
            for tname, table in TABLES.items():
                for mode, flag in (("static", "-s"), ("adaptive", "-a")):
                    rows[key][f"{tname} {mode}"] = sum(steno_sizes(group, table, flag, tmp)) / len(group)
            print(key, {k: round(v) for k, v in rows[key].items()}, flush=True)
    for mname, ckpt in MODELS.items():
        model = build("20m")
        model.load_state_dict(torch.load(CKPT / ckpt, map_location="cpu"))
        model.eval()
        for (dom, size), group in docs.items():
            bits = [float(pb.sum()) for pb in byte_bits(model, group, "cpu")]
            rows[f"{dom} {size}"][mname] = sum(math.ceil(b / 8) + CONTAINER for b in bits) / len(group)
            print(mname, dom, size, round(rows[f"{dom} {size}"][mname]), flush=True)
    (HERE / "results" / "domains.json").write_text(json.dumps(rows, indent=1))
    cols = list(next(iter(rows.values())))
    print("\n| set | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1))
    for key, row in rows.items():
        print(f"| {key} | " + " | ".join(f"{row[c]:,.0f}" for c in cols) + " |")
    print("\nAverage bytes per document (raw = input size). Tools and C codec: real bytes, C round trips checked. "
          "Models: ideal bits + 7 B container.")


if __name__ == "__main__":
    main()
