"""Trick 3: a code-aware tokenizer. Does it help the tables on code (and does it hurt prose)?

v1 tokenizer:  " ?[A-Za-z]+|[^A-Za-z]"   letters only; `self._parse_header2` is ~6 tokens, 8 spaces are 8 tokens
new tokenizer: " ?[A-Za-z_][A-Za-z0-9_]*|[0-9]+| {2,}|[\\s\\S]"   identifiers, digit runs, indentation runs
Both are lossless (every character is matched; checked below). The mixed table (same sources as
mixed_table.py) is rebuilt with the new tokenizer, and the Python adaptive codec (d3-sum-w32, bit-identical
to the C codec) is evaluated on the domains.py documents, static and adaptive, with every round trip checked.
The old-tokenizer numbers for the same documents come from results/domains_mixed.json (C codec).

Usage: python3 lab/round2/code_tokens.py
"""
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
sys.path[:0] = [str(HERE), str(LAB), str(LAB / "exp_adaptive")]

import adaptive  # noqa: E402
import codec  # noqa: E402
import run as e8b  # noqa: E402
from domains import domains  # noqa: E402
from mixed_table import MB, ROOT, code_text  # noqa: E402
from steno import split_books  # noqa: E402

CODE_TOKEN = re.compile(r" ?[A-Za-z_][A-Za-z0-9_]*|[0-9]+| {2,}|[\s\S]")
OUT = ROOT / "tables" / "round2_mixed_codetok.npz"


def main():
    codec.TOKEN = adaptive.TOKEN = CODE_TOKEN  # both modules look the tokenizer up at call time
    docs = domains()
    for group in docs.values():
        for d in group:
            text = d.decode("utf-8")
            assert "".join(CODE_TOKEN.findall(text)) == text, "tokenizer not lossless"
    if OUT.exists():
        table = codec.FrozenTable.load(OUT)
    else:
        t0 = time.time()
        books = [p.read_text(encoding="utf-8") for p in split_books()[0]]
        wiki = (ROOT / "corpus" / "enwik8").read_bytes()[:85 * MB].decode("utf-8", errors="replace")
        table = codec.FrozenTable.build(books + [wiki, code_text()])
        table.save(OUT)
        print(f"built {OUT.name}: {len(table.vocab):,} words, {OUT.stat().st_size / 1e6:.1f} MB, "
              f"fingerprint {table.fingerprint.hex()}, {time.time() - t0:.0f}s", flush=True)
    cfg = e8b.make_configs()["d3-sum-w32"]
    old = json.loads((HERE / "results" / "domains_mixed.json").read_text())
    out = {}
    for (dom, size), group in docs.items():
        static = sum(len(codec.compress(table, d)) for d in group) / len(group)
        sizes = []
        for d in group:
            blob = adaptive.compress(table, cfg, d)
            assert adaptive.decompress(table, cfg, blob) == d, "round trip failed"
            sizes.append(len(blob))
        key = f"{dom} {size}"
        out[key] = {"old static": old[key]["mixed static"], "new static": static,
                    "old adaptive": old[key]["mixed adaptive"], "new adaptive": sum(sizes) / len(sizes)}
        print(key, {k: round(v) for k, v in out[key].items()}, flush=True)
    (HERE / "results" / "code_tokens.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
