"""E11: Python reference outputs for the C port.

  python3 lab/c/ref.py inputs                  write the test inputs to lab/c/data/inputs/ + manifest.json
  python3 lab/c/ref.py encode TABLE [--procs N]
        TABLE = v1 | 3gb. Encodes every input with the Python reference, static (mode 1) and adaptive
        d3-sum-w32 (mode 2), into lab/c/data/py/<TABLE>_<mode>/<name>.stn; timings in .../times.jsonl.

Reference functions: static = codec.compress (snippets + edge cases) and exp_table/evaluate.compress_fast
(full books; asserted identical to codec.compress in E8c, it only speeds up order 0);
adaptive = exp_adaptive/adaptive.compress with run.make_configs()["d3-sum-w32"].
The 3 GB table is loaded as a LeanTable (uint32 arrays, same coding; fingerprint computed in slices).
"""
import hashlib
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
ROOT = LAB.parent
sys.path[:0] = [str(LAB), str(LAB / "exp_adaptive"), str(LAB / "exp_table")]

import adaptive  # noqa: E402
import codec  # noqa: E402
import evaluate  # noqa: E402
import run as adaptive_run  # noqa: E402
from short_docs import SIZES, snippets  # noqa: E402
from steno import split_books  # noqa: E402

DATA = HERE / "data"
TABLES = {"v1": ROOT / "tables" / "en-gutenberg-v1.npz", "3gb": ROOT / "tables" / "exp_table_3gb.npz"}


def edge_cases():
    return {"empty": b"", "one_char": b"a", "unicode": "Zoë’s café — naïve «quote» 日本\r\n".encode(),
            "binary": bytes(range(256)) * 4, "random": np.random.default_rng(1).bytes(3000),
            "unknown_words": b"Xqzzt flurbograph vorpalisms zzyzx."}


def cmd_inputs():
    out = DATA / "inputs"
    out.mkdir(parents=True, exist_ok=True)
    items = []
    for name, data in edge_cases().items():
        (out / f"edge_{name}.bin").write_bytes(data)
        items.append({"name": f"edge_{name}", "kind": "edge", "path": str(out / f"edge_{name}.bin")})
    _, test = split_books()
    for path in test:
        text = path.read_text(encoding="utf-8")
        for size in SIZES:
            for i, snip in enumerate(snippets(text, size)):
                name = f"snip_{size}_{path.stem}_{i}"
                (out / f"{name}.txt").write_bytes(snip.encode())
                items.append({"name": name, "kind": f"snip{size}", "path": str(out / f"{name}.txt")})
    for path in test:
        items.append({"name": f"book_{path.stem}", "kind": "book", "path": str(path)})
    (DATA / "manifest.json").write_text(json.dumps(items, indent=1))
    print(f"{len(items)} inputs")


class LeanTable2(evaluate.LeanTable):
    """LeanTable with the fingerprint computed in slices (same value, bounded memory)."""

    def __init__(self, arrays):
        import codec as c
        self.arrays = arrays
        blob, lens = arrays["vocab_blob"].tobytes(), arrays["vocab_lens"]
        offsets = [0] + np.cumsum(lens, dtype=np.int64).tolist()
        self.vocab = [blob[offsets[i]:offsets[i + 1]].decode() for i in range(len(lens))]
        self.vocab[c.UNK] = None
        self.index = {t: i for i, t in enumerate(self.vocab) if t is not None}
        self.b = len(self.vocab).bit_length()
        self.uni = arrays["uni"].astype(np.int64)
        self.uni_cum = np.cumsum(self.uni)
        self.total0 = int(self.uni_cum[-1])
        self.spell_cum = np.cumsum(arrays["spell"].astype(np.int64) + 1, axis=1)
        self.ctx = {k: arrays[f"c{k}_ctx"] for k in (1, 2)}
        self.start = {k: arrays[f"c{k}_start"] for k in (1, 2)}
        self.sym = {k: arrays[f"c{k}_sym"] for k in (1, 2)}
        self.cnt = {k: arrays[f"c{k}_cnt"] for k in (1, 2)}
        assert self.total0 < c.MAX_TOTAL
        digest = hashlib.sha256()
        for name in self.ARRAYS:
            digest.update(name.encode())
            flat = np.asarray(arrays[name]).reshape(-1)
            for i in range(0, len(flat), 16_000_000):
                digest.update(np.ascontiguousarray(flat[i:i + 16_000_000]).astype("<i8").tobytes())
        self.fingerprint = digest.digest()[:4]


_table = None
_cfg = adaptive_run.make_configs()["d3-sum-w32"]


def load(name):
    path = TABLES[name]
    if name == "v1":
        return codec.FrozenTable.load(path)
    with np.load(path) as data:
        return LeanTable2({k: data[k] for k in codec.FrozenTable.ARRAYS})


def job(args):
    table_name, mode, item = args
    data = Path(item["path"]).read_bytes()
    t0 = time.time()
    if mode == "adaptive":
        blob = adaptive.compress(_table, _cfg, data)
    elif item["kind"] == "book":
        blob = evaluate.compress_fast(_table, data)
    else:
        blob = codec.compress(_table, data)
    dt = time.time() - t0
    out = DATA / "py" / f"{table_name}_{mode}" / f"{item['name']}.stn"
    out.write_bytes(blob)
    return {"name": item["name"], "mode": mode, "raw": len(data), "out": len(blob), "enc_s": round(dt, 4)}


def cmd_encode(table_name, procs, modes):
    global _table
    t0 = time.time()
    _table = load(table_name)
    print(f"{table_name}: fingerprint {_table.fingerprint.hex()}, loaded in {time.time() - t0:.1f}s", flush=True)
    items = json.loads((DATA / "manifest.json").read_text())
    jobs = []
    for mode in modes:
        (DATA / "py" / f"{table_name}_{mode}").mkdir(parents=True, exist_ok=True)
        jobs += [(table_name, mode, it) for it in items]
    jobs.sort(key=lambda j: -Path(j[2]["path"]).stat().st_size)  # big books first
    log = DATA / "py" / f"{table_name}_times.jsonl"
    with get_context("fork").Pool(procs) as pool, open(log, "a") as f:
        for i, res in enumerate(pool.imap_unordered(job, jobs)):
            f.write(json.dumps(res) + "\n")
            f.flush()
            if res["name"].startswith("book") or i % 50 == 0:
                print(f"[{i + 1}/{len(jobs)}] {res}", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "inputs":
        cmd_inputs()
    elif sys.argv[1] == "encode":
        procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 1
        modes = sys.argv[sys.argv.index("--modes") + 1].split(",") if "--modes" in sys.argv else ["static", "adaptive"]
        cmd_encode(sys.argv[2], procs, modes)
    else:
        sys.exit(__doc__)
