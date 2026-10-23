"""E8b runner: measure adaptive configs on real bytes, with round-trip verification.

  short  the 200 snippets (lab/short_docs.py), every one compressed + decompressed
  books  the 10 full test books: all encoded, books <= --verify-max decoded and compared

Results are cached as JSON in lab/exp_adaptive/results/.

Usage:
  python3 lab/exp_adaptive/run.py short [CONFIG ...]
  python3 lab/exp_adaptive/run.py books CONFIG ...
  python3 lab/exp_adaptive/run.py verify CONFIG     decode + compare all 10 full books
  python3 lab/exp_adaptive/run.py perdoc CONFIG     per-snippet comparison with static
  python3 lab/exp_adaptive/run.py report
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import codec  # noqa: E402
from adaptive import Config, compress, decompress  # noqa: E402
from short_docs import SIZES, snippets  # noqa: E402
from steno import split_books  # noqa: E402

RESULTS = HERE / "results"
VERIFY_MAX = 500_000


def make_configs():
    cfgs = [Config("static", (("T", 2), ("T", 1)))]
    for w in (1, 2, 4, 8, 16, 32):
        cfgs.append(Config(f"sum-w{w}", (("M", 2, w), ("M", 1, w)), w0=w, new_words=True, ws=1))
    cfgs.append(Config("sum-w4-nonew", (("M", 2, 4), ("M", 1, 4)), w0=4, new_words=False, ws=1))
    for ws in (0, 4, 16):
        cfgs.append(Config(f"sum-w4-ws{ws}", (("M", 2, 4), ("M", 1, 4)), w0=4, new_words=True, ws=ws))
    for w in (1, 4, 16):
        cfgs.append(Config(f"cache-w{w}", (("D", 2, w), ("D", 1, w), ("T", 2), ("T", 1)), w0=w, new_words=True, ws=1))
        cfgs.append(Config(f"cache3-w{w}", (("D", 3, w), ("M", 2, w), ("M", 1, w)), w0=w, new_words=True, ws=1))
    for w in (64, 128):
        cfgs.append(Config(f"sum-w{w}", (("M", 2, w), ("M", 1, w)), w0=w, new_words=True, ws=1))
    for w in (8, 32, 64):
        cfgs.append(Config(f"d3-sum-w{w}", (("D", 3, 1), ("M", 2, w), ("M", 1, w)), w0=w, new_words=True, ws=1))
    cfgs.append(Config("d3d2-sum-w32", (("D", 3, 1), ("D", 2, 1), ("M", 2, 32), ("M", 1, 32)), w0=32, new_words=True, ws=1))
    return {c.name: c for c in cfgs}


CONFIGS = make_configs()
_table = None


def table():
    global _table
    if _table is None:
        _table = codec.FrozenTable.load()
    return _table


def run_short(name):
    cfg = CONFIGS[name]
    t = table()
    _, test = split_books()
    texts = [p.read_text(encoding="utf-8") for p in test]
    out = {"config": name, "sizes": {}}
    t0 = time.time()
    for size in SIZES:
        n = raw = real = 0
        for text in texts:
            for snip in snippets(text, size):
                data = snip.encode()
                blob = compress(t, cfg, data)
                assert decompress(t, cfg, blob) == data, (name, size)
                if name == "static" and blob[0] != codec.STORED:
                    assert blob[1:] == codec.compress(t, data)[1:], "static must equal codec v1"
                n += 1
                raw += len(data)
                real += len(blob)
        out["sizes"][str(size)] = {"docs": n, "raw": raw, "real": real}
    out["seconds"] = time.time() - t0
    save("short", name, out)
    return out


def run_books(name):
    cfg = CONFIGS[name]
    t = table()
    _, test = split_books()
    out = {"config": name, "books": {}}
    for path in test:
        data = path.read_bytes()
        t0 = time.time()
        blob = compress(t, cfg, data)
        verified = False
        if len(data) <= VERIFY_MAX:
            assert decompress(t, cfg, blob) == data, (name, path.name)
            verified = True
        out["books"][path.stem] = {"raw": len(data), "real": len(blob), "verified": verified,
                                   "seconds": time.time() - t0}
        print(f"{name} {path.stem}: {len(data):,} -> {len(blob):,} ({8 * len(blob) / len(data):.3f} bpc)"
              f"{' verified' if verified else ''}", flush=True)
    save("books", name, out)
    return out


def verify_book(args):
    name, stem = args
    path = next(p for p in split_books()[1] if p.stem == stem)
    data = path.read_bytes()
    t0 = time.time()
    blob = compress(table(), CONFIGS[name], data)
    t1 = time.time()
    ok = decompress(table(), CONFIGS[name], blob) == data
    return stem, len(data), len(blob), ok, t1 - t0, time.time() - t1


def run_verify(name):
    """Decode every full test book (not only the small ones) and compare."""
    jobs = [(name, p.stem) for p in sorted(split_books()[1], key=lambda p: -p.stat().st_size)]
    with Pool(3) as pool:
        for stem, raw, real, ok, enc_s, dec_s in pool.imap_unordered(verify_book, jobs):
            print(f"{name} {stem}: {raw:,} -> {real:,} round trip {'OK' if ok else 'FAILED'} "
                  f"(encode {enc_s:.0f}s, decode {dec_s:.0f}s)", flush=True)


def per_doc(name):
    """Per snippet: is the config ever worse than static?"""
    t = table()
    _, test = split_books()
    texts = [p.read_text(encoding="utf-8") for p in test]
    for size in SIZES:
        worse = better = 0
        worst = 0
        for text in texts:
            for snip in snippets(text, size):
                data = snip.encode()
                diff = len(compress(t, CONFIGS[name], data)) - len(compress(t, CONFIGS["static"], data))
                worse += diff > 0
                better += diff < 0
                worst = max(worst, diff)
        print(f"{name} {size:>6}: better {better}, worse {worse}, worst +{worst} bytes", flush=True)


def save(mode, name, out):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{mode}_{name}.json").write_text(json.dumps(out, indent=1))


def report():
    print(f"{'config':<16}" + "".join(f"{f'{s // 1000}K B':>9}{'bpc':>7}" for s in SIZES) + f"{'secs':>7}")
    for path in sorted(RESULTS.glob("short_*.json")):
        r = json.loads(path.read_text())
        row = f"{r['config']:<16}"
        for s in SIZES:
            d = r["sizes"][str(s)]
            row += f"{d['real'] / d['docs']:>9.0f}{8 * d['real'] / d['raw']:>7.3f}"
        print(row + f"{r['seconds']:>7.0f}")
    books = sorted(RESULTS.glob("books_*.json"))
    if books:
        first = json.loads(books[0].read_text())["books"]
        print(f"\n{'config':<16}" + "".join(f"{b:>8}" for b in first) + f"{'TOTAL':>8}")
        for path in books:
            r = json.loads(path.read_text())
            row = f"{r['config']:<16}"
            raw = real = 0
            for b, d in r["books"].items():
                row += f"{8 * d['real'] / d['raw']:>7.3f}{'*' if d['verified'] else ' '}"
                raw += d["raw"]
                real += d["real"]
            print(row + f"{8 * real / raw:>8.3f}")
        print("* = decoded and verified byte-identical")


def main():
    mode = sys.argv[1]
    if mode == "report":
        report()
        return
    if mode == "verify":
        run_verify(sys.argv[2])
        return
    if mode == "perdoc":
        per_doc(sys.argv[2])
        return
    names = sys.argv[2:] or list(CONFIGS)
    fn = run_short if mode == "short" else run_books
    with Pool(3) as pool:
        for out in pool.imap_unordered(fn, names):
            print(f"done {mode} {out['config']}", flush=True)
    report()


if __name__ == "__main__":
    main()
