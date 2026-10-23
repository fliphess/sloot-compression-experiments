"""E11 benchmark: C vs Python speed, table load time, peak RSS. Writes lab/c/results/bench.json + prints a table.

  python3 lab/c/bench.py c          C only (encode/decode MB/s per test set, load time, peak RSS)
  python3 lab/c/bench.py py TABLE   Python decode timing on a small sample + load time + peak RSS (TABLE = v1 | 3gb)

C timings come from `steno -v` (coding only, excludes process start and table mmap). Python encode timings are
taken from the reference run (lab/c/data/py/<TABLE>_times.jsonl); Python decode is measured here on a sample.
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "data"
STENO = HERE / "steno"
CTAB = {"v1": ROOT / "tables" / "c_en-gutenberg-v1.bin", "3gb": ROOT / "tables" / "c_exp_table_3gb.bin"}
OUT = HERE / "results" / "bench.json"


def load_json():
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def save(res):
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))


def timed(args):
    """Run under /usr/bin/time -l; return (stderr, wall s, peak RSS bytes)."""
    t0 = time.time()
    r = subprocess.run(["/usr/bin/time", "-l"] + args, capture_output=True, text=True)
    wall = time.time() - t0
    if r.returncode:
        raise SystemExit(r.stderr)
    rss = int(re.search(r"(\d+)\s+maximum resident set size", r.stderr).group(1))
    foot = int(re.search(r"(\d+)\s+peak memory footprint", r.stderr).group(1))
    return r.stderr, wall, (rss, foot)


def steno_times(stderr):
    return {Path(m.group(1)).name: float(m.group(2))
            for m in re.finditer(r"^(?:compress|decompress) (\S+) \d+ -> \d+ bytes ([0-9.]+) s$", stderr, re.M)}


def bench_c():
    res = load_json()
    items = json.loads((DATA / "manifest.json").read_text())
    sets = {"snip1000": [], "snip30000": [], "book": []}
    for it in items:
        if it["kind"] in sets:
            sets[it["kind"]].append(it)
    tmp = Path("/tmp/e11_bench")
    tmp.mkdir(exist_ok=True)
    c = res.setdefault("c", {})
    for tname, tpath in CTAB.items():
        info = subprocess.run([str(STENO), "info", str(tpath)], capture_output=True, text=True).stdout.strip()
        c.setdefault(tname, {})["info"] = info
        # load: wall time of a process that maps the table and codes one 1 KB snippet (warm page cache)
        one = sets["snip1000"][0]["path"]
        walls = []
        for _ in range(3):
            _, wall, rss = timed([str(STENO), "compress", "-a", str(tpath), one, str(tmp / "one.stn")])
            walls.append(wall)
        c[tname]["one_1kb_process_wall_s"] = min(walls)
        c[tname]["one_1kb_peak_rss_mb"] = rss[0] / 1e6
        c[tname]["one_1kb_peak_footprint_mb"] = rss[1] / 1e6
        for mode, flag in (("static", "-s"), ("adaptive", "-a")):
            for sname, its in sets.items():
                pairs = []
                for it in its:
                    pairs += [it["path"], str(tmp / f"{it['name']}.stn")]
                runs = []
                for rep_i in range(2):  # run 1 may page the table in from disk; run 2 is warm
                    err, _, rss_e = timed([str(STENO), "compress", "-v", flag, str(tpath)] + pairs)
                    te = steno_times(err)
                    dpairs = []
                    for it in its:
                        dpairs += [str(tmp / f"{it['name']}.stn"), str(tmp / f"{it['name']}.dec")]
                    err, _, rss_d = timed([str(STENO), "decompress", "-v", str(tpath)] + dpairs)
                    td = steno_times(err)
                    runs.append((te, td, rss_e, rss_d))
                raw = sum(Path(it["path"]).stat().st_size for it in its)
                for it in its:
                    assert (tmp / f"{it['name']}.dec").read_bytes() == Path(it["path"]).read_bytes()
                row = {"raw": raw}
                for label, (te, td, rss_e, rss_d) in zip(("first", "warm"), runs):
                    enc_s = sum(te[Path(it["path"]).name] for it in its)
                    dec_s = sum(td[f"{it['name']}.stn"] for it in its)
                    row[label] = {"enc_s": enc_s, "dec_s": dec_s, "enc_MBps": raw / 1e6 / enc_s,
                                  "dec_MBps": raw / 1e6 / dec_s, "peak_rss_enc_mb": rss_e[0] / 1e6,
                                  "peak_rss_dec_mb": rss_d[0] / 1e6, "footprint_enc_mb": rss_e[1] / 1e6,
                                  "footprint_dec_mb": rss_d[1] / 1e6}
                c[tname][f"{mode}_{sname}"] = row
                print(tname, mode, sname, json.dumps(c[tname][f"{mode}_{sname}"]), flush=True)
                save(res)
    save(res)


def bench_py(tname):
    sys.path[:0] = [str(HERE), str(ROOT / "lab"), str(ROOT / "lab" / "exp_adaptive")]
    import resource
    import adaptive
    import codec
    import ref
    res = load_json()
    p = res.setdefault("py", {}).setdefault(tname, {})
    t0 = time.time()
    table = ref.load(tname)
    p["load_s"] = time.time() - t0
    cfg = ref._cfg
    items = {it["name"]: it for it in json.loads((DATA / "manifest.json").read_text())}
    sample = [n for n in items if n.startswith("snip_30000_")][:10] + ["book_1952"]
    for mode in ("static", "adaptive"):
        raw = dec_s = 0
        for name in sample:
            blob = (DATA / "py" / f"{tname}_{mode}" / f"{name}.stn").read_bytes()
            data = Path(items[name]["path"]).read_bytes()
            t1 = time.time()
            out = codec.decompress(table, blob) if mode == "static" else adaptive.decompress(table, cfg, blob)
            dec_s += time.time() - t1
            assert out == data, name
            raw += len(data)
        p[f"{mode}_decode_sample"] = {"files": sample, "raw": raw, "dec_s": dec_s, "dec_MBps": raw / 1e6 / dec_s}
        enc = [json.loads(line) for line in (DATA / "py" / f"{tname}_times.jsonl").read_text().splitlines()]
        for kind, pred in (("book", lambda n: n.startswith("book_")), ("snip30000", lambda n: n.startswith("snip_30000_")),
                           ("snip1000", lambda n: n.startswith("snip_1000_"))):
            rows = [r for r in enc if r["mode"] == mode and pred(r["name"])]
            rr, ss = sum(r["raw"] for r in rows), sum(r["enc_s"] for r in rows)
            p[f"{mode}_encode_{kind}"] = {"raw": rr, "enc_s": ss, "enc_MBps": rr / 1e6 / ss if ss else None}
        print(tname, mode, json.dumps(p[f"{mode}_decode_sample"]["dec_MBps"]), flush=True)
    p["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # bytes on macOS
    save(res)
    print(json.dumps(p, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "c":
        bench_c()
    else:
        bench_py(sys.argv[2])
