"""E11 test harness: C port vs the Python reference.

  python3 lab/c/test.py TABLE [MODE ...]      TABLE = v1 | 3gb, MODE = static | adaptive (default both)

For every input in lab/c/data/manifest.json: C encodes it (one steno process per table+mode, all files),
the output is compared byte for byte with the Python encoder output in lab/c/data/py/<TABLE>_<MODE>/,
then C decodes its own file and the result is compared with the input. Writes
lab/c/data/c/<TABLE>_<MODE>/{*.stn,*.dec,times.tsv} and results/<TABLE>_<MODE>.json, prints a summary.
"""
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "data"
STENO = HERE / "steno"
TABLES = {"v1": ROOT / "tables" / "c_en-gutenberg-v1.bin", "3gb": ROOT / "tables" / "c_exp_table_3gb.bin"}


def run_batch(cmd, flag, table, pairs):
    args = [str(STENO), cmd, "-v"] + ([flag] if flag else []) + [str(table)]
    for a, b in pairs:
        args += [str(a), str(b)]
    res = subprocess.run(args, capture_output=True, text=True)
    if res.returncode:
        raise SystemExit(f"steno {cmd} failed: {res.stderr}")
    times = {}
    for line in res.stderr.splitlines():
        parts = line.split()
        times[Path(parts[1]).name] = float(parts[-2])
    return times


def test(table_name, mode):
    items = json.loads((DATA / "manifest.json").read_text())
    out = DATA / "c" / f"{table_name}_{mode}"
    out.mkdir(parents=True, exist_ok=True)
    py = DATA / "py" / f"{table_name}_{mode}"
    flag = "-a" if mode == "adaptive" else "-s"
    enc_t = run_batch("compress", flag, TABLES[table_name], [(it["path"], out / f"{it['name']}.stn") for it in items])
    dec_t = run_batch("decompress", None, TABLES[table_name],
                      [(out / f"{it['name']}.stn", out / f"{it['name']}.dec") for it in items])
    groups = defaultdict(lambda: {"n": 0, "identical": 0, "py_missing": 0, "roundtrip": 0, "raw": 0, "out": 0,
                                  "enc_s": 0.0, "dec_s": 0.0, "fail": []})
    rows = []
    for it in items:
        g = groups[it["kind"]]
        data = Path(it["path"]).read_bytes()
        cblob = (out / f"{it['name']}.stn").read_bytes()
        dec = (out / f"{it['name']}.dec").read_bytes()
        pyf = py / f"{it['name']}.stn"
        ident = None
        if pyf.exists():
            ident = pyf.read_bytes() == cblob
        g["n"] += 1
        g["raw"] += len(data)
        g["out"] += len(cblob)
        g["enc_s"] += enc_t[Path(it["path"]).name]
        g["dec_s"] += dec_t[f"{it['name']}.stn"]
        if ident is None:
            g["py_missing"] += 1
        elif ident:
            g["identical"] += 1
        else:
            g["fail"].append(it["name"] + ":differs")
        if dec == data:
            g["roundtrip"] += 1
        else:
            g["fail"].append(it["name"] + ":roundtrip")
        rows.append(f"{it['name']}\t{len(data)}\t{len(cblob)}\t{cblob[0] if cblob else '-'}\t{ident}\t{dec == data}\t"
                    f"{enc_t[Path(it['path']).name]:.6f}\t{dec_t[it['name'] + '.stn']:.6f}")
    (out / "times.tsv").write_text("name\traw\tout\tmode\tidentical_to_py\troundtrip\tenc_s\tdec_s\n" + "\n".join(rows) + "\n")
    res = {"table": table_name, "mode": mode, "groups": groups}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"{table_name}_{mode}.json").write_text(json.dumps(res, indent=1))
    print(f"\n{table_name} {mode}")
    print(f"{'set':<10}{'n':>4}{'identical':>10}{'py_pend':>8}{'roundtrip':>10}{'raw':>11}{'out':>10}{'bpc':>7}"
          f"{'C enc MB/s':>11}{'C dec MB/s':>11}")
    for kind, g in groups.items():
        print(f"{kind:<10}{g['n']:>4}{g['identical']:>10}{g['py_missing']:>8}{g['roundtrip']:>10}{g['raw']:>11,}"
              f"{g['out']:>10,}{8 * g['out'] / max(1, g['raw']):>7.3f}{g['raw'] / 1e6 / max(1e-9, g['enc_s']):>11.2f}"
              f"{g['raw'] / 1e6 / max(1e-9, g['dec_s']):>11.2f}" + (f"  FAIL {g['fail'][:5]}" if g["fail"] else ""))
    return res


if __name__ == "__main__":
    modes = sys.argv[2:] or ["static", "adaptive"]
    for m in modes:
        test(sys.argv[1], m)
