"""How much does a big shared reference save, and how does that scale with its size?

Encoder and decoder both hold the same reference file. Only the "patch" (the
target compressed against the reference) would go on paper. zstd --patch-from
is the stand-in compressor: LZ matching into the reference + entropy coding.

References: the first N MB of enwik8. Targets:
  wiki held-out  enwik8[95M:96M], never part of any reference
  wiki in table  enwik8[50M:51M], inside the 90 MB reference only
  python source  different domain from the reference

Usage: python3 lab/reference.py
"""
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENWIK8 = ROOT / "corpus" / "enwik8"
MB = 1_000_000
REF_SIZES_MB = [0, 1, 10, 30, 90]


def zstd_size(target, ref, workdir):
    out = workdir / "out.zst"
    cmd = ["zstd", "-q", "-f", "--ultra", "-22", "--long=31", str(target), "-o", str(out)]
    if ref is not None:
        cmd.insert(1, f"--patch-from={ref}")
    subprocess.run(cmd, check=True)
    return out.stat().st_size


def main():
    enwik = ENWIK8.read_bytes()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        targets = {
            "wiki held-out": enwik[95 * MB:96 * MB],
            "wiki in table": enwik[50 * MB:51 * MB],
            "python source": (ROOT / "corpus" / "samples" / "pysrc_1M.txt").read_bytes(),
        }
        for name, data in targets.items():
            (work / name).write_bytes(data)
        refs = {}
        for size in REF_SIZES_MB:
            if size:
                refs[size] = work / f"ref{size}"
                refs[size].write_bytes(enwik[:size * MB])

        print(f"{'target (1 MB)':<16}" + "".join(f"{f'ref {s} MB':>13}" for s in REF_SIZES_MB))
        for name, data in targets.items():
            row = f"{name:<16}"
            for size in REF_SIZES_MB:
                compressed = zstd_size(work / name, refs.get(size), work)
                row += f"{8 * compressed / len(data):>9.3f} bpc"
            print(row, flush=True)


if __name__ == "__main__":
    main()
