"""Measure classic compressors on the sample corpus.

Usage: python3 lab/baselines.py corpus/samples/*
"""
import subprocess
import sys
from pathlib import Path

COMPRESSORS = {
    "gzip -9": ["gzip", "-9", "-c"],
    "bzip2 -9": ["bzip2", "-9", "-c"],
    "xz -9e": ["xz", "-9e", "-c"],
    "zstd -22": ["zstd", "--ultra", "-22", "-c", "-q"],
    "brotli -q11": ["brotli", "-q", "11", "--large_window=30", "-c"],
}


def compressed_size(cmd, data):
    return len(subprocess.run(cmd, input=data, capture_output=True, check=True).stdout)


def main(paths):
    names = list(COMPRESSORS)
    print(f"{'file':<22}{'size':>10}  " + "".join(f"{n:>13}" for n in names))
    for path in map(Path, paths):
        data = path.read_bytes()
        row = f"{path.name:<22}{len(data):>10}  "
        for name in names:
            size = compressed_size(COMPRESSORS[name], data)
            row += f"{8 * size / len(data):>9.3f} bpc"
        print(row)
    print("\nbpc = bits per input byte (8.000 = no gain, >8 = negative compression)")


if __name__ == "__main__":
    main(sys.argv[1:])
