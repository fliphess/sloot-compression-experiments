#!/bin/bash
# E12 determinism checks (compare the sha256 per document against the main f32 run with 1 BLAS thread).
# Usage: bash lab/round2/intmodel/det.sh > lab/round2/intmodel/logs/det.log 2>&1
cd "$(dirname "$0")/../../.." || exit 1
PY=.venv/bin/python
R=lab/round2/intmodel/run_eval.py
set -x
# 1. pure-integer backend (no BLAS, no floats) vs the f32/f64 accelerators: same files
VECLIB_MAXIMUM_THREADS=1 E12_TAG=_det $PY $R 1m int 0 1 --sizes 1000 --limit 10
VECLIB_MAXIMUM_THREADS=1 E12_TAG=_det $PY $R 5m int 0 1 --sizes 1000 --limit 3
VECLIB_MAXIMUM_THREADS=1 E12_TAG=_det $PY $R 20m int 0 1 --sizes 1000 --limit 1
VECLIB_MAXIMUM_THREADS=1 E12_TAG=_det $PY $R 20m f64 0 1 --sizes 1000,3000 --limit 2 --encode-only
# 2. second run + default thread settings (no thread env vars at all) for the BLAS backends
unset VECLIB_MAXIMUM_THREADS OMP_NUM_THREADS OPENBLAS_NUM_THREADS
for m in 1m 5m 20m; do
  E12_TAG=_detdefault $PY $R $m f32 0 1 --sizes 1000,3000,10000 --limit 3
  E12_TAG=_detdefault $PY $R $m f64 0 1 --sizes 1000,10000 --limit 2 --encode-only
done
VECLIB_MAXIMUM_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 E12_TAG=_det4 $PY $R 5m f32 0 1 --sizes 30000 --limit 1
# 3. CLI in separate processes, 30 KB snippet, cross-backend (compress int-free f32, decompress f64)
T=$(mktemp -d)
$PY -c "
import sys; sys.path[:0]=['lab/round2/intmodel','lab']
from run_eval import short_docs
open('$T/doc','wb').write([d for s,j,d in short_docs() if s==30000][7])"
VECLIB_MAXIMUM_THREADS=1 $PY lab/round2/intmodel/nncodec.py compress tables/intmodel_5m.bin $T/doc $T/doc.nn --backend f32
$PY lab/round2/intmodel/nncodec.py decompress tables/intmodel_5m.bin $T/doc.nn $T/doc.out --backend f64
cmp $T/doc $T/doc.out && echo "CLI round trip OK: $(wc -c < $T/doc) -> $(wc -c < $T/doc.nn) bytes"
shasum -a 256 $T/doc.nn
rm -r $T
echo det-done
