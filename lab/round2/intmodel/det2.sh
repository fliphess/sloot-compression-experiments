#!/bin/bash
# E12 determinism, second pass (after the main runs): a third run of 20 docs per model spread over all
# sizes (every 10th), default thread settings, different batch composition (5), encode only (files compared
# by SHA-256 with the main run); and the pure-integer backend on multi-window documents (round trip).
cd "$(dirname "$0")/../../.." || exit 1
PY=.venv/bin/python
R=lab/round2/intmodel/run_eval.py
set -x
unset VECLIB_MAXIMUM_THREADS OMP_NUM_THREADS OPENBLAS_NUM_THREADS
for m in 1m 5m 20m; do
  E12_TAG=_det2 $PY $R $m f32 0 10 --encode-only --batch 5
done
VECLIB_MAXIMUM_THREADS=1 E12_TAG=_det2 $PY $R 1m int 0 1 --sizes 10000 --limit 1
VECLIB_MAXIMUM_THREADS=1 E12_TAG=_det2 $PY $R 5m int 0 1 --sizes 3000 --limit 1
echo det2-done
