#!/bin/sh
# E8c: full 3 GB tier, detached-safe. Logs: lab/exp_table/logs/run_3gb.log
set -e
cd "$(dirname "$0")/../.."
date
python3 lab/exp_table/leakage.py --procs 3
python3 lab/exp_table/manifest.py
python3 lab/exp_table/build_table.py --mb 3000 --out tables/exp_table_3gb.npz | sed 's/, "book_ids.*//'
python3 lab/exp_table/evaluate.py tables/exp_table_3gb.npz
python3 lab/exp_table/summarize.py > lab/exp_table/results/summary.md
date
echo RUN_3GB_DONE
