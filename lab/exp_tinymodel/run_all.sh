#!/bin/bash
# E10 overnight sweep: train + evaluate 1m, 5m, 20m one after another (detached, sleep-proof).
# Usage (from project root): nohup caffeinate -i bash lab/exp_tinymodel/run_all.sh > lab/exp_tinymodel/logs/queue.log 2>&1 &
cd "$(dirname "$0")"
for name in 1m 5m 20m; do
  if [ -f results/$name.json ]; then echo "skip $name (done)"; continue; fi
  echo "start $name $(date '+%F %T')"
  ../../.venv/bin/python train.py $name > logs/$name.log 2>&1 || echo "FAILED $name (see logs/$name.log)"
  echo "end $name $(date '+%F %T')"
done
echo "queue finished $(date '+%F %T')"
