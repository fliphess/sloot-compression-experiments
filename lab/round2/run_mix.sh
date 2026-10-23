#!/bin/bash
# Round 2 mixing queue (detached): 3 GB table tokens -> mix with E10's 20M -> wait for the
# 3 GB-trained 20M model -> mix with it. Usage from the project root:
#   nohup bash lab/round2/run_mix.sh > lab/round2/logs/mix.log 2>&1 &
set -e
PY=./.venv/bin/python
$PY lab/round2/mix_big.py tokens exp_table_3gb
$PY lab/round2/mix_big.py mix exp_table_3gb exp_tinymodel_20m.pt
while ! grep -q "^done 20m_big" lab/round2/logs/20m_big.log; do sleep 300; done
$PY lab/round2/mix_big.py mix exp_table_3gb round2_20m_big.pt
echo "MIX_QUEUE_DONE $(date '+%F %T')"
