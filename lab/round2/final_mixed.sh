#!/bin/bash
# Final step (detached): wait for the mixed-domain training to finish, then run the domain test on it.
# Usage from the project root: nohup bash lab/round2/final_mixed.sh > lab/round2/logs/final_mixed.log 2>&1 &
while ! grep -q "^done 20m_mixed" lab/round2/logs/20m_mixed.log; do sleep 600; done
./.venv/bin/python lab/round2/domains.py --models-only mixed_model=round2_20m_mixed.pt
echo "FINAL_MIXED_DONE $(date '+%F %T')"
