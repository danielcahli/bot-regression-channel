#!/usr/bin/env bash
set -euo pipefail
cd /home/danielcahli/projetos/algo-trading       # go to project root
LOG="logs/pipeline_$(date +%F).log"              # daily rolling log
mkdir -p logs                                    # ensure logs/ exists
exec flock -n /tmp/algo_pipeline_lock -c '       # avoid overlapping runs
  . .venv/bin/activate
  {
    echo "=== START $(date -Is) ==="
    echo "git=$(command -v git)"; git --version || true
    export VERSIONEER_OVERRIDE="0+local"         # <- key: bypass Git in versioneer
    nice -n 10 .venv/bin/python run_pipeline.py --days 180 --fee_bps 12
    ec=$?
    echo "=== END $(date -Is) ec=$ec ==="
    exit $ec
  } 2>&1 | tee -a "'"$LOG"'"
'
