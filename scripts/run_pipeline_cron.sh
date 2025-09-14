#!/usr/bin/env bash
# scripts/run_pipeline_cron.sh — locked cron wrapper for the LS backtest pipeline
#
# What it does:
# - Strict bash safety flags.
# - Runs from repo root so paths/venv resolve.
# - Appends stdout+stderr to a daily log file.
# - Uses a non-blocking lock to avoid overlapping runs.
# - Activates venv and executes run_pipeline.py (which now enables shorts).
# - Exits with the child exit code so cron can alert on failures.

set -euo pipefail
cd /home/danielcahli/projetos/bot-regression-channel

LOG="logs/pipeline_$(date +%F).log"
mkdir -p logs

# Fail immediately if another run holds the lock.
exec flock -n /tmp/bot_regression_pipeline.lock -c '
  . .venv/bin/activate

  {
    echo "=== START $(date -Is) ==="
    echo "python=$(command -v python3)"

    # Run pipeline:
    # - fetch 1m data for top-10 (unless you add --no_fetch below)
    # - backtest ETH/XRP in TREND mode with LONG+SHORT enabled
    # - build equal-weight portfolio CSV
    nice -n 10 python3 run_pipeline.py --days 180 --fee_bps 12

    ec=$?
    echo "=== END $(date -Is) ec=$ec ==="
    exit $ec

  } >> "'"$LOG"'" 2>&1
'
