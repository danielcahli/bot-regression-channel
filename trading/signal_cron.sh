#!/usr/bin/env bash
# trading/signal_cron.sh — locked cron wrapper for LONG+SHORT signal execution
#
# What it does:
# - Strict bash flags for safety.
# - Runs from repo root so paths/venv resolve.
# - Appends stdout+stderr to a daily log file.
# - Uses a non-blocking lock to prevent overlapping runs.
# - Activates venv and executes:
#     1) trading/signal_executor.py  (supports LONG and SHORT)
#     2) trading/ensure_brackets.py  (reduce-only TP/SL for any open pos)
# - Exits with the child exit code for observability in cron.

set -euo pipefail
cd /home/danielcahli/projetos/bot-regression-channel

LOG="logs/trader_$(date +%F).log"
mkdir -p logs

# Fail immediately if another run holds the lock.
exec flock -n /tmp/signal_exec.lock -c '
  . .venv/bin/activate

  {
    echo "=== START $(date -Is) ==="

    # LONG+SHORT signal executor:
    # - Symbols: edit as needed
    # - Window/K/stop_k: regression channel params on 5m bars
    # - qty_usdt: target notional per order (min-notional aware)
    # - minutes: 1m history to pull for resample (4320 = 3 days)
    nice -n 10 python3 trading/signal_executor.py \
      --symbols ETH/USDT:USDT,XRP/USDT:USDT \
      --window 800 --k 2.0 --stop_k 1.0 \
      --qty_usdt 25 --minutes 4320

    # Safety net: ensure reduce-only TP/SL exist for any open long OR short.
    python3 trading/ensure_brackets.py --symbols ETH/USDT:USDT,XRP/USDT:USDT

    ec=$?
    echo "=== END $(date -Is) ec=$ec ==="
    exit $ec

  } >> "'"$LOG"'" 2>&1
'
