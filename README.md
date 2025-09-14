# Bot — Regression Channel (Binance USDT-M)

Long-only trend implementation on 5-minute bars with optional live order placement and safety brackets (TP/SL reduce-only). Includes data fetch, backtests, and portfolio aggregation.

---

## 1) Environment

```bash
# Ubuntu/WSL example
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

.env (project root):

BINANCE_KEY=
BINANCE_SECRET=

2) Repo layout

backtests/
  fetch_binance_perps.py        # fetch 1m perp data (top-N or explicit)
  regime_switch_backtest.py     # regression channel MR/Trend backtest
  portfolio_aggregate.py        # combine *_perf.csv into portfolio
scripts/
  run_pipeline_cron.sh          # safe cron wrapper for pipeline
trading/
  signal_executor.py            # live signal + orders (long-only)
  ensure_brackets.py            # add reduce-only TP/SL on open positions
run_pipeline.py                 # fetch → backtest (trend,long-only) → portfolio

3) Quick start
Fetch 180d of data for top-10, backtest ETH/XRP (trend,long-only), then build 50/50 portfolio:

bash
. .venv/bin/activate
python run_pipeline.py --days 180 --fee_bps 12

Artifacts:

Summary: trend_longonly_eth_xrp_<DAYS>d_5T_W800K2.0_fee<fee>.csv

Portfolio: portfolio_eth_xrp_<DAYS>d_5T_W800K2.0_fee<fee>.csv

Per-asset perf: in data/perp_1m_data/*_regime_trend_w800_k2.0_perf.csv

Reuse existing data:

bash
python run_pipeline.py --days 180 --fee_bps 12 --no_fetch

4) Live trading (manual run)
Dry-run first:

bash
python trading/signal_executor.py \
  --symbols ETH/USDT:USDT,XRP/USDT:USDT \
  --window 800 --k 2.0 --stop_k 1.0 \
  --qty_usdt 25 --minutes 4320 --dry_run
Place orders if signal says ENTER/EXIT:


python trading/signal_executor.py \
  --symbols ETH/USDT:USDT,XRP/USDT:USDT \
  --window 800 --k 2.0 --stop_k 1.0 \
  --qty_usdt 25 --minutes 4320
Ensure TP/SL reduce-only brackets exist for open positions:

python trading/ensure_brackets.py --symbols ETH/USDT:USDT,XRP/USDT:USDT

5) Cron wrappers
Pipeline (fetch→backtest→portfolio) with lock + daily log:


bash scripts/run_pipeline_cron.sh
Signals + brackets with lock + daily log:


bash trading/signal_cron.sh
Example crontab (crontab -e):

swift
Copiar código
*/15 * * * * /bin/bash /home/danielcahli/projetos/bot-regression-channel/trading/signal_cron.sh
30 2 * * *   /bin/bash /home/danielcahli/projetos/bot-regression-channel/scripts/run_pipeline_cron.sh
Logs:

logs/trader_YYYY-MM-DD.log

logs/pipeline_YYYY-MM-DD.log

Locks:

/tmp/signal_exec.lock

/tmp/algo_pipeline_lock

6) Strategy summary
Indicator: Rolling linear regression channel on 5-minute closes
mid = intercept, band = k * sigma(residuals), upper/lower = mid ± band

Trend entries: close > upper and slope > 0

Stop: close < mid - stop_k * sigma → flat

Costs: fee_bps + slippage_bps applied on position changes

Portfolio: per-bar equal-weight across available series with row-wise renorm

Key params:

window=800, k=2.0, stop_k=1.0, resample=5T

7) Safety and notes
Use testnet credentials when testing real order flow.

Maker orders use GTX to avoid taking; placement nudges beyond bid/ask by ticks.

ensure_brackets.py skips if any reduce-only order is already open.

All times are UTC; resampling is right-labeled and right-closed to avoid lookahead.

8) Reproduce backtest standalone
bash
python backtests/regression-channel/regime_switch_backtest.py \
  --csv data/perp_1m_data/ETHUSDT:USDT_1m_180d.csv data/perp_1m_data/XRPUSDT:USDT_1m_180d.csv \
  --resample 5T --mode trend --long_only --window 800 --k 2.0 --stop_k 1.0 \
  --fee_bps 12 --summary_out trend_eth_xrp_LS_180d_5T.csv
Then:

python backtests/regression-channel/portfolio_aggregate.py \
  --csv data/perp_1m_data/ETHUSDT:USDT_1m_180d_regime_trend_w800_k2.0_perf.csv \
         data/perp_1m_data/XRPUSDT:USDT_1m_180d_regime_trend_w800_k2.0_perf.csv \
  --out_perf portfolio_eth_xrp_180d_5T_W800K2.0_fee12.csv

9) Troubleshooting
Missing CSVs after --no_fetch: rerun without --no_fetch.

Binance filters error: ensure symbol is BASE/USDT:USDT and futures are enabled.

Rate limits: CCXT enableRateLimit=True is set; heavy fetches still sleep per call.

Timezone: all timestamps stored as UTC; do not pass local naive times.