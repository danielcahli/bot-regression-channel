PY ?= python3
PIP ?= pip

VENV := .venv
ACT  := . $(VENV)/bin/activate

SYMS := ETH/USDT:USDT,XRP/USDT:USDT
DAYS := 180
FEE  := 12
WIN  := 800
KVAL := 2.0
STOP := 1.0
MINS := 4320
QUSDT:= 25

.PHONY: all venv install fetch backtest_ls portfolio pipeline signal signal_dry brackets logs clean

all: install

venv:
	@test -d $(VENV) || $(PY) -m venv $(VENV)

install: venv
	@$(ACT); $(PIP) install -U pip
	@$(ACT); $(PIP) install -r requirements.txt

fetch:
	@$(ACT); $(PY) backtests/regression-channel/fetch_binance_perps.py \
	  --days $(DAYS) --timeframe 1m --top 10 --out data/perp_1m_data

backtest_ls:
	@$(ACT); $(PY) backtests/regression-channel/regime_switch_backtest.py \
	  --csv data/perp_1m_data/ETHUSDT:USDT_1m_$(DAYS)d.csv data/perp_1m_data/XRPUSDT:USDT_1m_$(DAYS)d.csv \
	  --resample 5T --mode trend --allow_short \
	  --window $(WIN) --k $(KVAL) --stop_k $(STOP) \
	  --fee_bps $(FEE) --summary_out trend_long_short_eth_xrp_$(DAYS)d_5T_W$(WIN)K$(KVAL)_fee$(FEE).csv

portfolio:
	@$(ACT); $(PY) backtests/regression-channel/portfolio_aggregate.py \
	  --csv data/perp_1m_data/ETHUSDT:USDT_1m_$(DAYS)d_regime_trend_w$(WIN)_k$(KVAL)_perf.csv \
	        data/perp_1m_data/XRPUSDT:USDT_1m_$(DAYS)d_regime_trend_w$(WIN)_k$(KVAL)_perf.csv \
	  --out_perf portfolio_eth_xrp_$(DAYS)d_5T_W$(WIN)K$(KVAL)_fee$(FEE).csv

pipeline:
	@$(ACT); $(PY) run_pipeline.py --days $(DAYS) --fee_bps $(FEE)

signal:
	@$(ACT); $(PY) trading/signal_executor.py \
	  --symbols $(SYMS) --window $(WIN) --k $(KVAL) --stop_k $(STOP) \
	  --qty_usdt $(QUSDT) --minutes $(MINS)

signal_dry:
	@$(ACT); $(PY) trading/signal_executor.py \
	  --symbols $(SYMS) --window $(WIN) --k $(KVAL) --stop_k $(STOP) \
	  --qty_usdt $(QUSDT) --minutes $(MINS) --dry_run

brackets:
	@$(ACT); $(PY) trading/ensure_brackets.py --symbols $(SYMS)

logs:
	@mkdir -p logs
	@ls -1tr logs | tail -n 20

clean:
	@find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	@find . -name "*.pyc" -delete
