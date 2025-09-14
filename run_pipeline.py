#!/usr/bin/env python3
# run_pipeline.py — fetch + backtest ETH/XRP (TREND with LONG+SHORT) + portfolio
#
# Change from your previous version:
# - Removed --long_only (so longs are allowed but NOT forced).
# - Added --allow_short to enable SHORT entries in the backtest.
#
# Pipeline:
# 1) Optionally fetch recent 1m OHLCV (top-10 Binance USDT-M perps).
# 2) Ensure ETH/XRP CSVs exist.
# 3) Run regression-channel backtest in TREND mode with LONG+SHORT enabled.
# 4) Aggregate ETH/XRP *_perf.csv into an equal-weight portfolio.

import argparse, sys, subprocess
from pathlib import Path


def sh(cmd: list[str]):
    """Run a subprocess command. Exit with the child’s code on failure."""
    print(">>", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=180, help="Lookback days for fetch and file names.")
    ap.add_argument("--fee_bps", type=float, default=12.0, help="Fees in bps used by backtest.")
    ap.add_argument("--out_dir", default="data/perp_1m_data", help="Folder for raw 1m CSVs.")
    ap.add_argument("--no_fetch", action="store_true", help="Skip market data fetch step.")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Fetch 1m bars (top10). Later we use only ETH/XRP.
    if not args.no_fetch:
        sh([
            sys.executable,
            "backtests/regression-channel/fetch_binance_perps.py",
            "--days", str(args.days),
            "--timeframe", "1m",
            "--top", "10",
            "--out", str(out),
        ])

    # Expected raw data files
    eth_csv = out / f"ETHUSDT:USDT_1m_{args.days}d.csv"
    xrp_csv = out / f"XRPUSDT:USDT_1m_{args.days}d.csv"
    for p in (eth_csv, xrp_csv):
        if not p.exists():
            print(f"ERROR: missing {p}. Re-run without --no_fetch.", file=sys.stderr)
            sys.exit(2)

    # 2) Backtest per-asset (5T, TREND, LONG+SHORT)
    #    NOTE: --allow_short enables shorts; we intentionally DO NOT pass --long_only.
    summary = f"trend_long_short_eth_xrp_{args.days}d_5T_W800K2.0_fee{int(args.fee_bps)}.csv"
    sh([
        sys.executable,
        "backtests/regression-channel/regime_switch_backtest.py",
        "--csv", str(eth_csv), str(xrp_csv),
        "--resample", "5T",
        "--mode", "trend",
        "--allow_short",            # <— enable shorts
        "--window", "800",
        "--k", "2.0",
        "--stop_k", "1.0",
        "--fee_bps", str(args.fee_bps),
        "--summary_out", summary,
    ])

    # 3) Portfolio equal-weight from the produced *_perf.csv files
    eth_perf = out / f"ETHUSDT:USDT_1m_{args.days}d_regime_trend_w800_k2.0_perf.csv"
    xrp_perf = out / f"XRPUSDT:USDT_1m_{args.days}d_regime_trend_w800_k2.0_perf.csv"
    port_out = f"portfolio_eth_xrp_{args.days}d_5T_W800K2.0_fee{int(args.fee_bps)}.csv"

    sh([
        sys.executable,
        "backtests/regression-channel/portfolio_aggregate.py",
        "--csv", str(eth_perf), str(xrp_perf),
        "--out_perf", port_out,
    ])

    print("\nDone.")
    print(f"- Summary: {summary}")
    print(f"- Portfolio: {port_out}")


if __name__ == "__main__":
    main()
