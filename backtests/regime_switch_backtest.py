#!/usr/bin/env python3
# backtests/regression-channel/regime_switch_backtest.py
"""
Regression-channel backtest with regime switching between Mean Reversion (MR) and Trend.

What this script does:
1) Loads OHLCV (or price) CSV files.
2) Optionally resamples to a coarser timeframe and localizes timezone.
3) Builds a rolling linear-regression channel (mid, upper, lower bands) with width = k * sigma(residuals).
4) Chooses regime per bar:
   - "auto": trend if rolling R^2 >= auto_r2, else MR.
   - "trend": only breakout-based entries.
   - "mr": only mean-reversion entries.
5) Generates long/short signals with entry, stop, min-hold, and cooldown rules, optional RSI filter.
6) Backtests positions with fees and slippage. Outputs *_perf.csv per input and a summary CSV.

Usage examples:
  # Batch over folder, auto regime, resample to 5 minutes, allow shorts
  ./regime_switch_backtest.py --glob "data/perp_1m/*.csv" --resample 5T --allow_short

  # Single file, force trend regime, tighter channel
  ./regime_switch_backtest.py --csv data/ETHUSDT_1m_180d.csv --mode trend --k 1.5
"""

from __future__ import annotations
import sys, argparse, glob, os
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# ---------- Config ----------
@dataclass
class Config:
    """
    All knobs for IO, indicator, and trading logic.
    """
    # Input data
    csv_path: str
    price_col: str = "close"       # column to read price from (used for signals and PnL)
    time_col: str = "timestamp"    # datetime column; will be parsed to UTC
    resample: str | None = None    # e.g. "5T" for 5-minute bars. None = no resample
    tz_localize: str | None = None # e.g. "America/New_York" for display; logic stays in UTC

    # Regression channel params
    window: int = 800              # rolling window length for linear regression
    k: float = 2.0                 # channel width multiplier: band = k * sigma(residuals)

    # Trading logic
    mode: str = "auto"             # "mr" | "trend" | "auto"
    auto_r2: float = 0.25          # regime switch threshold: trend if R^2 >= auto_r2
    z_entry: float = 0.75          # min |z-score| of prior bar vs mid to consider MR entries
    slope_min: float = 0.0         # require slope>=slope_min (long) or <=-slope_min (short) in MR
    stop_k: float = 1.0            # stop distance in band units around mid (tighter than band)
    min_hold: int = 3              # bars to hold after entry before exits allowed
    cooldown: int = 10             # bars to wait after an exit before new entries allowed
    allow_short: bool = True       # allow short trades
    long_only: bool = False        # force long-only if True

    # Trading costs (per side; applied on position changes)
    fee_bps: float = 12.0          # fees in basis points
    slippage_bps: float = 0.0      # slippage in basis points

    # Safety and numerics
    sigma_eps: float = 1e-12       # floor for sigma to avoid division by zero

    # Optional RSI filter
    rsi_len: int = 14
    rsi_long: float | None = None  # allow long entry only if RSI <= rsi_long (e.g., 35)
    rsi_short: float | None = None # allow short entry only if RSI >= rsi_short (e.g., 65)


# ---------- IO ----------
def load_data(cfg: Config) -> pd.DataFrame:
    """
    Read CSV, parse timestamps to UTC, optional timezone convert for display, sort by time,
    and optionally resample to a coarser frequency.

    Returns:
        DataFrame sorted by time, with time_col as UTC tz-aware datetimes.
    """
    df = pd.read_csv(cfg.csv_path)

    # Parse timestamp; errors become NaT and get dropped below
    ts = pd.to_datetime(df[cfg.time_col], utc=True, errors="coerce")

    # Optional timezone conversion for readability; internal math remains UTC-based
    if cfg.tz_localize:
        ts = ts.dt.tz_convert(cfg.tz_localize)

    df[cfg.time_col] = ts

    # Drop bad timestamps and ensure chronological order
    df = df.dropna(subset=[cfg.time_col]).sort_values(cfg.time_col).reset_index(drop=True)

    # Optional OHLCV resampling
    if cfg.resample:
        df = _resample_df(df, cfg)

    return df


def _resample_df(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Resample to cfg.resample using OHLC aggregation if available, else last price.
    """
    df = df.set_index(cfg.time_col)

    # If full OHLC present, use standard OHLC rules and sum volume if present
    has_ohlc = {"open", "high", "low", "close"}.issubset(df.columns)
    if has_ohlc:
        agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
        if "volume" in df.columns:
            agg["volume"] = "sum"
        out = df.resample(cfg.resample).agg(agg).dropna(subset=["close"])
    else:
        # Otherwise just resample the chosen price column with last value
        out = df[[cfg.price_col]].resample(cfg.resample).last().dropna()

    out = out.reset_index()
    # Ensure timestamps are UTC tz-aware for downstream code
    out[cfg.time_col] = pd.to_datetime(out[cfg.time_col], utc=True)
    return out.reset_index(drop=True)


# ---------- Indicators ----------
def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """
    Exponential RSI:
      - Computes up and down moves
      - Applies EMA with alpha=1/length
      - Returns RSI in [0,100], NaNs filled with neutral 50
    """
    d = series.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    up_ema = up.ewm(alpha=1/length, adjust=False).mean()
    dn_ema = dn.ewm(alpha=1/length, adjust=False).mean()
    rs = up_ema / dn_ema.replace(0.0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def rolling_regression_channel(
    price: pd.Series, window: int, k: float
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Build a rolling linear regression on a sliding window:
      - Fit y ~ a + b * t for each window.
      - mid  = predicted price at current t (using the fitted line).
      - band width = k * sigma(residuals) where sigma is std of regression residuals.
      - up   = mid + band
      - lo   = mid - band
      - slope = b
      - r2   = window R^2 (goodness of fit), used for regime auto-switching.

    Returns:
      mid, up, lo, slope, band_width, r2 as Series aligned to 'price' index.
    """
    n = len(price)
    t = np.arange(n, dtype=float)

    # Preallocate with NaNs until enough history is available
    mid = np.full(n, np.nan)
    up = np.full(n, np.nan)
    lo = np.full(n, np.nan)
    slope = np.full(n, np.nan)
    bw = np.full(n, np.nan)
    r2 = np.full(n, np.nan)

    lr = LinearRegression()

    for i in range(window, n):
        # Windowed regressors and target
        t_win = t[i - window:i].reshape(-1, 1)
        y_win = price.iloc[i - window:i].values

        # Fit y = b0 + b1 * t
        lr.fit(t_win, y_win)
        b0 = float(lr.intercept_)
        b1 = float(lr.coef_[0])

        # Residuals and R^2 on the window
        y_hat = lr.predict(t_win)
        resid = y_win - y_hat
        sse = float(np.sum(resid ** 2))
        sst = float(np.sum((y_win - y_win.mean()) ** 2))
        r2[i] = 1 - (sse / sst) if sst > 0 else 0.0

        # Channel width = k * sigma(residuals)
        sigma = float(np.std(resid, ddof=1))
        band = k * sigma

        # Midline at current index, then upper/lower bands
        mid[i] = b0 + b1 * t[i]
        up[i] = mid[i] + band
        lo[i] = mid[i] - band
        slope[i] = b1
        bw[i] = band

    idx = price.index
    return (
        pd.Series(mid, idx, name="mid"),
        pd.Series(up, idx, name="up"),
        pd.Series(lo, idx, name="lo"),
        pd.Series(slope, idx, name="slope"),
        pd.Series(bw, idx, name="band_width"),
        pd.Series(r2, idx, name="r2"),
    )


# ---------- Signals ----------
def generate_signals(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Compute indicators and generate position signals:
      - MR entries: cross back inside bands with adequate z-score and slope filter.
      - Trend entries: breakouts through bands in slope direction.
      - Exits: cross back to mid or stop around mid +/- stop_k*band.
      - Constraints: min_hold bars, cooldown bars, optional RSI filter.
    """
    # Ensure price numeric
    price = pd.to_numeric(df[cfg.price_col], errors="coerce").astype(float)

    # Regression channel
    mid, up, lo, slope, bw, r2 = rolling_regression_channel(price, cfg.window, cfg.k)

    # Assemble indicator columns
    out = df.copy()
    out["mid"], out["up"], out["lo"], out["slope"], out["band_width"], out["r2"] = mid, up, lo, slope, bw, r2

    # RSI filter values (we look at i-1 on entries to avoid lookahead)
    rsi_vals = rsi(price, cfg.rsi_len).values

    # Position vector: 1 long, -1 short, 0 flat
    pos = np.zeros(len(out), dtype=int)

    # State flags
    in_long = False
    in_short = False
    hold = 0         # bars remaining until exits allowed
    cool = 0         # cooldown bars remaining until new entries allowed

    # Raw numpy views for speed in loop
    p = price.values
    m = mid.values
    u = up.values
    l = lo.values
    s = slope.values
    band = bw.values
    r2v = r2.values

    for i in range(1, len(out)):
        # Default: carry position forward
        pos[i] = pos[i - 1]

        # Decrement timers
        hold = max(0, hold - 1)
        cool = max(0, cool - 1)

        # Sigma from band; guard against zeros
        sigma = band[i] / max(cfg.k, 1e-9)
        if not np.isfinite(sigma) or sigma < cfg.sigma_eps:
            # Skip this bar if channel is not yet defined or too narrow
            continue

        # Prior-bar z-score relative to mid
        z_prev = (p[i - 1] - m[i - 1]) / max(sigma, cfg.sigma_eps)

        # ---- Exits ----
        if in_long and hold == 0:
            # Stop below mid at stop_k*band; exit also on mean touch
            stop_level = m[i] - cfg.stop_k * band[i]
            if np.isfinite(m[i]) and p[i] <= m[i]:
                in_long = False; pos[i] = 0; cool = cfg.cooldown
            elif np.isfinite(stop_level) and p[i] <= stop_level:
                in_long = False; pos[i] = 0; cool = cfg.cooldown

        if in_short and hold == 0:
            # Stop above mid; exit also on mean touch
            stop_level = m[i] + cfg.stop_k * band[i]
            if np.isfinite(m[i]) and p[i] >= m[i]:
                in_short = False; pos[i] = 0; cool = cfg.cooldown
            elif np.isfinite(stop_level) and p[i] >= stop_level:
                in_short = False; pos[i] = 0; cool = cfg.cooldown

        # ---- Entries ----
        if not in_long and not in_short and cool == 0:
            # Decide regime for this bar
            regime = cfg.mode
            if cfg.mode == "auto":
                # Trend if R^2 above threshold; otherwise MR
                regime = "trend" if (np.isfinite(r2v[i]) and r2v[i] >= cfg.auto_r2) else "mr"

            # RSI filters at prior bar (avoid lookahead)
            r_ok_long  = True if cfg.rsi_long  is None else (rsi_vals[i - 1] <= cfg.rsi_long)
            r_ok_short = True if cfg.rsi_short is None else (rsi_vals[i - 1] >= cfg.rsi_short)

            if regime == "mr":
                # Mean reversion:
                # Long: crossed up through lower band; slope non-negative enough; sufficiently far below mid at i-1
                cross_long = np.isfinite(l[i - 1]) and np.isfinite(l[i]) and p[i - 1] <= l[i - 1] and p[i] > l[i]
                if cross_long and s[i] >= cfg.slope_min and z_prev <= -cfg.z_entry and r_ok_long:
                    in_long = True; pos[i] = 1; hold = cfg.min_hold

                # Short: crossed down through upper band; slope non-positive enough; far above mid
                elif cfg.allow_short and not cfg.long_only:
                    cross_short = np.isfinite(u[i - 1]) and np.isfinite(u[i]) and p[i - 1] >= u[i - 1] and p[i] < u[i]
                    if cross_short and s[i] <= -cfg.slope_min and z_prev >= cfg.z_entry and r_ok_short:
                        in_short = True; pos[i] = -1; hold = cfg.min_hold

            else:
                # Trend following:
                # Long: breakout above upper band with positive slope
                brk_long = np.isfinite(u[i - 1]) and np.isfinite(u[i]) and p[i - 1] < u[i - 1] and p[i] >= u[i] and s[i] > 0
                if brk_long:
                    in_long = True; pos[i] = 1; hold = cfg.min_hold

                # Short: breakdown below lower band with negative slope
                elif cfg.allow_short and not cfg.long_only:
                    brk_short = np.isfinite(l[i - 1]) and np.isfinite(l[i]) and p[i - 1] > l[i - 1] and p[i] <= l[i] and s[i] < 0
                    if brk_short:
                        in_short = True; pos[i] = -1; hold = cfg.min_hold

    out["pos"] = pos
    return out


# ---------- Backtest ----------
def _bars_per_year(perf: pd.DataFrame, time_col: str) -> float:
    """
    Estimate bars/year from the median time delta. Fallback to ~1-min US equities pace.
    """
    ts = perf[time_col]
    if not pd.api.types.is_datetime64_any_dtype(ts):
        return 252 * 6.5 * 60
    dt = ts.diff().median()
    if pd.isna(dt) or dt == pd.Timedelta(0):
        return 252 * 6.5 * 60
    return (365.0 * 24 * 3600) / (dt / pd.Timedelta(seconds=1))


def backtest(signals: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Convert positions to returns with simple close-to-close PnL and trading costs.

    Returns:
      perf DataFrame with columns:
        [time, price, mid, up, lo, slope, band_width, r2, pos,
         ret_mkt, ret_gross, cost, ret_net, equity]
    """
    df = signals.copy()

    # Inputs
    price = pd.to_numeric(df[cfg.price_col], errors="coerce").astype(float)
    pos = df["pos"].fillna(0).astype(int)

    # Market return per bar
    ret = price.pct_change().fillna(0.0)

    # Strategy gross return: previous bar's position times bar return
    strat = pos.shift(1).fillna(0) * ret

    # Trading costs per unit position change (round turns cost twice)
    per_side = (cfg.fee_bps + cfg.slippage_bps) / 1e4
    cost = (pos - pos.shift(1)).abs().fillna(0) * per_side

    # Net returns and equity curve
    net = strat - cost
    eq = (1.0 + net).cumprod()

    # Package outputs
    out = df[[cfg.time_col, cfg.price_col, "mid", "up", "lo", "slope", "band_width", "r2", "pos"]].copy()
    out["ret_mkt"] = ret
    out["ret_gross"] = strat
    out["cost"] = cost
    out["ret_net"] = net
    out["equity"] = eq
    return out


def summarize(perf: pd.DataFrame, time_col: str) -> pd.Series:
    """
    Compute summary performance stats:
      - cum_return: total return over sample (equity[-1] - 1)
      - sharpe: annualized using bars_per_year and rf=0
      - max_drawdown: min of equity / cummax - 1
      - volatility: annualized std of net returns
      - avg_bar_return: mean net return per bar
      - bars_per_year, bars, trades
    """
    net = perf["ret_net"].fillna(0.0)
    eq = perf["equity"].ffill()

    n = _bars_per_year(perf, time_col)
    avg = float(net.mean())
    vol = float(net.std(ddof=1))
    sharpe = np.sqrt(n) * (avg / vol) if vol > 0 else np.nan

    cumret = float(eq.iloc[-1] - 1.0) if len(eq) else np.nan
    dd = (eq / eq.cummax() - 1.0).min() if len(eq) else np.nan
    trades = int((perf["pos"].diff().abs() > 0).sum())

    return pd.Series(
        dict(
            cum_return=cumret,
            sharpe=sharpe,
            max_drawdown=float(dd),
            volatility=float(vol * np.sqrt(n)),
            avg_bar_return=avg,
            bars_per_year=float(n),
            bars=int(len(perf)),
            trades=trades,
        )
    )


# ---------- Batch ----------
def parse_args():
    """
    CLI for batch backtesting.
      Data selection:
        --glob  : glob pattern for CSVs (mutually exclusive with --csv)
        --csv   : explicit list of CSV paths
      Time/columns:
        --price_col, --time_col, --resample, --tz
      Regime and channel:
        --mode, --auto_r2, --window, --k, --z_entry, --slope_min, --stop_k
        --min_hold, --cooldown, --allow_short, --long_only
      Costs and filters:
        --fee_bps, --slippage_bps, --rsi_len, --rsi_long, --rsi_short
      Output:
        --summary_out
    """
    ap = argparse.ArgumentParser(description="Regression-channel with regime switching (MR vs Trend).")

    # Choose input set
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--glob")
    g.add_argument("--csv", nargs="+")

    # Columns and preprocessing
    ap.add_argument("--price_col", default="close")
    ap.add_argument("--time_col", default="timestamp")
    ap.add_argument("--resample", default="5T")
    ap.add_argument("--tz", dest="tz_localize", default=None)

    # Regime, channel, and entries
    ap.add_argument("--mode", choices=["mr", "trend", "auto"], default="auto")
    ap.add_argument("--auto_r2", type=float, default=0.25)
    ap.add_argument("--window", type=int, default=800)
    ap.add_argument("--k", type=float, default=2.0)
    ap.add_argument("--z_entry", type=float, default=0.75)
    ap.add_argument("--slope_min", type=float, default=0.0)
    ap.add_argument("--stop_k", type=float, default=1.0)
    ap.add_argument("--min_hold", type=int, default=3)
    ap.add_argument("--cooldown", type=int, default=10)
    ap.add_argument("--allow_short", action="store_true")
    ap.add_argument("--long_only", action="store_true")

    # Costs and RSI filter
    ap.add_argument("--fee_bps", type=float, default=12.0)
    ap.add_argument("--slippage_bps", type=float, default=0.0)
    ap.add_argument("--rsi_len", type=int, default=14)
    ap.add_argument("--rsi_long", type=float, default=None)
    ap.add_argument("--rsi_short", type=float, default=None)

    # Output summary path
    ap.add_argument("--summary_out", default="regime_summary.csv")
    return ap.parse_args()


def main():
    """
    Run backtests for each input CSV:
      - Skip files that already look like *_perf.csv to avoid recursion.
      - Save one *_perf.csv per input.
      - Print OK/ERROR per file.
      - Save a sorted summary CSV across all files by Sharpe.
    """
    args = parse_args()

    # Build list of input paths
    paths = sorted(glob.glob(args.glob)) if args.glob else args.csv
    # Avoid reprocessing previously generated performance files
    paths = [p for p in paths if "_perf" not in os.path.basename(p)]
    if not paths:
        print("No CSVs found.", file=sys.stderr)
        return 2

    rows: List[dict] = []

    for pth in paths:
        try:
            # Assemble config for this file
            cfg = Config(
                csv_path=pth,
                price_col=args.price_col,
                time_col=args.time_col,
                resample=args.resample,
                tz_localize=args.tz_localize,
                window=args.window,
                k=args.k,
                mode=args.mode,
                auto_r2=args.auto_r2,
                z_entry=args.z_entry,
                slope_min=args.slope_min,
                stop_k=args.stop_k,
                min_hold=args.min_hold,
                cooldown=args.cooldown,
                allow_short=args.allow_short,
                long_only=args.long_only,
                fee_bps=args.fee_bps,
                slippage_bps=args.slippage_bps,
                rsi_len=args.rsi_len,
                rsi_long=args.rsi_long,
                rsi_short=args.rsi_short,
            )

            # Load -> indicators/signals -> backtest
            df = load_data(cfg)
            sig = generate_signals(df, cfg)
            perf = backtest(sig, cfg)

            # Aggregate stats for summary
            stats = summarize(perf, cfg.time_col).to_dict()
            stats["file"] = os.path.basename(pth)
            stats["symbol_hint"] = os.path.basename(pth).split("_")[0]
            rows.append(stats)

            # Persist per-file performance CSV next to input
            out_path = pth.rsplit(".", 1)[0] + f"_regime_{cfg.mode}_w{cfg.window}_k{cfg.k}_perf.csv"
            perf.to_csv(out_path, index=False)
            print(f"OK: {pth} -> {out_path}  rows={len(perf)}")

        except Exception as e:
            # Continue other files while reporting the error
            print(f"ERROR: {pth}: {e}", file=sys.stderr)

    # Write and print summary table if any results were produced
    if rows:
        df_sum = pd.DataFrame(rows)
        cols = [
            "file",
            "symbol_hint",
            "cum_return",
            "sharpe",
            "max_drawdown",
            "volatility",
            "avg_bar_return",
            "bars_per_year",
            "bars",
            "trades",
        ]
        # Keep only columns that are present
        df_sum = df_sum[[c for c in cols if c in df_sum.columns]]
        # Sort by Sharpe descending for quick inspection
        df_sum.sort_values("sharpe", ascending=False, inplace=True)

        # Save and print
        df_sum.to_csv(args.summary_out, index=False)
        print(f"\nSaved summary: {args.summary_out}")
        print(df_sum.to_string(index=False))

    return 0


if __name__ == "__main__":
    # Exit with the return code from main()
    sys.exit(main())
