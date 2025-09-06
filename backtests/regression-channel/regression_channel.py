#!/usr/bin/env python3
# batch_backtest.py
# Regression-channel mean reversion with stricter entries, cooldown, min-hold, and optional resampling.
# Usage examples:
#   ./batch_backtest.py --glob "../../data/perp_1m_data/*.csv"
#   ./batch_backtest.py --glob "../../data/perp_1m_data/*.csv" --resample "5T" --window 800 --k 2.0 --z_entry 1.25 --stop_k 1.0

from __future__ import annotations

import sys, argparse, glob, os
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# ---------------- Core config ----------------
@dataclass
class Config:
    csv_path: str
    price_col: str = "close"
    time_col: str = "timestamp"
    window: int = 800
    k: float = 2.0
    stop_k: float = 1.0
    allow_short: bool = False
    fee_bps: float = 10.0         # per side
    slippage_bps: float = 0.0     # per side
    tz_localize: str | None = None
    resample: str | None = None   # e.g. "5T"
    z_entry: float = 1.25
    slope_min: float = 0.0
    min_hold: int = 3
    cooldown: int = 10
    sigma_eps: float = 1e-12


# ---------------- Data loading ----------------
def load_data(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.csv_path)

    if cfg.time_col not in df.columns:
        raise ValueError(f"Missing time column: {cfg.time_col}")
    if cfg.price_col not in df.columns:
        raise ValueError(f"Missing price column: {cfg.price_col}")

    ts = pd.to_datetime(df[cfg.time_col], utc=True, errors="coerce")
    if cfg.tz_localize:
        ts = ts.dt.tz_convert(cfg.tz_localize)
    df[cfg.time_col] = ts
    df = df.dropna(subset=[cfg.time_col]).sort_values(cfg.time_col).reset_index(drop=True)

    if cfg.resample:
        df = _resample_df(df, cfg)
    return df


def _resample_df(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.set_index(cfg.time_col)
    has_ohlc = {"open", "high", "low", "close"}.issubset(df.columns)
    if has_ohlc:
        agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
        if "volume" in df.columns:
            agg["volume"] = "sum"
        df_r = df.resample(cfg.resample).agg(agg).dropna(subset=["close"])
    else:
        df_r = df[[cfg.price_col]].resample(cfg.resample).last().dropna()
    df_r = df_r.reset_index()
    df_r[cfg.time_col] = pd.to_datetime(df_r[cfg.time_col], utc=True)
    return df_r.reset_index(drop=True)


# ---------------- Rolling regression channel ----------------
def rolling_regression_channel(
    price: pd.Series, window: int, k: float
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    n = len(price)
    t_all = np.arange(n, dtype=float)
    mid = np.full(n, np.nan); up = np.full(n, np.nan); lo = np.full(n, np.nan)
    slope = np.full(n, np.nan); bw = np.full(n, np.nan)
    lr = LinearRegression()

    for i in range(window, n):
        t_win = t_all[i - window : i].reshape(-1, 1)
        y_win = price.iloc[i - window : i].values
        lr.fit(t_win, y_win)
        beta0 = float(lr.intercept_); beta1 = float(lr.coef_[0])

        y_hat_now = beta0 + beta1 * t_all[i]
        y_hat_win = lr.predict(t_win)
        resid = y_win - y_hat_win
        sigma = np.std(resid, ddof=1)

        band = k * sigma
        mid[i] = y_hat_now
        up[i] = y_hat_now + band
        lo[i] = y_hat_now - band
        slope[i] = beta1
        bw[i] = band

    idx = price.index
    return (
        pd.Series(mid, idx, name="mid"),
        pd.Series(up, idx, name="up"),
        pd.Series(lo, idx, name="lo"),
        pd.Series(slope, idx, name="slope"),
        pd.Series(bw, idx, name="band_width"),
    )


# ---------------- Signals (z-threshold, cross, cooldown, min-hold) ----------------
def generate_signals(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    price = pd.to_numeric(df[cfg.price_col], errors="coerce").astype(float)
    mid, up, lo, slope, band_width = rolling_regression_channel(price, cfg.window, cfg.k)

    out = df.copy()
    out["mid"], out["up"], out["lo"], out["slope"], out["band_width"] = mid, up, lo, slope, band_width

    pos = np.zeros(len(out), dtype=int)
    in_long = False; in_short = False
    hold = 0; cooldown = 0

    p = price.values; m = mid.values; u = up.values; l = lo.values; s = slope.values; bw = band_width.values

    for i in range(1, len(out)):
        pos[i] = pos[i - 1]
        hold = max(0, hold - 1)
        cooldown = max(0, cooldown - 1)

        sigma = bw[i] / max(cfg.k, 1e-9)
        if not np.isfinite(sigma) or sigma < cfg.sigma_eps:
            continue

        z_prev = (p[i - 1] - m[i - 1]) / max(sigma, cfg.sigma_eps)

        # Exits (after min_hold)
        if in_long and hold == 0:
            stop_level = l[i] + (-cfg.stop_k * bw[i]) if np.isfinite(l[i]) and np.isfinite(bw[i]) else -np.inf
            if np.isfinite(m[i]) and p[i] >= m[i]:
                in_long = False; pos[i] = 0; cooldown = cfg.cooldown
            elif p[i] <= stop_level:
                in_long = False; pos[i] = 0; cooldown = cfg.cooldown

        if in_short and hold == 0:
            stop_level = u[i] + cfg.stop_k * bw[i] if np.isfinite(u[i]) and np.isfinite(bw[i]) else np.inf
            if np.isfinite(m[i]) and p[i] <= m[i]:
                in_short = False; pos[i] = 0; cooldown = cfg.cooldown
            elif p[i] >= stop_level:
                in_short = False; pos[i] = 0; cooldown = cfg.cooldown

        # Entries (flat, not cooling)
        if not in_long and not in_short and cooldown == 0:
            cross_long = (np.isfinite(l[i - 1]) and np.isfinite(l[i]) and p[i - 1] <= l[i - 1] and p[i] > l[i])
            if cross_long and s[i] >= cfg.slope_min and z_prev <= -cfg.z_entry:
                in_long = True; pos[i] = 1; hold = cfg.min_hold
            elif cfg.allow_short:
                cross_short = (np.isfinite(u[i - 1]) and np.isfinite(u[i]) and p[i - 1] >= u[i - 1] and p[i] < u[i])
                if cross_short and s[i] <= -cfg.slope_min and z_prev >= cfg.z_entry:
                    in_short = True; pos[i] = -1; hold = cfg.min_hold

    out["pos"] = pos
    return out


# ---------------- Backtest ----------------
def _bars_per_year(perf: pd.DataFrame, time_col: str) -> float:
    ts = perf[time_col]
    if not pd.api.types.is_datetime64_any_dtype(ts):
        return 252 * 6.5 * 60
    dt = ts.diff().median()
    if pd.isna(dt) or dt == pd.Timedelta(0):
        return 252 * 6.5 * 60
    return (365.0 * 24 * 3600) / (dt / pd.Timedelta(seconds=1))


def backtest(signals: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = signals.copy()
    price = pd.to_numeric(df[cfg.price_col], errors="coerce").astype(float)
    pos = df["pos"].fillna(0).astype(int)

    ret = price.pct_change().fillna(0.0)
    strat_ret = pos.shift(1).fillna(0) * ret

    per_side = (cfg.fee_bps + cfg.slippage_bps) / 1e4
    pos_change = (pos - pos.shift(1)).abs().fillna(0)
    cost = pos_change * per_side

    net_ret = strat_ret - cost
    equity = (1.0 + net_ret).cumprod()

    out = df[[cfg.time_col, cfg.price_col, "mid", "up", "lo", "slope", "band_width", "pos"]].copy()
    out["ret_mkt"] = ret
    out["ret_gross"] = strat_ret
    out["cost"] = cost
    out["ret_net"] = net_ret
    out["equity"] = equity
    return out


# ---------------- Metrics ----------------
def summarize(perf: pd.DataFrame, time_col: str) -> pd.Series:
    net = perf["ret_net"].fillna(0.0)
    eq = perf["equity"].ffill()
    n_per_year = _bars_per_year(perf, time_col)

    avg = float(net.mean())
    vol = float(net.std(ddof=1))
    sharpe = np.sqrt(n_per_year) * (avg / vol) if vol > 0 else np.nan
    cumret = float(eq.iloc[-1] - 1.0) if len(eq) else np.nan
    roll_max = eq.cummax()
    drawdown = eq / roll_max - 1.0
    max_dd = float(drawdown.min()) if len(drawdown) else np.nan
    trades = int((perf["pos"].diff().abs() > 0).sum())

    return pd.Series(
        dict(
            cum_return=cumret,
            sharpe=sharpe,
            max_drawdown=max_dd,
            volatility=float(vol * np.sqrt(n_per_year)),
            avg_bar_return=avg,
            bars_per_year=float(n_per_year),
            bars=int(len(perf)),
            trades=trades,
        )
    )


# ---------------- Batch runner ----------------
def parse_args():
    ap = argparse.ArgumentParser(description="Batch backtest regression-channel strategy with stricter entries.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--glob", help='Glob for input CSVs, e.g. "../../data/perp_1m_data/*.csv"')
    g.add_argument("--csv", nargs="+", help="Explicit list of CSV paths")
    ap.add_argument("--price_col", default="close")
    ap.add_argument("--time_col", default="timestamp")
    ap.add_argument("--window", type=int, default=800)
    ap.add_argument("--k", type=float, default=2.0)
    ap.add_argument("--stop_k", type=float, default=1.0)
    ap.add_argument("--allow_short", action="store_true")
    ap.add_argument("--fee_bps", type=float, default=10.0)
    ap.add_argument("--slippage_bps", type=float, default=0.0)
    ap.add_argument("--tz", dest="tz_localize", default=None)
    ap.add_argument("--resample", default=None, help='Pandas freq like "5T" or "15T"')
    ap.add_argument("--z_entry", type=float, default=1.25)
    ap.add_argument("--slope_min", type=float, default=0.0)
    ap.add_argument("--min_hold", type=int, default=3)
    ap.add_argument("--cooldown", type=int, default=10)
    ap.add_argument("--summary_out", default="batch_summary.csv")
    return ap.parse_args()


def main():
    args = parse_args()
    paths: List[str] = sorted(glob.glob(args.glob)) if args.glob else args.csv
    # skip previously generated perf files
    paths = [p for p in paths if "_perf" not in os.path.basename(p)]
    if not paths:
        print("No CSVs found.", file=sys.stderr)
        return 2

    summaries = []
    for pth in paths:
        try:
            cfg = Config(
                csv_path=pth,
                price_col=args.price_col,
                time_col=args.time_col,
                window=args.window,
                k=args.k,
                stop_k=args.stop_k,
                allow_short=args.allow_short,
                fee_bps=args.fee_bps,
                slippage_bps=args.slippage_bps,
                tz_localize=args.tz_localize,
                resample=args.resample,
                z_entry=args.z_entry,
                slope_min=args.slope_min,
                min_hold=args.min_hold,
                cooldown=args.cooldown,
            )
            df = load_data(cfg)
            sig = generate_signals(df, cfg)
            perf = backtest(sig, cfg)
            sm = summarize(perf, cfg.time_col).to_dict()
            sm["file"] = os.path.basename(pth)
            sm["symbol_hint"] = os.path.basename(pth).split("_")[0]
            summaries.append(sm)

            out_path = pth.rsplit(".", 1)[0] + f"_lr_k{cfg.k}_w{cfg.window}_z{cfg.z_entry}_perf.csv"
            perf.to_csv(out_path, index=False)
            print(f"OK: {pth} -> {out_path}  rows={len(perf)}")
        except Exception as e:
            print(f"ERROR: {pth}: {e}", file=sys.stderr)

    if summaries:
        df_sum = pd.DataFrame(summaries)
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
        df_sum = df_sum.loc[:, [c for c in cols if c in df_sum.columns]]
        df_sum.sort_values("sharpe", ascending=False, inplace=True)
        df_sum.to_csv(args.summary_out, index=False)
        print(f"\nSaved summary: {args.summary_out}")
        print(df_sum.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
