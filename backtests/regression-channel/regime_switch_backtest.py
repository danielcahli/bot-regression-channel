#!/usr/bin/env python3
# backtests/regression-channel/regime_switch_backtest.py
from __future__ import annotations
import sys, argparse, glob, os
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# ---------- Config ----------
@dataclass
class Config:
    csv_path: str
    price_col: str = "close"
    time_col: str = "timestamp"
    resample: str | None = None
    tz_localize: str | None = None
    # channel
    window: int = 800
    k: float = 2.0
    # trading
    mode: str = "auto"           # "mr" | "trend" | "auto"
    auto_r2: float = 0.25        # trend if R^2 >= this
    z_entry: float = 0.75
    slope_min: float = 0.0
    stop_k: float = 1.0
    min_hold: int = 3
    cooldown: int = 10
    allow_short: bool = True
    long_only: bool = False
    # costs
    fee_bps: float = 12.0
    slippage_bps: float = 0.0
    # safety
    sigma_eps: float = 1e-12
    # RSI filter (optional)
    rsi_len: int = 14
    rsi_long: float | None = None
    rsi_short: float | None = None

# ---------- IO ----------
def load_data(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.csv_path)
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
    has_ohlc = {"open","high","low","close"}.issubset(df.columns)
    if has_ohlc:
        agg = {"open":"first","high":"max","low":"min","close":"last"}
        if "volume" in df.columns: agg["volume"] = "sum"
        out = df.resample(cfg.resample).agg(agg).dropna(subset=["close"])
    else:
        out = df[[cfg.price_col]].resample(cfg.resample).last().dropna()
    out = out.reset_index()
    out[cfg.time_col] = pd.to_datetime(out[cfg.time_col], utc=True)
    return out.reset_index(drop=True)

# ---------- Indicators ----------
def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    up_ema = up.ewm(alpha=1/length, adjust=False).mean()
    dn_ema = dn.ewm(alpha=1/length, adjust=False).mean()
    rs = up_ema / dn_ema.replace(0.0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)

def rolling_regression_channel(price: pd.Series, window: int, k: float
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    n = len(price); t = np.arange(n, dtype=float)
    mid = np.full(n, np.nan); up = np.full(n, np.nan); lo = np.full(n, np.nan)
    slope = np.full(n, np.nan); bw = np.full(n, np.nan); r2 = np.full(n, np.nan)
    lr = LinearRegression()
    for i in range(window, n):
        t_win = t[i-window:i].reshape(-1,1)
        y_win = price.iloc[i-window:i].values
        lr.fit(t_win, y_win)
        b0 = float(lr.intercept_); b1 = float(lr.coef_[0])
        y_hat = lr.predict(t_win)
        resid = y_win - y_hat
        sse = float(np.sum(resid**2))
        sst = float(np.sum((y_win - y_win.mean())**2))
        r2[i] = 1 - (sse / sst) if sst > 0 else 0.0
        sigma = float(np.std(resid, ddof=1))
        band = k * sigma
        mid[i] = b0 + b1 * t[i]
        up[i]  = mid[i] + band
        lo[i]  = mid[i] - band
        slope[i] = b1
        bw[i] = band
    idx = price.index
    return (pd.Series(mid, idx, name="mid"),
            pd.Series(up, idx, name="up"),
            pd.Series(lo, idx, name="lo"),
            pd.Series(slope, idx, name="slope"),
            pd.Series(bw, idx, name="band_width"),
            pd.Series(r2, idx, name="r2"))

# ---------- Signals ----------
def generate_signals(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    price = pd.to_numeric(df[cfg.price_col], errors="coerce").astype(float)
    mid, up, lo, slope, bw, r2 = rolling_regression_channel(price, cfg.window, cfg.k)
    out = df.copy()
    out["mid"], out["up"], out["lo"], out["slope"], out["band_width"], out["r2"] = mid, up, lo, slope, bw, r2
    rsi_vals = rsi(price, cfg.rsi_len).values

    pos = np.zeros(len(out), dtype=int)
    in_long = False; in_short = False
    hold = 0; cool = 0

    p = price.values; m = mid.values; u = up.values; l = lo.values
    s = slope.values; band = bw.values; r2v = r2.values

    for i in range(1, len(out)):
        pos[i] = pos[i-1]
        hold = max(0, hold - 1)
        cool = max(0, cool - 1)

        sigma = band[i] / max(cfg.k, 1e-9)
        if not np.isfinite(sigma) or sigma < cfg.sigma_eps:
            continue
        z_prev = (p[i-1] - m[i-1]) / max(sigma, cfg.sigma_eps)

        # exits
        if in_long and hold == 0:
            stop_level = m[i] - cfg.stop_k * band[i]
            if np.isfinite(m[i]) and p[i] <= m[i]: in_long=False; pos[i]=0; cool=cfg.cooldown
            elif np.isfinite(stop_level) and p[i] <= stop_level: in_long=False; pos[i]=0; cool=cfg.cooldown
        if in_short and hold == 0:
            stop_level = m[i] + cfg.stop_k * band[i]
            if np.isfinite(m[i]) and p[i] >= m[i]: in_short=False; pos[i]=0; cool=cfg.cooldown
            elif np.isfinite(stop_level) and p[i] >= stop_level: in_short=False; pos[i]=0; cool=cfg.cooldown

        # entries
        if not in_long and not in_short and cool == 0:
            regime = cfg.mode
            if cfg.mode == "auto":
                regime = "trend" if (np.isfinite(r2v[i]) and r2v[i] >= cfg.auto_r2) else "mr"

            r_ok_long  = True if cfg.rsi_long  is None else (rsi_vals[i-1] <= cfg.rsi_long)
            r_ok_short = True if cfg.rsi_short is None else (rsi_vals[i-1] >= cfg.rsi_short)

            if regime == "mr":
                cross_long = np.isfinite(l[i-1]) and np.isfinite(l[i]) and p[i-1] <= l[i-1] and p[i] > l[i]
                if cross_long and s[i] >= cfg.slope_min and z_prev <= -cfg.z_entry and r_ok_long:
                    in_long=True; pos[i]=1; hold=cfg.min_hold
                elif cfg.allow_short and not cfg.long_only:
                    cross_short = np.isfinite(u[i-1]) and np.isfinite(u[i]) and p[i-1] >= u[i-1] and p[i] < u[i]
                    if cross_short and s[i] <= -cfg.slope_min and z_prev >= cfg.z_entry and r_ok_short:
                        in_short=True; pos[i]=-1; hold=cfg.min_hold
            else:
                brk_long = np.isfinite(u[i-1]) and np.isfinite(u[i]) and p[i-1] < u[i-1] and p[i] >= u[i] and s[i] > 0
                if brk_long:
                    in_long=True; pos[i]=1; hold=cfg.min_hold
                elif cfg.allow_short and not cfg.long_only:
                    brk_short = np.isfinite(l[i-1]) and np.isfinite(l[i]) and p[i-1] > l[i-1] and p[i] <= l[i] and s[i] < 0
                    if brk_short:
                        in_short=True; pos[i]=-1; hold=cfg.min_hold

    out["pos"] = pos
    return out

# ---------- Backtest ----------
def _bars_per_year(perf: pd.DataFrame, time_col: str) -> float:
    ts = perf[time_col]
    if not pd.api.types.is_datetime64_any_dtype(ts): return 252*6.5*60
    dt = ts.diff().median()
    if pd.isna(dt) or dt == pd.Timedelta(0): return 252*6.5*60
    return (365.0*24*3600)/(dt/pd.Timedelta(seconds=1))

def backtest(signals: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = signals.copy()
    price = pd.to_numeric(df[cfg.price_col], errors="coerce").astype(float)
    pos = df["pos"].fillna(0).astype(int)
    ret = price.pct_change().fillna(0.0)
    strat = pos.shift(1).fillna(0) * ret
    per_side = (cfg.fee_bps + cfg.slippage_bps)/1e4
    cost = (pos - pos.shift(1)).abs().fillna(0) * per_side
    net = strat - cost
    eq = (1.0 + net).cumprod()
    out = df[[cfg.time_col, cfg.price_col, "mid","up","lo","slope","band_width","r2","pos"]].copy()
    out["ret_mkt"]=ret; out["ret_gross"]=strat; out["cost"]=cost; out["ret_net"]=net; out["equity"]=eq
    return out

def summarize(perf: pd.DataFrame, time_col: str) -> pd.Series:
    net = perf["ret_net"].fillna(0.0); eq = perf["equity"].ffill()
    n = _bars_per_year(perf, time_col)
    avg = float(net.mean()); vol = float(net.std(ddof=1))
    sharpe = np.sqrt(n)*(avg/vol) if vol>0 else np.nan
    cumret = float(eq.iloc[-1]-1.0) if len(eq) else np.nan
    dd = (eq/eq.cummax()-1.0).min() if len(eq) else np.nan
    trades = int((perf["pos"].diff().abs()>0).sum())
    return pd.Series(dict(cum_return=cumret, sharpe=sharpe, max_drawdown=float(dd),
                          volatility=float(vol*np.sqrt(n)), avg_bar_return=avg,
                          bars_per_year=float(n), bars=int(len(perf)), trades=trades))

# ---------- Batch ----------
def parse_args():
    ap = argparse.ArgumentParser(description="Regression-channel with regime switching (MR vs Trend).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--glob")
    g.add_argument("--csv", nargs="+")
    ap.add_argument("--price_col", default="close"); ap.add_argument("--time_col", default="timestamp")
    ap.add_argument("--resample", default="5T"); ap.add_argument("--tz", dest="tz_localize", default=None)

    ap.add_argument("--mode", choices=["mr","trend","auto"], default="auto")
    ap.add_argument("--auto_r2", type=float, default=0.25)
    ap.add_argument("--window", type=int, default=800); ap.add_argument("--k", type=float, default=2.0)
    ap.add_argument("--z_entry", type=float, default=0.75); ap.add_argument("--slope_min", type=float, default=0.0)
    ap.add_argument("--stop_k", type=float, default=1.0)
    ap.add_argument("--min_hold", type=int, default=3); ap.add_argument("--cooldown", type=int, default=10)
    ap.add_argument("--allow_short", action="store_true"); ap.add_argument("--long_only", action="store_true")

    ap.add_argument("--fee_bps", type=float, default=12.0); ap.add_argument("--slippage_bps", type=float, default=0.0)
    ap.add_argument("--rsi_len", type=int, default=14); ap.add_argument("--rsi_long", type=float, default=None)
    ap.add_argument("--rsi_short", type=float, default=None)

    ap.add_argument("--summary_out", default="regime_summary.csv")
    return ap.parse_args()

def main():
    args = parse_args()
    paths = sorted(glob.glob(args.glob)) if args.glob else args.csv
    paths = [p for p in paths if "_perf" not in os.path.basename(p)]
    if not paths: print("No CSVs found.", file=sys.stderr); return 2

    rows = []
    for pth in paths:
        try:
            cfg = Config(csv_path=pth, price_col=args.price_col, time_col=args.time_col,
                         resample=args.resample, tz_localize=args.tz_localize,
                         window=args.window, k=args.k, mode=args.mode, auto_r2=args.auto_r2,
                         z_entry=args.z_entry, slope_min=args.slope_min, stop_k=args.stop_k,
                         min_hold=args.min_hold, cooldown=args.cooldown,
                         allow_short=args.allow_short, long_only=args.long_only,
                         fee_bps=args.fee_bps, slippage_bps=args.slippage_bps,
                         rsi_len=args.rsi_len, rsi_long=args.rsi_long, rsi_short=args.rsi_short)
            df = load_data(cfg)
            sig = generate_signals(df, cfg)
            perf = backtest(sig, cfg)
            stats = summarize(perf, cfg.time_col).to_dict()
            stats["file"] = os.path.basename(pth); stats["symbol_hint"] = os.path.basename(pth).split("_")[0]
            rows.append(stats)
            out_path = pth.rsplit(".",1)[0] + f"_regime_{cfg.mode}_w{cfg.window}_k{cfg.k}_perf.csv"
            perf.to_csv(out_path, index=False)
            print(f"OK: {pth} -> {out_path}  rows={len(perf)}")
        except Exception as e:
            print(f"ERROR: {pth}: {e}", file=sys.stderr)

    if rows:
        df_sum = pd.DataFrame(rows)
        cols = ["file","symbol_hint","cum_return","sharpe","max_drawdown","volatility",
                "avg_bar_return","bars_per_year","bars","trades"]
        df_sum = df_sum[[c for c in cols if c in df_sum.columns]]
        df_sum.sort_values("sharpe", ascending=False, inplace=True)
        df_sum.to_csv(args.summary_out, index=False)
        print(f"\nSaved summary: {args.summary_out}")
        print(df_sum.to_string(index=False))
    return 0

if __name__ == "__main__":
    sys.exit(main())
