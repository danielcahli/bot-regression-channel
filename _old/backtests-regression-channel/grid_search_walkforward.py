#!/usr/bin/env python3
# grid_search_walkforward.py
# Usage example:
#   ./grid_search_walkforward.py --glob "data/perp_1m_data/*_1m_14d.csv" \
#     --resample "5T" --split 0.7 --windows 400 600 800 --ks 1.5 2.0 \
#     --z_entries 0.75 1.0 --stop_ks 1.0 --allow_short --fee_bps 12

from __future__ import annotations
import sys, argparse, glob, os, itertools
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# ---------- Strategy primitives (same core as your batch script) ----------
@dataclass
class Config:
    price_col: str = "close"
    time_col: str = "timestamp"
    window: int = 800
    k: float = 2.0
    stop_k: float = 1.0
    allow_short: bool = False
    fee_bps: float = 10.0
    slippage_bps: float = 0.0
    tz_localize: str | None = None
    resample: str | None = None
    z_entry: float = 1.25
    slope_min: float = 0.0
    min_hold: int = 3
    cooldown: int = 10
    sigma_eps: float = 1e-12

def load_data(csv_path: str, cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if cfg.time_col not in df.columns: raise ValueError(f"Missing time column: {cfg.time_col}")
    if cfg.price_col not in df.columns: raise ValueError(f"Missing price column: {cfg.price_col}")
    ts = pd.to_datetime(df[cfg.time_col], utc=True, errors="coerce")
    if cfg.tz_localize: ts = ts.dt.tz_convert(cfg.tz_localize)
    df[cfg.time_col] = ts
    df = df.dropna(subset=[cfg.time_col]).sort_values(cfg.time_col).reset_index(drop=True)
    if cfg.resample: df = _resample_df(df, cfg)
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

def rolling_regression_channel(price: pd.Series, window: int, k: float
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    n = len(price); t_all = np.arange(n, dtype=float)
    mid = np.full(n, np.nan); up = np.full(n, np.nan); lo = np.full(n, np.nan)
    slope = np.full(n, np.nan); bw = np.full(n, np.nan)
    lr = LinearRegression()
    for i in range(window, n):
        t_win = t_all[i-window:i].reshape(-1,1); y_win = price.iloc[i-window:i].values
        lr.fit(t_win, y_win)
        y_hat_now = float(lr.intercept_) + float(lr.coef_[0]) * t_all[i]
        resid = y_win - lr.predict(t_win); sigma = np.std(resid, ddof=1)
        band = k * sigma
        mid[i]=y_hat_now; up[i]=y_hat_now+band; lo[i]=y_hat_now-band; slope[i]=float(lr.coef_[0]); bw[i]=band
    idx = price.index
    return (pd.Series(mid, idx, name="mid"),
            pd.Series(up, idx, name="up"),
            pd.Series(lo, idx, name="lo"),
            pd.Series(slope, idx, name="slope"),
            pd.Series(bw, idx, name="band_width"))

def generate_signals(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    price = pd.to_numeric(df[cfg.price_col], errors="coerce").astype(float)
    mid, up, lo, slope, band_width = rolling_regression_channel(price, cfg.window, cfg.k)
    out = df.copy()
    out["mid"], out["up"], out["lo"], out["slope"], out["band_width"] = mid, up, lo, slope, band_width
    pos = np.zeros(len(out), dtype=int); in_long=False; in_short=False; hold=0; cooldown=0
    p=price.values; m=mid.values; u=up.values; l=lo.values; s=slope.values; bw=band_width.values
    for i in range(1, len(out)):
        pos[i]=pos[i-1]; hold=max(0,hold-1); cooldown=max(0,cooldown-1)
        sigma = bw[i]/max(cfg.k,1e-9)
        if not np.isfinite(sigma) or sigma<cfg.sigma_eps: continue
        z_prev = (p[i-1]-m[i-1])/max(sigma,cfg.sigma_eps)
        if in_long and hold==0:
            stop_level = l[i] - cfg.stop_k*bw[i] if np.isfinite(l[i]) and np.isfinite(bw[i]) else -np.inf
            if np.isfinite(m[i]) and p[i] >= m[i]: in_long=False; pos[i]=0; cooldown=cfg.cooldown
            elif p[i] <= stop_level: in_long=False; pos[i]=0; cooldown=cfg.cooldown
        if in_short and hold==0:
            stop_level = u[i] + cfg.stop_k*bw[i] if np.isfinite(u[i]) and np.isfinite(bw[i]) else np.inf
            if np.isfinite(m[i]) and p[i] <= m[i]: in_short=False; pos[i]=0; cooldown=cfg.cooldown
            elif p[i] >= stop_level: in_short=False; pos[i]=0; cooldown=cfg.cooldown
        if not in_long and not in_short and cooldown==0:
            cross_long = (np.isfinite(l[i-1]) and np.isfinite(l[i]) and p[i-1] <= l[i-1] and p[i] > l[i])
            if cross_long and s[i] >= cfg.slope_min and z_prev <= -cfg.z_entry:
                in_long=True; pos[i]=1; hold=cfg.min_hold
            elif cfg.allow_short:
                cross_short = (np.isfinite(u[i-1]) and np.isfinite(u[i]) and p[i-1] >= u[i-1] and p[i] < u[i])
                if cross_short and s[i] <= -cfg.slope_min and z_prev >= cfg.z_entry:
                    in_short=True; pos[i]=-1; hold=cfg.min_hold
    out["pos"]=pos; return out

def _bars_per_year(perf: pd.DataFrame, time_col: str) -> float:
    ts = perf[time_col]
    if not pd.api.types.is_datetime64_any_dtype(ts): return 252*6.5*60
    dt = ts.diff().median()
    if pd.isna(dt) or dt==pd.Timedelta(0): return 252*6.5*60
    return (365.0*24*3600)/(dt/pd.Timedelta(seconds=1))

def backtest(signals: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = signals.copy()
    price = pd.to_numeric(df[cfg.price_col], errors="coerce").astype(float)
    pos = df["pos"].fillna(0).astype(int)
    ret = price.pct_change().fillna(0.0)
    strat_ret = pos.shift(1).fillna(0) * ret
    per_side = (cfg.fee_bps + cfg.slippage_bps)/1e4
    cost = (pos - pos.shift(1)).abs().fillna(0) * per_side
    net_ret = strat_ret - cost
    equity = (1.0 + net_ret).cumprod()
    out = df[[cfg.time_col,cfg.price_col,"mid","up","lo","slope","band_width","pos"]].copy()
    out["ret_mkt"]=ret; out["ret_gross"]=strat_ret; out["cost"]=cost; out["ret_net"]=net_ret; out["equity"]=equity
    return out

def summarize(perf: pd.DataFrame, time_col: str) -> Dict[str, float]:
    net = perf["ret_net"].fillna(0.0); eq = perf["equity"].ffill()
    n = _bars_per_year(perf, time_col)
    avg=float(net.mean()); vol=float(net.std(ddof=1))
    sharpe = np.sqrt(n)*(avg/vol) if vol>0 else np.nan
    cumret = float(eq.iloc[-1]-1.0) if len(eq) else np.nan
    roll_max = eq.cummax(); drawdown = eq/roll_max - 1.0
    max_dd = float(drawdown.min()) if len(drawdown) else np.nan
    trades = int((perf["pos"].diff().abs()>0).sum())
    return dict(cum_return=cumret, sharpe=sharpe, max_drawdown=max_dd,
                volatility=float(vol*np.sqrt(n)), avg_bar_return=avg,
                bars_per_year=float(n), bars=int(len(perf)), trades=trades)

# ---------- Grid search + walk-forward ----------
def parse_args():
    ap = argparse.ArgumentParser(description="Walk-forward grid search for regression-channel strategy.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--glob", help='Glob for CSVs, e.g. "data/perp_1m_data/*_1m_14d.csv"')
    g.add_argument("--csv", nargs="+", help="List of CSV paths")
    ap.add_argument("--price_col", default="close")
    ap.add_argument("--time_col", default="timestamp")
    ap.add_argument("--resample", default="5T")
    ap.add_argument("--split", type=float, default=0.7, help="Train fraction (0-1)")
    ap.add_argument("--windows", type=int, nargs="+", default=[400,600,800])
    ap.add_argument("--ks", type=float, nargs="+", default=[1.5,2.0])
    ap.add_argument("--z_entries", type=float, nargs="+", default=[0.75,1.0])
    ap.add_argument("--stop_ks", type=float, nargs="+", default=[1.0])
    ap.add_argument("--allow_short", action="store_true")
    ap.add_argument("--fee_bps", type=float, default=12.0)
    ap.add_argument("--slippage_bps", type=float, default=0.0)
    ap.add_argument("--tz", dest="tz_localize", default=None)
    ap.add_argument("--min_trades", type=int, default=8, help="Min trades on train to accept a combo")
    ap.add_argument("--summary_out", default="grid_summary.csv")
    ap.add_argument("--dump_perf", action="store_true", help="Save OOS perf CSV per symbol with chosen params")
    return ap.parse_args()

def split_train_test(df: pd.DataFrame, frac: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if frac <= 0 or frac >= 1: raise ValueError("--split must be in (0,1)")
    n = len(df); cut = max(1, int(n * frac))
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)

def run_combo(df: pd.DataFrame, base_cfg: Config) -> Tuple[pd.DataFrame, Dict[str,float]]:
    sig = generate_signals(df, base_cfg)
    perf = backtest(sig, base_cfg)
    stats = summarize(perf, base_cfg.time_col)
    return perf, stats

def main():
    args = parse_args()
    paths = sorted(glob.glob(args.glob)) if args.glob else args.csv
    paths = [p for p in paths if "_perf" not in os.path.basename(p)]
    if not paths: print("No CSVs found.", file=sys.stderr); return 2

    param_grid = list(itertools.product(args.windows, args.ks, args.z_entries, args.stop_ks))
    rows = []
    best_rows = []

    for pth in paths:
        symbol_hint = os.path.basename(pth).split("_")[0]
        # Load full
        base_cfg_common = Config(price_col=args.price_col, time_col=args.time_col,
                                 tz_localize=args.tz_localize, resample=args.resample,
                                 allow_short=args.allow_short, fee_bps=args.fee_bps,
                                 slippage_bps=args.slippage_bps)
        try:
            df_full = load_data(pth, base_cfg_common)
        except Exception as e:
            print(f"ERROR load {pth}: {e}", file=sys.stderr)
            continue
        if len(df_full) < max(args.windows) + 50:
            print(f"SKIP {pth}: too few bars ({len(df_full)})", file=sys.stderr)
            continue

        # Split
        df_train, df_test = split_train_test(df_full, args.split)

        # Evaluate all combos on train
        train_results = []
        for window, k, z_ent, stop_k in param_grid:
            cfg = Config(price_col=args.price_col, time_col=args.time_col, tz_localize=args.tz_localize,
                         resample=args.resample, allow_short=args.allow_short, fee_bps=args.fee_bps,
                         slippage_bps=args.slippage_bps, window=window, k=k, z_entry=z_ent, stop_k=stop_k)
            try:
                perf_tr, stats_tr = run_combo(df_train, cfg)
            except Exception as e:
                print(f"ERROR train {symbol_hint} {window,k,z_ent,stop_k}: {e}", file=sys.stderr)
                continue
            trades_tr = stats_tr.get("trades", 0)
            train_results.append((stats_tr.get("sharpe", np.nan), trades_tr, cfg))

            rows.append(dict(file=os.path.basename(pth), symbol_hint=symbol_hint, phase="train",
                             window=window, k=k, z_entry=z_ent, stop_k=stop_k, allow_short=args.allow_short,
                             **stats_tr))

        # Pick best by Sharpe subject to min_trades
        train_results.sort(key=lambda x: (-(x[0] if np.isfinite(x[0]) else -1e9), ), reverse=False)
        chosen_cfg = None
        for sh, tr, cfg in sorted(train_results, key=lambda x: (x[0] if np.isfinite(x[0]) else -1e9), reverse=True):
            if tr >= args.min_trades:
                chosen_cfg = cfg
                break
        if chosen_cfg is None:
            # fallback to max trades
            chosen_cfg = max(train_results, key=lambda x: x[1])[2]

        # Test with chosen cfg
        try:
            perf_te, stats_te = run_combo(df_test, chosen_cfg)
        except Exception as e:
            print(f"ERROR test {symbol_hint}: {e}", file=sys.stderr)
            continue

        rows.append(dict(file=os.path.basename(pth), symbol_hint=symbol_hint, phase="test",
                         window=chosen_cfg.window, k=chosen_cfg.k, z_entry=chosen_cfg.z_entry,
                         stop_k=chosen_cfg.stop_k, allow_short=chosen_cfg.allow_short, **stats_te))

        best_rows.append(dict(file=os.path.basename(pth), symbol_hint=symbol_hint,
                              best_window=chosen_cfg.window, best_k=chosen_cfg.k,
                              best_z_entry=chosen_cfg.z_entry, best_stop_k=chosen_cfg.stop_k,
                              allow_short=chosen_cfg.allow_short,
                              test_cum_return=stats_te["cum_return"], test_sharpe=stats_te["sharpe"],
                              test_max_drawdown=stats_te["max_drawdown"],
                              test_trades=stats_te["trades"]))

        if args.dump_perf:
            out_path = pth.rsplit(".",1)[0] + f"_TEST_best_w{chosen_cfg.window}_k{chosen_cfg.k}_z{chosen_cfg.z_entry}_perf.csv"
            perf_te.to_csv(out_path, index=False)
            print(f"{symbol_hint}: best {chosen_cfg.window}/{chosen_cfg.k}/{chosen_cfg.z_entry}/{chosen_cfg.stop_k}  -> {out_path}  rows={len(perf_te)}")

    # Save summaries
    if rows:
        df_all = pd.DataFrame(rows)
        df_all.to_csv(args.summary_out.replace(".csv", "_full.csv"), index=False)
    if best_rows:
        df_best = pd.DataFrame(best_rows).sort_values("test_sharpe", ascending=False)
        df_best.to_csv(args.summary_out, index=False)
        print(f"\nSaved: {args.summary_out}")
        print(df_best.to_string(index=False))
    return 0

if __name__ == "__main__":
    sys.exit(main())
