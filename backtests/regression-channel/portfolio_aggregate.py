#!/usr/bin/env python3
# Combine multiple *_perf.csv into a single equal- or custom-weight portfolio.

from __future__ import annotations
import sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True, help="List of *_perf.csv files")
    ap.add_argument("--time_col", default="timestamp")
    ap.add_argument("--ret_col", default="ret_net")
    ap.add_argument("--weights", type=float, nargs="*", default=None, help="Optional weights matching --csv")
    ap.add_argument("--out_perf", default="portfolio_perf.csv")
    return ap.parse_args()

def bars_per_year(ts: pd.Series) -> float:
    if not pd.api.types.is_datetime64_any_dtype(ts): return 252*6.5*60
    dt = ts.diff().median()
    if pd.isna(dt) or dt == pd.Timedelta(0): return 252*6.5*60
    return (365.0*24*3600) / (dt / pd.Timedelta(seconds=1))

def main():
    a = parse_args()
    frames = []
    for p in a.csv:
        df = pd.read_csv(p)
        if a.time_col not in df.columns or a.ret_col not in df.columns:
            print(f"ERROR: {p} missing {a.time_col} or {a.ret_col}", file=sys.stderr); return 2
        df[a.time_col] = pd.to_datetime(df[a.time_col], utc=True, errors="coerce")
        df = df.dropna(subset=[a.time_col])[[a.time_col, a.ret_col]].rename(columns={a.ret_col: Path(p).name})
        frames.append(df)

    out = frames[0]
    for df in frames[1:]:
        out = out.merge(df, on=a.time_col, how="outer")
    out = out.sort_values(a.time_col).reset_index(drop=True)

    rets = out.drop(columns=[a.time_col]).to_numpy(dtype=float)
    n = rets.shape[1]
    w = np.array(a.weights, dtype=float) if a.weights is not None else np.ones(n)/n
    if len(w) != n: print("ERROR: weights length must match number of CSVs", file=sys.stderr); return 2
    w = w / w.sum()

    mask = ~np.isnan(rets)
    w_mat = np.where(mask, w, 0.0)
    row_sum = w_mat.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    w_norm = w_mat / row_sum
    port_ret = np.nansum(rets * w_norm, axis=1)

    perf = pd.DataFrame({a.time_col: out[a.time_col], "ret_port": port_ret})
    perf["equity"] = (1.0 + perf["ret_port"].fillna(0.0)).cumprod()

    n_py = bars_per_year(perf[a.time_col])
    avg = perf["ret_port"].mean()
    vol = perf["ret_port"].std(ddof=1)
    sharpe = np.sqrt(n_py) * (avg / vol) if vol > 0 else np.nan
    cumret = float(perf["equity"].iloc[-1] - 1.0)
    max_dd = float((perf["equity"] / perf["equity"].cummax() - 1.0).min())

    Path(a.out_perf).parent.mkdir(parents=True, exist_ok=True)
    perf.to_csv(a.out_perf, index=False)
    print(f"Saved: {a.out_perf}")
    print(f"cum_return={cumret:.6f}  sharpe={sharpe:.3f}  max_drawdown={max_dd:.6f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
