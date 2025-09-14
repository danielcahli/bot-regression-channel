#!/usr/bin/env python3
# Combine multiple *_perf.csv into a single equal- or custom-weight portfolio.
# The script reads several CSVs that each contain time-aligned returns for a strategy,
# merges them on a timestamp column, computes a portfolio return per bar with either
# equal weights or user-supplied weights (re-normalized when some series are NaN),
# outputs a portfolio performance CSV, and prints summary stats.

from __future__ import annotations
import sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd


def parse_args():
    """
    Parse CLI arguments.

    --csv        : list of input CSVs (each file must have time_col and ret_col)
    --time_col   : name of the timestamp column in input CSVs (default: 'timestamp')
    --ret_col    : name of the per-bar return column in input CSVs (default: 'ret_net')
    --weights    : optional list of portfolio weights matching the order of --csv files.
                   If omitted, equal weights are used.
    --out_perf   : output CSV path for the combined portfolio performance.

    Returns:
        argparse.Namespace with parsed fields.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True, help="List of *_perf.csv files")
    ap.add_argument("--time_col", default="timestamp")
    ap.add_argument("--ret_col", default="ret_net")
    ap.add_argument("--weights", type=float, nargs="*", default=None, help="Optional weights matching --csv")
    ap.add_argument("--out_perf", default="portfolio_perf.csv")
    return ap.parse_args()


def bars_per_year(ts: pd.Series) -> float:
    """
    Estimate number of bars per year from a datetime Series.
    - If ts is not datetime-like, return a default for 1-minute bars in US equities hours:
      252 trading days * 6.5 hours/day * 60 minutes/hour ≈ 98280 bars/year.
    - Otherwise compute the median bar spacing and invert to annualized bars.

    Args:
        ts: pandas Series of timestamps

    Returns:
        Estimated bars per year as float.
    """
    # Not datetime -> fallback constant for 1-minute equity bars
    if not pd.api.types.is_datetime64_any_dtype(ts):
        return 252 * 6.5 * 60

    # Median spacing between consecutive timestamps
    dt = ts.diff().median()
    if pd.isna(dt) or dt == pd.Timedelta(0):
        return 252 * 6.5 * 60

    # Convert spacing to seconds and compute how many per year
    return (365.0 * 24 * 3600) / (dt / pd.Timedelta(seconds=1))


def main():
    """
    Orchestrate:
      1) Read inputs and validate required columns.
      2) Keep only timestamp + return, rename return column to file's basename to make columns unique.
      3) Outer-merge all frames on the timestamp to align across different symbols/strategies.
      4) Build weight vector: user-provided or equal. Normalize to sum to 1.
      5) Handle missing data per row: re-normalize weights to the subset of available returns.
      6) Compute portfolio return and equity curve.
      7) Compute basic stats: cumulative return, annualized Sharpe, max drawdown.
      8) Save CSV and print stats.
    """
    a = parse_args()

    # 1) Load each CSV, ensure required columns exist, coerce timestamp to UTC, and keep only needed cols.
    frames = []
    for p in a.csv:
        df = pd.read_csv(p)
        if a.time_col not in df.columns or a.ret_col not in df.columns:
            print(f"ERROR: {p} missing {a.time_col} or {a.ret_col}", file=sys.stderr)
            return 2
        # Convert to timezone-aware UTC; drop rows with invalid timestamps
        df[a.time_col] = pd.to_datetime(df[a.time_col], utc=True, errors="coerce")
        df = (
            df.dropna(subset=[a.time_col])[[a.time_col, a.ret_col]]
              # Rename return column to the file's name so each series has a unique label
              .rename(columns={a.ret_col: Path(p).name})
        )
        frames.append(df)

    # 2) Merge all dataframes on timestamp using outer join to keep every timestamp seen in any series
    out = frames[0]
    for df in frames[1:]:
        out = out.merge(df, on=a.time_col, how="outer")
    # Chronological order
    out = out.sort_values(a.time_col).reset_index(drop=True)

    # 3) Extract the return matrix (rows = bars, cols = strategies)
    rets = out.drop(columns=[a.time_col]).to_numpy(dtype=float)
    n = rets.shape[1]

    # 4) Portfolio weights: user-specified or equal weights
    w = np.array(a.weights, dtype=float) if a.weights is not None else np.ones(n) / n
    if len(w) != n:
        print("ERROR: weights length must match number of CSVs", file=sys.stderr)
        return 2
    # Normalize weights to sum to 1 for safety
    w = w / w.sum()

    # 5) Per-row availability mask: True where strategy return is present, False when NaN
    mask = ~np.isnan(rets)  # shape (T, n)

    # Build a weight matrix where missing returns receive weight 0
    w_mat = np.where(mask, w, 0.0)  # shape (T, n)

    # Row-wise sum of available weights; if all missing, set sum to 1 to avoid division by zero
    row_sum = w_mat.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0

    # Re-normalize weights per row so that the weights across available series sum to 1
    w_norm = w_mat / row_sum

    # 6) Portfolio return = sum over strategies of (return_i * weight_i) ignoring NaNs
    port_ret = np.nansum(rets * w_norm, axis=1)

    # Build performance DataFrame with equity curve (cumulative product of 1 + returns)
    perf = pd.DataFrame({a.time_col: out[a.time_col], "ret_port": port_ret})
    perf["equity"] = (1.0 + perf["ret_port"].fillna(0.0)).cumprod()

    # 7) Basic statistics
    n_py = bars_per_year(perf[a.time_col])   # bars per year for annualization
    avg = perf["ret_port"].mean()            # mean per-bar return
    vol = perf["ret_port"].std(ddof=1)       # std dev per-bar return
    sharpe = np.sqrt(n_py) * (avg / vol) if vol > 0 else np.nan  # simple ann. Sharpe, rf=0
    cumret = float(perf["equity"].iloc[-1] - 1.0)                # total return over the sample
    # Max drawdown computed from equity vs its running max
    max_dd = float((perf["equity"] / perf["equity"].cummax() - 1.0).min())

    # 8) Persist and report
    Path(a.out_perf).parent.mkdir(parents=True, exist_ok=True)
    perf.to_csv(a.out_perf, index=False)
    print(f"Saved: {a.out_perf}")
    print(f"cum_return={cumret:.6f}  sharpe={sharpe:.3f}  max_drawdown={max_dd:.6f}")
    return 0


if __name__ == "__main__":
    # Run main and propagate exit code to shell
    sys.exit(main())
