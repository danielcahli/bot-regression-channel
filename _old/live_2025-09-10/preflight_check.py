#!/usr/bin/env python3
# live/preflight_check.py
#
# Purpose:
#   Sanity-check EVERYTHING before real money:
#     - API keys present and endpoint reachable
#     - Symbol filters (minQty/step)
#     - Klines pull works for the chosen interval
#     - Feature pipeline builds and has warmup
#     - Model and config load; last-bar inference runs
#     - Filters (ATR% and EMA trend) pass/fail
#     - Sizing from notional -> qty is valid
#     - Slippage vs last close is within guard
#
# Usage:
#   .venv/bin/python -m live.preflight_check \
#     --symbol DOGEUSDT --interval 30m \
#     --model outputs/models/DOGEUSDT/model_*.json \
#     --config outputs/models/DOGEUSDT/model_config_*.json \
#     --notional 5 --max_slippage_pct 0.7 --testnet
#
# Notes:
#   - Install deps: pip install xgboost pandas numpy binance-connector
#   - Expects BINANCE_KEY and BINANCE_SECRET in environment.

from __future__ import annotations
import os, sys, glob, argparse, json, math
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from binance.um_futures import UMFutures

# Use the same feature builder as backtests to avoid train/live skew.
from backtests.xgb.xgb_backtest import build_features  # type: ignore

def log(x:str)->None: print(x, flush=True)

def pick_latest(path_glob: str) -> str:
    """Pick the lexicographically last file for model/config glob inputs."""
    files = sorted(glob.glob(path_glob))
    if not files:
        raise SystemExit(f"No files match: {path_glob}")
    return files[-1]

def load_model(path:str)->xgb.Booster:
    b = xgb.Booster(); b.load_model(path); return b

def load_cfg(path:str)->Dict[str,Any]:
    return json.load(open(path, "r"))

def client(testnet: bool)->UMFutures:
    k, s = os.environ.get("BINANCE_KEY"), os.environ.get("BINANCE_SECRET")
    if not k or not s:
        raise SystemExit("Set BINANCE_KEY and BINANCE_SECRET")
    base = "https://testnet.binancefuture.com" if testnet else None
    return UMFutures(key=k, secret=s, base_url=base)

def filters(c:UMFutures, symbol:str)->dict:
    info = c.exchange_info()
    sym = next(s for s in info["symbols"] if s["symbol"] == symbol)
    lot = next(f for f in sym["filters"] if f["filterType"] == "LOT_SIZE")
    return {"qty_step": float(lot["stepSize"]), "min_qty": float(lot["minQty"])}

def klines(c:UMFutures, symbol:str, interval:str, limit:int=1200)->pd.DataFrame:
    raw = c.klines(symbol=symbol, interval=interval, limit=limit)
    cols = ["open_time","open","high","low","close","volume","close_time","q","n","tb","tq","ig"]
    df = pd.DataFrame(raw, columns=cols)
    for k in ["open","high","low","close","volume"]:
        df[k] = pd.to_numeric(df[k], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df[["timestamp","open","high","low","close","volume"]].dropna().set_index("timestamp").sort_index()
    return df

def last_price(c:UMFutures, symbol:str)->float:
    return float(c.ticker_price(symbol=symbol)["price"])

def round_step(x: float, step: float) -> float:
    return math.floor(x/step)*step if step>0 else x

def notional_to_qty(notional: float, price: float, step: float, min_qty: float) -> float:
    if price <= 0: return 0.0
    q = round_step(notional/price, step)
    return q if q >= min_qty else 0.0

def decide_reg(cfg:dict, feats:pd.DataFrame, booster:xgb.Booster, feat_cols:list[str]) -> dict:
    # Require warmup for EMA/ATR
    if len(feats) < 210:
        return {"warmup_needed": True}
    # Volatility gate
    atr_thr = cfg["filters"].get("atrp_threshold")
    vol_ok = True if atr_thr is None else bool(feats["atrp_14"].iloc[-1] >= float(atr_thr))
    # Trend gate
    long_ok = short_ok = True
    if cfg["filters"].get("trend_filter","none").lower() == "ema200":
        span = int(cfg["filters"].get("ema_span",200))
        ema = feats["close"].ewm(span=span, adjust=False).mean()
        long_ok  = bool((feats["close"].iloc[-1] > ema.iloc[-1]) and (ema.diff().iloc[-1] > 0))
        short_ok = bool((feats["close"].iloc[-1] < ema.iloc[-1]) and (ema.diff().iloc[-1] < 0))
    # Inference on last closed bar
    X = feats.iloc[[-1]][feat_cols]
    yhat = float(booster.predict(xgb.DMatrix(X))[0])
    thrL = float(cfg["thresholds"]["long"]); thrS = float(cfg["thresholds"]["short"])
    mult = float(cfg["thresholds"].get("reg_exit_mult", 0.35))
    # Entry check
    action = "hold"
    if vol_ok and long_ok and yhat >= thrL: action = "long_enter"
    if vol_ok and short_ok and yhat <= thrS: action = "short_enter"
    # Exit hysteresis
    long_exit = bool(yhat <= thrL*mult)
    short_exit = bool(yhat >= thrS*mult)
    return {
        "warmup_needed": False,
        "yhat": yhat, "thrL": thrL, "thrS": thrS, "reg_exit_mult": mult,
        "vol_ok": vol_ok, "long_ok": long_ok, "short_ok": short_ok,
        "action": action, "long_exit": long_exit, "short_exit": short_exit,
    }

def main()->None:
    ap = argparse.ArgumentParser(description="Preflight for live XGB bot.")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--interval", required=True)
    ap.add_argument("--model", required=True, help="Path or glob to model_*.json")
    ap.add_argument("--config", required=True, help="Path or glob to model_config_*.json")
    ap.add_argument("--notional", type=float, default=5.0)
    ap.add_argument("--max_slippage_pct", type=float, default=0.7)
    ap.add_argument("--testnet", action="store_true")
    args = ap.parse_args()

    mpath = pick_latest(args.model); cpath = pick_latest(args.config)
    log(f"model:  {mpath}")
    log(f"config: {cpath}")

    booster = load_model(mpath)
    cfg = load_cfg(cpath)
    feat_cols = list(cfg["feature_cols"])

    c = client(testnet=args.testnet)
    flt = filters(c, args.symbol)
    log(f"filters: qty_step={flt['qty_step']} min_qty={flt['min_qty']}")

    df = klines(c, args.symbol, args.interval, limit=1200)
    if df.empty: raise SystemExit("no klines")
    log(f"klines: rows={len(df)} start={df.index[0]} end={df.index[-1]}")

    feats = build_features(df).dropna()
    log(f"features rows={len(feats)} cols={len(feats.columns)} (needs >=210 for warmup)")

    dec = decide_reg(cfg, feats, booster, feat_cols)
    if dec.get("warmup_needed", False):
        print("warmup_needed=True; collect more bars before going live")
        sys.exit(1)

    last_close = float(df["close"].iloc[-1])
    price = last_price(c, args.symbol)
    slip_pct = abs(price/last_close - 1.0) * 100.0
    qty = notional_to_qty(args.notional, price, flt["qty_step"], flt["min_qty"])

    print("\n=== PREFLIGHT SUMMARY ===")
    print(f"symbol={args.symbol} interval={args.interval}")
    print(f"yhat={dec['yhat']:.6f} thrL={dec['thrL']:.6f} thrS={dec['thrS']:.6f}")
    print(f"vol_ok={dec['vol_ok']} long_ok={dec['long_ok']} short_ok={dec['short_ok']}")
    print(f"action_now={dec['action']} long_exit={dec['long_exit']} short_exit={dec['short_exit']}")
    print(f"last_close={last_close:.8f} price_now={price:.8f} slippage_pct={slip_pct:.3f}% (max {args.max_slippage_pct}%)")
    print(f"notional={args.notional} -> qty={qty}")
    if slip_pct > args.max_slippage_pct:
        print("WARNING: slippage exceeds max; bot will skip entries until it narrows.")

if __name__ == "__main__":
    main()
