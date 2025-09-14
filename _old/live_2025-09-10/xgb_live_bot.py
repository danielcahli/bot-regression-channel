#!/usr/bin/env python3
# live/xgb_live_bot.py
# Minimal Binance USDT-M futures loop. Dry-run by default.

from __future__ import annotations
import os, json, time, math, argparse, signal
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from binance.um_futures import UMFutures

from backtests.xgb.xgb_backtest import build_features  # reuse same features

RUN = True
def _sig(_s,_f):  # graceful stop
    global RUN; RUN = False
signal.signal(signal.SIGINT, _sig); signal.signal(signal.SIGTERM, _sig)

def log(m:str)->None: print(time.strftime("[%Y-%m-%d %H:%M:%S]"), m, flush=True)

def load_model(p:str)->xgb.Booster:
    b = xgb.Booster(); b.load_model(p); return b

def fetch_klines(c:UMFutures, symbol:str, interval:str, limit:int=1000)->pd.DataFrame:
    raw = c.klines(symbol=symbol, interval=interval, limit=limit)
    cols = ["open_time","open","high","low","close","volume","close_time","q","n","tb","tq","ig"]
    df = pd.DataFrame(raw, columns=cols)
    for k in ["open","high","low","close","volume"]:
        df[k] = pd.to_numeric(df[k], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df[["timestamp","open","high","low","close","volume"]].dropna().set_index("timestamp").sort_index()
    return df

def exchange_filters(c:UMFutures, symbol:str)->dict:
    info = c.exchange_info()
    sym = next(s for s in info["symbols"] if s["symbol"] == symbol)
    lot = next(f for f in sym["filters"] if f["filterType"] == "LOT_SIZE")
    return {"qty_step": float(lot["stepSize"]), "min_qty": float(lot["minQty"])}

def round_step(x:float, step:float)->float:
    return math.floor(x/step)*step if step>0 else x

def notional_to_qty(notional:float, price:float, step:float, min_qty:float)->float:
    q = round_step(notional/max(price,1e-8), step)
    return q if q>=min_qty else 0.0

def decide_reg(cfg:dict, feats:pd.DataFrame, booster:xgb.Booster, feat_cols:list[str])->dict:
    if len(feats)<210: return {"action":"hold","why":"warmup"}
    # filters
    atr_thr = cfg["filters"].get("atrp_threshold")
    vol_ok = bool(feats["atrp_14"].iloc[-1] >= float(atr_thr)) if atr_thr is not None else True
    long_ok = short_ok = True
    if cfg["filters"].get("trend_filter","none").lower()=="ema200":
        span = int(cfg["filters"].get("ema_span",200))
        ema = feats["close"].ewm(span=span, adjust=False).mean()
        long_ok  = bool((feats["close"].iloc[-1]>ema.iloc[-1]) and (ema.diff().iloc[-1]>0))
        short_ok = bool((feats["close"].iloc[-1]<ema.iloc[-1]) and (ema.diff().iloc[-1]<0))
    # predict
    yhat = float(booster.predict(xgb.DMatrix(feats.iloc[[-1]][feat_cols]))[0])
    thrL = float(cfg["thresholds"]["long"]); thrS = float(cfg["thresholds"]["short"])
    mult = float(cfg["thresholds"].get("reg_exit_mult",0.35))
    dec = {"action":"hold","yhat":yhat,"vol_ok":vol_ok,"long_ok":long_ok,"short_ok":short_ok,"thrL":thrL,"thrS":thrS}
    if not vol_ok: return dec
    if long_ok and yhat>=thrL:  dec["action"]="long_enter";  return dec
    if short_ok and yhat<=thrS: dec["action"]="short_enter"; return dec
    dec["long_exit"]  = bool(yhat <= thrL*mult)
    dec["short_exit"] = bool(yhat >= thrS*mult)
    return dec

def place(c:UMFutures, symbol:str, side:str, qty:float, reduce:bool, live:bool)->dict:
    if qty<=0: return {"status":"skip","reason":"qty<=0"}
    if not live:
        log(f"[DRY] {side} {qty} {symbol} reduceOnly={reduce}"); return {"status":"dry"}
    try:
        r = c.new_order(symbol=symbol, side=side, type="MARKET", quantity=str(qty), reduceOnly=reduce, recvWindow=5000)
        log(f"[LIVE] ok {side} {qty}"); return {"status":"ok","resp":r}
    except Exception as e:
        log(f"[LIVE] err {e}"); return {"status":"error","error":str(e)}

def main()->None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="DOGEUSDT")
    ap.add_argument("--interval", default="30m")  # must match resample
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--notional", type=float, default=100.0)
    ap.add_argument("--poll_sec", type=int, default=30)
    ap.add_argument("--testnet", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--state_path", default="outputs/live_state.json")
    args = ap.parse_args()

    booster = load_model(args.model)
    cfg = json.load(open(args.config))
    feat_cols = list(cfg["feature_cols"])

    key = os.environ.get("BINANCE_KEY"); sec = os.environ.get("BINANCE_SECRET")
    base = "https://testnet.binancefuture.com" if args.testnet else None
    client = UMFutures(key=key, secret=sec, base_url=base)

    os.makedirs(os.path.dirname(args.state_path), exist_ok=True)
    state = {"pos":0,"qty":0.0}
    if os.path.exists(args.state_path):
        try: state.update(json.load(open(args.state_path)))
        except Exception: pass

    f = exchange_filters(client, args.symbol)
    last_ts: Optional[pd.Timestamp] = None
    log(f"start symbol={args.symbol} interval={args.interval} live={args.live} testnet={args.testnet}")

    while RUN:
        try:
            bars = fetch_klines(client, args.symbol, args.interval, limit=1200)
        except Exception as e:
            log(f"klines err {e}"); time.sleep(args.poll_sec); continue
        ts = bars.index[-1] if not bars.empty else None
        if ts is None or ts==last_ts:
            time.sleep(args.poll_sec); continue
        last_ts = ts

        feats = build_features(bars).dropna()
        if feats.empty: time.sleep(args.poll_sec); continue
        dec = decide_reg(cfg, feats, booster, feat_cols)

        # price and qty
        price = float(client.ticker_price(symbol=args.symbol)["price"])
        qty = notional_to_qty(args.notional, price, f["qty_step"], f["min_qty"])

        if state["pos"]==0:
            if dec["action"]=="long_enter":
                r=place(client,args.symbol,"BUY",qty,False,args.live); 
                if r["status"] in ("ok","dry"): state={"pos":1,"qty":qty}
            elif dec["action"]=="short_enter":
                r=place(client,args.symbol,"SELL",qty,False,args.live);
                if r["status"] in ("ok","dry"): state={"pos":-1,"qty":qty}
        elif state["pos"]==1:
            if dec.get("long_exit",False) or dec["action"]=="short_enter":
                r=place(client,args.symbol,"SELL",state["qty"],True,args.live);
                if r["status"] in ("ok","dry"): state={"pos":0,"qty":0.0}
        elif state["pos"]==-1:
            if dec.get("short_exit",False) or dec["action"]=="long_enter":
                r=place(client,args.symbol,"BUY",state["qty"],True,args.live);
                if r["status"] in ("ok","dry"): state={"pos":0,"qty":0.0}

        try: json.dump(state, open(args.state_path,"w"), indent=2)
        except Exception as e: log(f"state save err {e}")

        log(f"{args.symbol} {ts} action={dec['action']} pos={state['pos']} yhat={dec.get('yhat')}")
        time.sleep(args.poll_sec)

if __name__=="__main__":
    main()
