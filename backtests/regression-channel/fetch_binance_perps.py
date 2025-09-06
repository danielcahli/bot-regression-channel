#!/usr/bin/env python3
import argparse, time, random
from datetime import datetime, timezone
from pathlib import Path
from typing import List
import pandas as pd
import ccxt
from ccxt.base.errors import DDoSProtection, NetworkError, ExchangeError

def ms_now() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)

def to_perp_symbols(spec: str) -> List[str]:
    out = []
    for x in (s.strip() for s in spec.split(",") if s.strip()):
        u = x.upper()
        # already CCXT perp format
        if "/" in u and u.endswith(":USDT"):
            out.append(u)
            continue
        # spot-style => add :USDT
        if "/" in u:
            base, quote = u.split("/", 1)
            if quote != "USDT":
                raise ValueError(f"Only USDT quote supported: {x}")
            out.append(f"{base}/USDT:USDT")
            continue
        # compact like ETHUSDT
        if u.endswith("USDT") and len(u) > 4:
            base = u[:-4]
            out.append(f"{base}/USDT:USDT")
            continue
        # bare base like ETH
        out.append(f"{u}/USDT:USDT")
    return out

def discover_symbols(ex, top_n: int) -> List[str]:
    ex.load_markets()
    syms = [m["symbol"] for m in ex.markets.values()
            if m.get("swap") and m.get("linear") and m.get("quote") == "USDT"]
    tickers = ex.fetch_tickers(syms)
    def qv(s):
        info = tickers.get(s, {}).get("info", {})
        try: return float(info.get("quoteVolume") or 0.0)
        except Exception: return 0.0
    ranked = sorted(((s, qv(s)) for s in syms), key=lambda x: x[1], reverse=True)
    return [s for s, _ in ranked[:top_n]]

def fetch_ohlcv_all(ex, symbol: str, timeframe: str, since_ms: int, until_ms: int,
                    limit: int, base_sleep_ms: int | None = None) -> pd.DataFrame:
    if base_sleep_ms is None:
        base_sleep_ms = getattr(ex, "rateLimit", 200)
    rows, t = [], since_ms
    one_bar_ms = ex.parse_timeframe(timeframe) * 1000
    backoff = 1.0
    while t < until_ms:
        try:
            batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=t, limit=limit)
            if not batch:
                break
            rows.extend(batch)
            last_t = batch[-1][0]
            t = max(t + one_bar_ms, last_t + one_bar_ms)
            time.sleep(max(base_sleep_ms, 50) / 1000.0)
            backoff = 1.0
        except DDoSProtection:
            wait = min(90.0, backoff)
            time.sleep(wait + random.random())
            backoff *= 1.8
        except NetworkError:
            time.sleep(3.0)
        except ExchangeError:
            raise
    if not rows:
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
    df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--timeframe", default="1m")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--symbols", default="", help="ETHUSDT,ETH/USDT,ETH/USDT:USDT")
    ap.add_argument("--out", default="./data")
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--skip_if_exists", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    ex = ccxt.binance({"options": {"defaultType": "future"}}); ex.enableRateLimit = True

    if args.symbols.strip():
        symbols = to_perp_symbols(args.symbols)
        print("Symbols:", symbols)
    else:
        symbols = discover_symbols(ex, args.top)
        print("Top symbols:", symbols)

    until = ms_now()
    since = until - args.days * 24 * 60 * 60 * 1000

    for sym in symbols:
        tag = sym.replace("/", "")
        fname = out_dir / f"{tag}_{args.timeframe}_{args.days}d.csv"
        if args.skip_if_exists and fname.exists():
            print(f"Skip (exists): {fname}")
            continue
        print(f"Downloading {sym} {args.timeframe} for {args.days}d ...")
        df = fetch_ohlcv_all(ex, sym, args.timeframe, since, until, args.limit)
        df.to_csv(fname, index=False)
        print(f"  Saved: {fname}  rows={len(df)}")

if __name__ == "__main__":
    main()
