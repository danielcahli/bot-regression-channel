#!/usr/bin/env python3
"""
Fetch OHLCV for Binance USDⓈ-M perpetuals and write CSVs.

- Symbols can be auto-discovered by quoted volume (top N) or provided explicitly.
- Output one CSV per symbol with UTC timestamps.
"""

import argparse, time, random
from datetime import datetime, timezone
from pathlib import Path
from typing import List
import pandas as pd
import ccxt
from ccxt.base.errors import DDoSProtection, NetworkError, ExchangeError

def ms_now() -> int:
    """UTC now in ms."""
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)

def to_perp_symbols(spec: str) -> List[str]:
    """
    Normalize user input to CCXT perp format "<BASE>/USDT:USDT".
    Accepts: ETH, ETHUSDT, ETH/USDT, ETH/USDT:USDT
    """
    out = []
    for x in (s.strip() for s in spec.split(",") if s.strip()):
        u = x.upper()
        if "/" in u and u.endswith(":USDT"):
            out.append(u); continue
        if "/" in u:
            base, quote = u.split("/", 1)
            if quote != "USDT":
                raise ValueError(f"Only USDT quote supported: {x}")
            out.append(f"{base}/USDT:USDT"); continue
        if u.endswith("USDT") and len(u) > 4:
            out.append(f"{u[:-4]}/USDT:USDT"); continue
        out.append(f"{u}/USDT:USDT")
    return out

def discover_symbols(ex, top_n: int) -> List[str]:
    """Return top-N USDT linear swaps by quoted volume."""
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
    """
    Paged OHLCV fetch [since_ms, until_ms). Resilient to transient errors.
    """
    if base_sleep_ms is None:
        base_sleep_ms = getattr(ex, "rateLimit", 200)
    rows, t = [], since_ms
    one_bar_ms = ex.parse_timeframe(timeframe) * 1000
    backoff = 1.0
    while t < until_ms:
        try:
            batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=t, limit=limit)
            if not batch: break
            rows.extend(batch)
            last_t = batch[-1][0]
            t = max(t + one_bar_ms, last_t + one_bar_ms)
            time.sleep(max(base_sleep_ms, 50) / 1000.0)
            backoff = 1.0
        except DDoSProtection:
            wait = min(90.0, backoff)
            time.sleep(wait + random.random()); backoff *= 1.8
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
    ap.add_argument("--days", type=int, default=60, help="Lookback window in days")
    ap.add_argument("--timeframe", default="1m", help="CCXT timeframe, e.g. 1m, 5m, 1h")
    ap.add_argument("--top", type=int, default=10, help="Top-N by quoted volume when auto-discovering")
    ap.add_argument("--symbols", default="", help="ETHUSDT,ETH/USDT,ETH/USDT:USDT")
    ap.add_argument("--out", default="./data/perp_1m_data", help="Output directory")
    ap.add_argument("--limit", type=int, default=1500, help="Max candles per call")
    ap.add_argument("--skip_if_exists", action="store_true", help="Skip if target CSV exists")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    ex = ccxt.binance({"options": {"defaultType": "future"}}); ex.enableRateLimit = True

    # Resolve target symbols
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
            print(f"Skip (exists): {fname}"); continue
        print(f"Downloading {sym} {args.timeframe} for {args.days}d ...")
        df = fetch_ohlcv_all(ex, sym, args.timeframe, since, until, args.limit)
        df.to_csv(fname, index=False)
        print(f"  Saved: {fname}  rows={len(df)}")

if __name__ == "__main__":
    main()
