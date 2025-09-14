#!/usr/bin/env python3
"""
Download OHLCV candles for Binance USDT-margined perpetual futures using CCXT.

Key features:
- Accepts symbols in multiple formats and normalizes to CCXT perp format (e.g., "ETH" -> "ETH/USDT:USDT").
- Can auto-discover the top N most-liquid USDT linear swaps by quoted volume.
- Robust paginated OHLCV fetching from a start time ("since") up to "until" with basic backoff on transient errors.
- Saves one CSV per symbol with UTC timestamps.

Usage examples:
  # Top 10 most liquid perps, last 60 days, 1m timeframe
  ./script.py

  # Specific symbols in mixed formats, last 180 days, 5m timeframe
  ./script.py --symbols "BTCUSDT, ETH/USDT, SOL" --days 180 --timeframe 5m

  # Skip files already downloaded
  ./script.py --skip_if_exists
"""
import argparse, time, random
from datetime import datetime, timezone
from pathlib import Path
from typing import List
import pandas as pd
import ccxt
from ccxt.base.errors import DDoSProtection, NetworkError, ExchangeError


def ms_now() -> int:
    """
    Return current UNIX time in milliseconds in UTC.

    Rationale:
    - CCXT OHLCV uses milliseconds since epoch.
    - Keeping times in UTC avoids tz confusion across systems.
    """
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)


def to_perp_symbols(spec: str) -> List[str]:
    """
    Normalize a comma-separated list of user-provided symbols to CCXT's
    USDT-perp format "<BASE>/USDT:USDT".

    Accepted inputs per token:
      - Already normalized perp, e.g. "ETH/USDT:USDT"  -> kept as is
      - Spot style, e.g. "ETH/USDT"                   -> "ETH/USDT:USDT"
      - Compact, e.g. "ETHUSDT"                       -> "ETH/USDT:USDT"
      - Bare base, e.g. "ETH"                          -> "ETH/USDT:USDT"

    Raises:
      ValueError: if a spot pair with non-USDT quote is given (e.g., "ETH/BTC").

    Notes:
      - Only linear USDT swaps are targeted here to simplify downstream logic.
    """
    out = []
    for x in (s.strip() for s in spec.split(",") if s.strip()):
        u = x.upper()

        # Already in CCXT perp format, e.g. "ETH/USDT:USDT"
        if "/" in u and u.endswith(":USDT"):
            out.append(u)
            continue

        # Spot-style pair provided; ensure quote is USDT and convert to perp
        if "/" in u:
            base, quote = u.split("/", 1)
            if quote != "USDT":
                raise ValueError(f"Only USDT quote supported: {x}")
            out.append(f"{base}/USDT:USDT")
            continue

        # Compact like "ETHUSDT" -> split base and attach ":USDT"
        if u.endswith("USDT") and len(u) > 4:
            base = u[:-4]
            out.append(f"{base}/USDT:USDT")
            continue

        # Bare base like "ETH" -> assume USDT perp
        out.append(f"{u}/USDT:USDT")
    return out


def discover_symbols(ex, top_n: int) -> List[str]:
    """
    Discover the top N most-liquid USDT-quoted, linear swap markets on the exchange.

    Inputs:
      ex     : a ccxt exchange instance (configured for futures)
      top_n  : number of symbols to return

    Process:
      1) load_markets() to populate ex.markets.
      2) Filter for markets with:
         - swap=True (perpetuals)
         - linear=True (USDT-margined)
         - quote == "USDT"
      3) fetch_tickers() for candidates and rank by 'quoteVolume' from raw 'info'.

    Output:
      List[str]: sorted symbols like ["BTC/USDT:USDT", "ETH/USDT:USDT", ...]

    Caveats:
      - 'quoteVolume' comes from exchange payload. Types and presence vary.
      - If 'quoteVolume' missing or unparsable, treat as 0 to avoid crashes.
    """
    ex.load_markets()
    syms = [
        m["symbol"]
        for m in ex.markets.values()
        if m.get("swap") and m.get("linear") and m.get("quote") == "USDT"
    ]

    # Bulk query to reduce HTTP requests
    tickers = ex.fetch_tickers(syms)

    def qv(s: str) -> float:
        info = tickers.get(s, {}).get("info", {})
        try:
            return float(info.get("quoteVolume") or 0.0)
        except Exception:
            return 0.0

    ranked = sorted(((s, qv(s)) for s in syms), key=lambda x: x[1], reverse=True)
    return [s for s, _ in ranked[:top_n]]


def fetch_ohlcv_all(
    ex,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int,
    limit: int,
    base_sleep_ms: int | None = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV rows for [since_ms, until_ms) in pages and return as a DataFrame.

    Inputs:
      ex            : ccxt exchange instance
      symbol        : e.g. "ETH/USDT:USDT"
      timeframe     : e.g. "1m", "5m", "1h"
      since_ms      : inclusive start in ms epoch
      until_ms      : exclusive end in ms epoch
      limit         : max candles per API call (exchange-dependent)
      base_sleep_ms : inter-request sleep; defaults to ex.rateLimit with floor 50ms

    Behavior:
      - Iteratively calls fetch_ohlcv() with 'since' and 'limit'.
      - Advances 't' by one bar size to avoid repeating last candle, even if
        the exchange returns exact boundary candles.
      - Applies simple backoff for DDoSProtection and short wait for NetworkError.
      - Lets ExchangeError bubble up (user action required).

    Returns:
      pd.DataFrame with columns: ["timestamp","open","high","low","close","volume"]
      - 'timestamp' converted to pandas UTC datetime.
      - Empty DataFrame if nothing returned.

    Notes:
      - Some exchanges cap how far back 'since' can be. This function will stop
        if an empty batch is returned.
      - 'limit' should respect the exchange's max for the chosen timeframe.
    """
    if base_sleep_ms is None:
        base_sleep_ms = getattr(ex, "rateLimit", 200)

    rows, t = [], since_ms
    one_bar_ms = ex.parse_timeframe(timeframe) * 1000  # duration of one candle
    backoff = 1.0  # multiplicative backoff seconds on DDoSProtection

    while t < until_ms:
        try:
            # Fetch a page of candles starting at 't'
            batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=t, limit=limit)
            if not batch:
                # No more data available from the server for this range
                break

            rows.extend(batch)

            # Last candle time from this batch
            last_t = batch[-1][0]

            # Move 't' to at least one full bar after both our request 't' and the last candle time.
            # This prevents duplicates when exchanges include the boundary candle.
            t = max(t + one_bar_ms, last_t + one_bar_ms)

            # Respect rate limits. Also impose a minimum of 50ms to be courteous.
            time.sleep(max(base_sleep_ms, 50) / 1000.0)

            # Reset backoff after a successful call
            backoff = 1.0

        except DDoSProtection:
            # Exchange indicates rate/traffic pressure. Back off with jitter.
            wait = min(90.0, backoff)
            time.sleep(wait + random.random())
            backoff *= 1.8  # exponential growth up to ~90s cap
        except NetworkError:
            # Transient network issue. Short fixed wait and retry.
            time.sleep(3.0)
        except ExchangeError:
            # Likely a hard error (bad symbol/timeframe/etc.). Re-raise to caller.
            raise

    if not rows:
        # Return a typed empty frame to keep downstream code simple
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    # Build DataFrame and convert timestamps to timezone-aware UTC datetimes
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def main():
    """
    Parse CLI arguments, resolve symbols, compute time range, fetch data, and save CSVs.

    Files are named like: "<BASE>USDT:USDT_<timeframe>_<days>d.csv"
      Example: "ETHUSDT:USDT_1m_60d.csv"
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60, help="Lookback window in days.")
    ap.add_argument("--timeframe", default="1m", help="CCXT timeframe, e.g. 1m, 5m, 1h.")
    ap.add_argument("--top", type=int, default=10, help="Top N symbols by quoted volume when auto-discovering.")
    ap.add_argument(
        "--symbols",
        default="",
        help="Comma-separated symbols. Accepts ETHUSDT, ETH/USDT, ETH, or ETH/USDT:USDT.",
    )
    ap.add_argument("--out", default="./data", help="Output directory for CSV files.")
    ap.add_argument(
        "--limit",
        type=int,
        default=1500,
        help="Max candles per API call. Respect exchange caps for the timeframe.",
    )
    ap.add_argument(
        "--skip_if_exists",
        action="store_true",
        help="If set, do not re-download when the target CSV already exists.",
    )
    args = ap.parse_args()

    # Ensure output directory exists
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Binance futures (USDT-margined) via CCXT.
    # - options.defaultType="future" targets USDⓈ-M perpetuals.
    ex = ccxt.binance({"options": {"defaultType": "future"}})
    ex.enableRateLimit = True  # CCXT will throttle requests to ex.rateLimit automatically

    # Resolve symbols: use provided list or auto-discover by liquidity.
    if args.symbols.strip():
        symbols = to_perp_symbols(args.symbols)
        print("Symbols:", symbols)
    else:
        symbols = discover_symbols(ex, args.top)
        print("Top symbols:", symbols)

    # Compute time window [since, until). 'until' is now.
    until = ms_now()
    since = until - args.days * 24 * 60 * 60 * 1000

    # Download each symbol and save to CSV
    for sym in symbols:
        # Sanitize filename: remove '/' to keep it filesystem-friendly
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
