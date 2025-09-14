#!/usr/bin/env python3
"""
Signal executor with LONG + SHORT support.

What it does:
- Pulls recent 1m futures data from Binance via CCXT.
- Resamples to 5-minute bars and builds a regression channel on the last `window` bars.
- Signal rules:
    * Long enter  : close > upper  and slope > 0
    * Short enter : close < lower  and slope < 0
    * Long stop   : close < center - stop_k * sigma  (exit long → flat)
    * Short stop  : close > center + stop_k * sigma  (exit short → flat)
- Places MAKER LIMIT (GTX) orders only:
    * buy price a few ticks BELOW bid
    * sell price a few ticks ABOVE ask
- Exits use reduceOnly to avoid flipping position.

Requirements:
  pip install ccxt python-dotenv numpy pandas

.env (project root):
  BINANCE_KEY=
  BINANCE_SECRET=
"""

import argparse
import math, sys, time
import numpy as np
import pandas as pd
import ccxt
from dotenv import dotenv_values

RESAMPLE = "5min"  # explicit minutes alias


# ----------------------------- Helpers: math / bands ----------------------------- #
def last_regression_band(closes: pd.Series, window: int, k: float) -> dict:
    """
    Fit linear y = a + b*x on the LAST `window` closes with x shifted so x=0 is the most recent bar.
    Returns dict with center=a, upper/lower=a±k*sigma(residuals), sigma, and slope=b.
    """
    y = closes.iloc[-window:].to_numpy(dtype=float)
    x = np.arange(window, dtype=float)
    x -= x[-1]                                  # stabilize intercept at last bar
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    sigma = float(np.std(resid, ddof=1))
    return {
        "center": float(intercept),
        "upper":  float(intercept + k * sigma),
        "lower":  float(intercept - k * sigma),
        "sigma":  sigma,
        "slope":  float(slope),
    }


def load_1m_from_ccxt(ex, symbol: str, minutes: int) -> pd.DataFrame:
    """
    Pull ~`minutes` of 1m OHLCV using paged fetch_ohlcv. Returns dataframe indexed by UTC timestamp.
    """
    since = int(time.time() * 1000) - minutes * 60_000
    tf, limit = "1m", 1000
    rows = []
    while True:
        chunk = ex.fetch_ohlcv(symbol, timeframe=tf, since=since, limit=limit)
        if not chunk:
            break
        rows += chunk
        if len(chunk) < limit:
            break
        since = chunk[-1][0] + 60_000
        if len(rows) >= minutes:
            break
        time.sleep(ex.rateLimit / 1000)
    if not rows:
        raise RuntimeError(f"No OHLCV from exchange for {symbol}")
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp").sort_index()


def resample_5m(df1m: pd.DataFrame) -> pd.DataFrame:
    """
    Right-labeled/right-closed resample to 5-minute bars with standard OHLCV aggregation.
    """
    return df1m.resample(RESAMPLE, label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()


# ----------------------------- Helpers: exchange / orders ----------------------------- #
def market_filters(ex, symbol: str):
    """
    Extract PRICE_FILTER.tickSize, LOT_SIZE.stepSize, and MIN_NOTIONAL if present.
    """
    ex.load_markets()
    f = {x["filterType"]: x for x in ex.market(symbol)["info"]["filters"]}
    tick = float(f["PRICE_FILTER"]["tickSize"])
    step = float(f["LOT_SIZE"]["stepSize"])
    min_notional = float(f.get("MIN_NOTIONAL", {}).get("notional", 20.0))  # fallback if missing
    return tick, step, min_notional


def maker_px_qty(ex, symbol: str, side: str, qty_usdt: float):
    """
    Compute a MAKER price a few ticks away from the book, and a valid quantity.

    side="buy"  → price below bid  (posts, not takes)
    side="sell" → price above ask  (posts, not takes)
    """
    tick, step, min_not = market_filters(ex, symbol)
    t = ex.fetch_ticker(symbol)
    bid = t.get("bid") or t.get("last")
    ask = t.get("ask") or t.get("last")

    if side == "buy":
        px = math.floor((bid - 2 * tick) / tick) * tick
    else:
        px = math.ceil((ask + 2 * tick) / tick) * tick

    notional = max(min_not, qty_usdt)
    raw_qty = notional / px
    qty_steps = math.ceil(raw_qty / step)
    qty = qty_steps * step

    # Apply exchange precision
    px = float(ex.price_to_precision(symbol, px))
    qty = float(ex.amount_to_precision(symbol, qty))
    return px, qty


def fetch_net_contracts(ex, symbol: str) -> float:
    """
    Return NET position in contracts:
      >0 long, <0 short, 0 flat.
    CCXT unified positions usually include 'contracts' and 'side'.
    """
    try:
        pos = ex.fetch_positions([symbol]) or []
    except Exception:
        return 0.0
    if not pos:
        return 0.0
    p0 = pos[0]
    qty = float(p0.get("contracts") or 0.0)
    side = (p0.get("side") or "").lower()
    if qty == 0:
        return 0.0
    if side == "short":
        return -abs(qty)
    return abs(qty)  # assume long if side missing


def cancel_all_open(ex, symbol: str):
    """
    Best-effort cancel of all working orders for the symbol.
    """
    try:
        for o in ex.fetch_open_orders(symbol):
            try:
                ex.cancel_order(o["id"], symbol)
            except Exception:
                pass
    except Exception:
        pass


# ----------------------------- Main logic ----------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True,
                    help="Comma-separated CCXT symbols, e.g. ETH/USDT:USDT,XRP/USDT:USDT")
    ap.add_argument("--window", type=int, default=800)
    ap.add_argument("--k", type=float, default=2.0)
    ap.add_argument("--stop_k", type=float, default=1.0)
    ap.add_argument("--minutes", type=int, default=48 * 60, help="1m history depth to fetch")
    ap.add_argument("--qty_usdt", type=float, default=25.0)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    # Load API keys
    cfg = dotenv_values(".env")
    if not (cfg.get("BINANCE_KEY") and cfg.get("BINANCE_SECRET")):
        print("[FATAL] Put BINANCE_KEY and BINANCE_SECRET in .env", file=sys.stderr)
        sys.exit(2)

    # Exchange (USDⓈ-M futures)
    ex = ccxt.binance({
        "apiKey": cfg["BINANCE_KEY"],
        "secret": cfg["BINANCE_SECRET"],
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })

    for sym in [s.strip() for s in args.symbols.split(",") if s.strip()]:
        try:
            # 1) Data
            df1m = load_1m_from_ccxt(ex, sym, minutes=args.minutes)
            ohlc5 = resample_5m(df1m)
            if len(ohlc5) < args.window:
                raise ValueError(f"Not enough 5m bars for window={args.window} (have {len(ohlc5)})")

            # 2) Bands
            band = last_regression_band(ohlc5["close"], args.window, args.k)
            close = float(ohlc5["close"].iloc[-1])

            long_stop  = band["center"] - args.stop_k * band["sigma"]
            short_stop = band["center"] + args.stop_k * band["sigma"]

            long_enter  = (close > band["upper"] and band["slope"] > 0)
            short_enter = (close < band["lower"] and band["slope"] < 0)

            print(f"{sym} | close={close:.4f}  center={band['center']:.4f}  "
                  f"upper={band['upper']:.4f}  lower={band['lower']:.4f}  slope={band['slope']:.6f}")
            print(f" → long_enter={long_enter}  short_enter={short_enter}  "
                  f"long_stop={long_stop:.4f}  short_stop={short_stop:.4f}")

            # 3) Position
            net = fetch_net_contracts(ex, sym)  # >0 long, <0 short, 0 flat
            print(f"   NET contracts: {net:+.6f}")

            # 4) Actions (priority: exits → entries). No flip in one tick.
            if net > 0 and close < long_stop:
                # Exit LONG to flat
                _, step, _ = market_filters(ex, sym)
                qty = math.floor(net / step) * step
                qty = float(ex.amount_to_precision(sym, qty))
                px, _ = maker_px_qty(ex, sym, "sell", args.qty_usdt)
                print(f"   EXIT LONG plan: SELL {qty} @ {px} (GTX reduceOnly)")
                if qty > 0:
                    if not args.dry_run:
                        cancel_all_open(ex, sym)
                        o = ex.create_order(sym, "limit", "sell", qty, px,
                                            params={"timeInForce": "GTX", "reduceOnly": True, "recvWindow": 5000})
                        print("   Placed SELL id:", o.get("id"))
                else:
                    print("   (nothing to close)")

            elif net < 0 and close > short_stop:
                # Exit SHORT to flat
                _, step, _ = market_filters(ex, sym)
                qty = math.floor(abs(net) / step) * step
                qty = float(ex.amount_to_precision(sym, qty))
                px, _ = maker_px_qty(ex, sym, "buy", args.qty_usdt)
                print(f"   EXIT SHORT plan: BUY {qty} @ {px} (GTX reduceOnly)")
                if qty > 0:
                    if not args.dry_run:
                        cancel_all_open(ex, sym)
                        o = ex.create_order(sym, "limit", "buy", qty, px,
                                            params={"timeInForce": "GTX", "reduceOnly": True, "recvWindow": 5000})
                        print("   Placed BUY id:", o.get("id"))
                else:
                    print("   (nothing to close)")

            elif net == 0 and long_enter:
                # Enter LONG
                px, qty = maker_px_qty(ex, sym, "buy", args.qty_usdt)
                print(f"   ENTER LONG plan: BUY {qty} @ {px} (GTX)")
                if not args.dry_run:
                    cancel_all_open(ex, sym)
                    o = ex.create_order(sym, "limit", "buy", qty, px,
                                        params={"timeInForce": "GTX", "recvWindow": 5000})
                    print("   Placed BUY id:", o.get("id"))

            elif net == 0 and short_enter:
                # Enter SHORT
                px, qty = maker_px_qty(ex, sym, "sell", args.qty_usdt)
                print(f"   ENTER SHORT plan: SELL {qty} @ {px} (GTX)")
                if not args.dry_run:
                    cancel_all_open(ex, sym)
                    o = ex.create_order(sym, "limit", "sell", qty, px,
                                        params={"timeInForce": "GTX", "recvWindow": 5000})
                    print("   Placed SELL id:", o.get("id"))

            else:
                print("   HOLD: no action.")

        except ValueError as ve:
            print(f"[WARN] {sym}: {ve}")
        except Exception as e:
            print(f"[ERROR] {sym}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
