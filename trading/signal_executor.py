#!/usr/bin/env python3
"""
trading/signal_executor.py

Run a regression-channel trading signal on 5-minute bars for Binance USDⓈ-M
perpetuals and optionally place orders via CCXT.

Safety model:
- SHORTS ARE DISABLED BY DEFAULT in live trading.
- To allow shorts, you must pass --enable_shorts OR set ENABLE_SHORTS=1 in .env.
- If a short position exists while shorts are disabled, the bot will FLATTEN it
  with a reduce-only maker order (safety net).

Strategy (trend breakout, symmetric rules):
- Compute a linear regression channel on the last `window` 5m closes.
- Enter long  if: close > upper and slope > 0
- Enter short if: close < lower and slope < 0      (only if shorts enabled)
- Long stop  : close < center - stop_k * sigma  → exit to flat
- Short stop : close > center + stop_k * sigma  → exit to flat

Order placement:
- Maker LIMIT with timeInForce=GTX to avoid taking liquidity.
- Buy maker price: a few ticks BELOW bid.
- Sell maker price: a few ticks ABOVE ask.
- Exits use reduceOnly=True to prevent flip.

Requirements:
  pip install ccxt python-dotenv numpy pandas
  .env with BINANCE_KEY, BINANCE_SECRET; optionally ENABLE_SHORTS=1

Example:
  python3 trading/signal_executor.py \
    --symbols ETH/USDT:USDT,XRP/USDT:USDT \
    --window 800 --k 2.0 --stop_k 1.0 \
    --qty_usdt 25 --minutes 4320            # no shorts unless --enable_shorts
"""

import argparse
import math
import sys
import time
import numpy as np
import pandas as pd
import ccxt
from dotenv import dotenv_values

# Use explicit alias. "5T" is deprecated in some stacks.
RESAMPLE = "5min"


# ============================== Math / Bands ==============================

def last_regression_band(closes: pd.Series, window: int, k: float) -> dict:
    """
    Fit a line y = a + b*x on the LAST `window` closes, shifting x so the most
    recent bar is x=0. This stabilizes the intercept as the channel "center".

    Returns:
        dict(center, upper, lower, sigma, slope)
          center = intercept a
          slope  = b
          sigma  = std of residuals with ddof=1
          upper/lower = center ± k*sigma
    """
    if len(closes) < window:
        raise ValueError(f"Not enough closes for window={window} (have {len(closes)})")

    y = closes.iloc[-window:].to_numpy(dtype=float)
    x = np.arange(window, dtype=float)
    x -= x[-1]  # last bar at x=0 for numerical stability

    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    sigma = float(np.std(resid, ddof=1))

    return {
        "center": float(intercept),
        "upper": float(intercept + k * sigma),
        "lower": float(intercept - k * sigma),
        "sigma": sigma,
        "slope": float(slope),
    }


# ============================== Data Fetch ===============================

def load_1m_from_ccxt(ex, symbol: str, minutes: int) -> pd.DataFrame:
    """
    Pull ~`minutes` of 1m futures candles via CCXT with simple paging.
    Returns a DataFrame indexed by UTC timestamp.
    """
    since = int(time.time() * 1000) - minutes * 60_000
    tf = "1m"
    limit = 1000
    rows = []

    while True:
        chunk = ex.fetch_ohlcv(symbol, timeframe=tf, since=since, limit=limit)
        if not chunk:
            break
        rows += chunk
        # If the exchange gave fewer than limit, we've reached the end window.
        if len(chunk) < limit:
            break
        # Advance to just after the last candle timestamp to avoid duplication.
        since = chunk[-1][0] + 60_000
        # Stop early once we have enough rows.
        if len(rows) >= minutes:
            break
        time.sleep(ex.rateLimit / 1000)

    if not rows:
        raise RuntimeError(f"No OHLCV from exchange for {symbol}")

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp").sort_index()


def resample_to_5m(df1m: pd.DataFrame) -> pd.DataFrame:
    """
    Right-labeled, right-closed resample to 5-minute bars with standard OHLCV agg.
    Prevents lookahead by closing windows at the right edge.
    """
    return df1m.resample(RESAMPLE, label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()


# ============================== Exchange Utils ===========================

def market_filters(ex, symbol: str):
    """
    Extract key filters from the exchange metadata for `symbol`:
      - PRICE_FILTER.tickSize
      - LOT_SIZE.stepSize
      - MIN_NOTIONAL.notional  (fallback to 20.0 if missing)
    """
    ex.load_markets()
    f = {x["filterType"]: x for x in ex.market(symbol)["info"]["filters"]}
    tick = float(f["PRICE_FILTER"]["tickSize"])
    step = float(f["LOT_SIZE"]["stepSize"])
    min_notional = float(f.get("MIN_NOTIONAL", {}).get("notional", 20.0))
    return tick, step, min_notional


def maker_px_qty(ex, symbol: str, side: str, qty_usdt: float):
    """
    Compute a MAKER limit price a few ticks away from the book and a valid
    quantity derived from target notional while satisfying filters.

    side="buy"  → price below bid  (posts as maker)
    side="sell" → price above ask  (posts as maker)

    Returns:
        (price: float, qty: float) already formatted with exchange precision.
    """
    tick, step, min_not = market_filters(ex, symbol)
    t = ex.fetch_ticker(symbol)
    bid = t.get("bid") or t.get("last")
    ask = t.get("ask") or t.get("last")

    # Nudge away from top of book
    if side == "buy":
        px = math.floor(((bid or t["last"]) - 2 * tick) / tick) * tick
        if px <= 0:
            px = (bid or t["last"])
    else:
        px = math.ceil(((ask or t["last"]) + 2 * tick) / tick) * tick

    # Honor minimum notional requirement
    notional = max(min_not, float(qty_usdt))
    raw_qty = notional / px
    qty_steps = math.ceil(raw_qty / step)
    qty = qty_steps * step

    # Apply exchange precision
    px = float(ex.price_to_precision(symbol, px))
    qty = float(ex.amount_to_precision(symbol, qty))
    return px, qty


def fetch_net_contracts(ex, symbol: str) -> float:
    """
    Return NET position in contracts for `symbol`:
      > 0 = long, < 0 = short, 0 = flat

    Uses CCXT unified fields when present and falls back to Binance 'info'.
    """
    try:
        positions = ex.fetch_positions([symbol]) or []
    except Exception:
        positions = []

    if not positions:
        return 0.0

    p = positions[0]
    # CCXT unified
    contracts = float(p.get("contracts") or 0.0)
    side = (p.get("side") or "").lower()

    if contracts != 0.0 and side in ("long", "short"):
        return contracts if side == "long" else -abs(contracts)

    # Fallback: Binance raw info.positionAmt (string, can be negative)
    info = p.get("info", {})
    if "positionAmt" in info:
        try:
            amt = float(info["positionAmt"])
            return amt  # sign reflects side for binance futures
        except Exception:
            pass

    # Last resort: assume long if we only know size
    return contracts


def cancel_all_open(ex, symbol: str):
    """
    Best-effort cancel of all working orders for `symbol`. Errors ignored.
    """
    try:
        for o in ex.fetch_open_orders(symbol):
            try:
                ex.cancel_order(o["id"], symbol)
            except Exception:
                pass
    except Exception:
        pass


# ============================== Main ====================================

def main():
    ap = argparse.ArgumentParser(
        description="Regression-channel signal executor (LONG by default; shorts require explicit enable)."
    )
    ap.add_argument("--symbols", required=True,
                    help="Comma-separated CCXT symbols, e.g. ETH/USDT:USDT,XRP/USDT:USDT")
    ap.add_argument("--window", type=int, default=800, help="Regression window on 5m bars")
    ap.add_argument("--k", type=float, default=2.0, help="Band width multiplier")
    ap.add_argument("--stop_k", type=float, default=1.0, help="Stop distance in sigma units from center")
    ap.add_argument("--minutes", type=int, default=48 * 60, help="1m history depth to fetch for resampling")
    ap.add_argument("--qty_usdt", type=float, default=25.0, help="Target order notional (min-notional aware)")
    ap.add_argument("--dry_run", action="store_true", help="Print actions only, do not place/cancel orders")
    # SAFETY: shorts disabled unless explicitly enabled
    ap.add_argument("--enable_shorts", action="store_true",
                    help="Allow short entries. Default OFF. Can also set ENABLE_SHORTS=1 in .env")
    args = ap.parse_args()

    # Load API keys and optional kill switch from .env
    cfg = dotenv_values(".env")
    if not (cfg.get("BINANCE_KEY") and cfg.get("BINANCE_SECRET")):
        print("[FATAL] Put BINANCE_KEY and BINANCE_SECRET in .env", file=sys.stderr)
        sys.exit(2)

    # Shorts permission: CLI flag OR env var
    shorts_enabled = bool(args.enable_shorts) or (cfg.get("ENABLE_SHORTS") == "1")

    # Initialize exchange (USDⓈ-M futures)
    ex = ccxt.binance({
        "apiKey": cfg["BINANCE_KEY"],
        "secret": cfg["BINANCE_SECRET"],
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })

    # Iterate symbols; isolate failures per symbol
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    for sym in symbols:
        try:
            # 1) Data: 1m → 5m
            df1m = load_1m_from_ccxt(ex, sym, minutes=args.minutes)
            ohlc5 = resample_to_5m(df1m)
            if len(ohlc5) < args.window:
                raise ValueError(f"Not enough 5m bars for window={args.window} (have {len(ohlc5)})")

            # 2) Bands on last `window` bars
            band = last_regression_band(ohlc5["close"], args.window, args.k)
            close = float(ohlc5["close"].iloc[-1])

            # Compute stop lines
            long_stop = band["center"] - args.stop_k * band["sigma"]
            short_stop = band["center"] + args.stop_k * band["sigma"]

            # Entry conditions
            long_enter = (close > band["upper"] and band["slope"] > 0)
            short_enter = (close < band["lower"] and band["slope"] < 0)

            # Enforce shorts policy
            if not shorts_enabled:
                short_enter = False  # hard-disable new short entries

            print(f"{sym} | close={close:.6f}  center={band['center']:.6f}  "
                  f"upper={band['upper']:.6f}  lower={band['lower']:.6f}  slope={band['slope']:.8f}")
            print(f"   shorts_enabled={shorts_enabled}  "
                  f"long_stop={long_stop:.6f}  short_stop={short_stop:.6f}")
            print(f"   long_enter={bool(long_enter)}  short_enter={bool(short_enter)}")

            # 3) Position state
            net = fetch_net_contracts(ex, sym)  # >0 long, <0 short, 0 flat
            print(f"   NET contracts: {net:+.6f}")

            # 4) Actions. Priority: exits → entries. No flip in a single step.

            # 4a) Exit long to flat if stop broken
            if net > 0 and close < long_stop:
                _, step, _ = market_filters(ex, sym)
                qty = math.floor(net / step) * step
                qty = float(ex.amount_to_precision(sym, qty))
                px, _ = maker_px_qty(ex, sym, "sell", args.qty_usdt)
                print(f"   EXIT LONG plan: SELL {qty} @ {px} (GTX reduceOnly)")
                if qty > 0 and not args.dry_run:
                    cancel_all_open(ex, sym)
                    o = ex.create_order(
                        sym, "limit", "sell", qty, px,
                        params={"timeInForce": "GTX", "reduceOnly": True, "recvWindow": 5000}
                    )
                    print("   Placed SELL id:", o.get("id"))

            # 4b) Exit short to flat if stop broken OR shorts disabled (safety flatten)
            elif net < 0 and (close > short_stop or not shorts_enabled):
                _, step, _ = market_filters(ex, sym)
                qty = math.floor(abs(net) / step) * step
                qty = float(ex.amount_to_precision(sym, qty))
                px, _ = maker_px_qty(ex, sym, "buy", args.qty_usdt)
                reason = "short stop" if close > short_stop else "shorts disabled → safety flatten"
                print(f"   EXIT SHORT plan: BUY {qty} @ {px} (GTX reduceOnly)  [{reason}]")
                if qty > 0 and not args.dry_run:
                    cancel_all_open(ex, sym)
                    o = ex.create_order(
                        sym, "limit", "buy", qty, px,
                        params={"timeInForce": "GTX", "reduceOnly": True, "recvWindow": 5000}
                    )
                    print("   Placed BUY id:", o.get("id"))

            # 4c) Enter long from flat
            elif net == 0 and long_enter:
                px, qty = maker_px_qty(ex, sym, "buy", args.qty_usdt)
                print(f"   ENTER LONG plan: BUY {qty} @ {px} (GTX)")
                if not args.dry_run:
                    cancel_all_open(ex, sym)
                    o = ex.create_order(
                        sym, "limit", "buy", qty, px,
                        params={"timeInForce": "GTX", "recvWindow": 5000}
                    )
                    print("   Placed BUY id:", o.get("id"))

            # 4d) Enter short from flat (only if enabled)
            elif net == 0 and short_enter and shorts_enabled:
                px, qty = maker_px_qty(ex, sym, "sell", args.qty_usdt)
                print(f"   ENTER SHORT plan: SELL {qty} @ {px} (GTX)")
                if not args.dry_run:
                    cancel_all_open(ex, sym)
                    o = ex.create_order(
                        sym, "limit", "sell", qty, px,
                        params={"timeInForce": "GTX", "recvWindow": 5000}
                    )
                    print("   Placed SELL id:", o.get("id"))

            else:
                print("   HOLD: no action.")

        except ValueError as ve:
            # Common input issues (e.g., insufficient bars)
            print(f"[WARN] {sym}: {ve}")
        except Exception as e:
            # Log unexpected runtime errors; continue to next symbol
            print(f"[ERROR] {sym}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
