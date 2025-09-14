#!/usr/bin/env python3
# ensure_brackets.py
# -----------------------------------------------------------------------------
# Purpose:
#   For each configured USDT-margined perpetual symbol, check if there is an
#   OPEN futures position on Binance. If yes and there are NO existing
#   reduce-only exit orders, automatically place a TAKE-PROFIT (maker LIMIT GTX)
#   and a STOP-LOSS (STOP_MARKET), both reduce-only, sized to fully close the
#   current position.
#
#   This is a safety net to ensure every open position has bracket exits.
#
# Requirements:
#   - pip install ccxt python-dotenv
#   - .env file with:
#       BINANCE_KEY=
#       BINANCE_SECRET=
#
# Notes:
#   - Uses Binance USDⓈ-M Futures via CCXT with defaultType="future".
#   - TP price is placed at +PCT_TP over entry (rounded to tick) and forced to
#     be above current ask by at least several ticks so it posts as MAKER with GTX.
#   - SL trigger is placed at -PCT_SL below entry as a STOP_MARKET reduce-only.
#   - Both orders include reduceOnly=true to prevent net position flips.
#   - Script is idempotent per run: if any open reduce-only order exists, skip.
#   - All prices are rounded to the exchange tick size; quantities use amount
#     precision from CCXT.
# -----------------------------------------------------------------------------

import math, ccxt
from dotenv import dotenv_values

# Symbols to guard. Use CCXT perp format "<BASE>/USDT:USDT".
SYMBOLS = ["ETH/USDT:USDT", "XRP/USDT:USDT"]

# Bracket distances relative to entry price (e.g., 0.006 = 0.6%)
PCT_TP  = 0.006   # take-profit distance
PCT_SL  = 0.006   # stop-loss distance


def round_to_tick(px: float, tick: float, up: bool) -> float:
    """
    Round a price to the exchange tick grid.

    Args:
        px   : raw price
        tick : tick size (PRICE_FILTER.tickSize)
        up   : True -> round up (ceil to next tick); False -> round down (floor)

    Returns:
        Price snapped to tick grid as float.
    """
    q = math.ceil(px / tick) if up else math.floor(px / tick)
    return q * tick


def main():
    # Load API credentials from .env
    cfg = dotenv_values(".env")

    # Initialize CCXT for Binance USDⓈ-M Futures
    ex = ccxt.binance({
        "apiKey": cfg["BINANCE_KEY"],
        "secret": cfg["BINANCE_SECRET"],
        "enableRateLimit": True,                 # respect exchange rate limits
        "options": {"defaultType": "future"},    # USDⓈ-M futures
    })
    ex.load_markets()  # fetch market metadata including filters and precisions

    for sym in SYMBOLS:
        try:
            # --- Market filters and tick size ---
            m = ex.market(sym)
            # Binance embeds filter objects per symbol in raw 'info'
            filters = {f["filterType"]: f for f in m["info"]["filters"]}
            tick = float(filters["PRICE_FILTER"]["tickSize"])

            # --- Read current position for this symbol ---
            # fetch_positions returns a list; take first item if present
            pos = ex.fetch_positions([sym]) or []
            contracts = float(pos[0].get("contracts") or 0) if pos else 0.0
            if contracts == 0:
                # No open position -> nothing to bracket
                print(f"[{sym}] flat: nothing to bracket.")
                continue

            entry = float(pos[0].get("entryPrice") or 0.0)
            # Use exchange precision to format absolute position size
            qty = float(ex.amount_to_precision(sym, abs(contracts)))

            # --- Avoid duplicate brackets: skip if any open reduce-only order exists ---
            open_orders = ex.fetch_open_orders(sym)
            has_reduce = any(
                (o.get("reduceOnly") or o.get("info", {}).get("reduceOnly"))
                and (o.get("status", "open") == "open")
                for o in open_orders
            )
            if has_reduce:
                print(f"[{sym}] reduce-only orders already present → skip.")
                continue

            # --- Current book snapshot for maker placement logic ---
            tkr = ex.fetch_ticker(sym)
            last = tkr["last"]
            bid = tkr["bid"] or last
            ask = tkr["ask"] or last

            # Determine side from position sign:
            #   long  -> need sell TP and sell SL
            #   short -> need buy TP and buy SL
            is_long = contracts > 0
            tp_side = "sell" if is_long else "buy"
            sl_side = tp_side  # both exits close the same direction

            # =========================
            # Take-Profit (LIMIT GTX)
            # =========================
            # Place TP beyond current quote so it *posts* as maker (GTX rejects taker).
            if is_long:
                # Long TP above ask
                tp_px_raw = entry * (1 + PCT_TP)
                # Nudge above ask by several ticks to avoid immediate execution
                tp_anchor = ask + 6 * tick
                tp_price = round_to_tick(max(tp_px_raw, tp_anchor), tick, up=True)
            else:
                # Short TP below bid
                tp_px_raw = entry * (1 - PCT_TP)
                tp_anchor = bid - 6 * tick
                tp_price = round_to_tick(min(tp_px_raw, tp_anchor), tick, up=False)

            # Apply price precision formatting
            tp_price = float(ex.price_to_precision(sym, tp_price))

            tp = ex.create_order(
                sym,
                "limit",                # LIMIT order
                tp_side,                # 'sell' for long, 'buy' for short
                qty,
                tp_price,               # limit price
                params={
                    "timeInForce": "GTX",   # Good-Till-Crossing: reject if would cross (ensures maker)
                    "reduceOnly": True,     # close position only
                    "recvWindow": 5000,     # optional Binance recv window
                },
            )
            print(f"[{sym}] TP placed id={tp.get('id')} side={tp_side} @ {tp_price} qty={qty}")

            # =========================
            # Stop-Loss (STOP_MARKET)
            # =========================
            # Trigger is relative to entry. Use STOP_MARKET reduce-only for certainty of exit.
            if is_long:
                sl_px_raw = entry * (1 - PCT_SL)
                sl_price = round_to_tick(sl_px_raw, tick, up=False)  # round down for long SL
            else:
                sl_px_raw = entry * (1 + PCT_SL)
                sl_price = round_to_tick(sl_px_raw, tick, up=True)   # round up for short SL

            sl_price = float(ex.price_to_precision(sym, sl_price))

            sl = ex.create_order(
                sym,
                "STOP_MARKET",          # Binance futures stop market
                sl_side,                # same close side as TP
                qty,
                None,                   # no limit price for STOP_MARKET
                params={
                    "stopPrice": sl_price,  # trigger price
                    "reduceOnly": True,
                    "recvWindow": 5000,
                },
            )
            print(f"[{sym}] SL placed id={sl.get('id')} side={sl_side} @ {sl_price} qty={qty}")

        except Exception as e:
            # Log and proceed to the next symbol to avoid halting the entire pass
            print(f"[{sym}] ERROR:", type(e).__name__, e)


if __name__ == "__main__":
    main()
