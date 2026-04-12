#!/usr/bin/env python3
"""
Price Action + Volume Profile Futures Backtester — v11

v11 goals:
  - Keep the bot aligned with a small-account setup: $1 fixed margin per trade
  - Improve expectancy without overfitting entry logic
  - Make the report stable and easier to use for monthly-goal validation

Latest default tuning focus:
  1. Keep TP logic from v10 (2.5 ATR, 3.0 ATR for high-score trades)
  2. Tighten base SL to 0.8 ATR
  3. Start the trail earlier at 1.3 ATR
  4. Use a tighter 0.4 ATR trail distance to protect momentum moves
  5. Guarantee at least 0.7 ATR locked profit after trail activation
  6. Default to fixed $1 margin per trade unless disabled
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple


KLINES_URL    = "https://fapi.binance.com/fapi/v1/klines"
MAX_PER_FETCH = 1500


@dataclass
class Candle:
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int


@dataclass
class Trade:
    side: str
    entry_time: int
    exit_time: int
    entry_price: float
    exit_price: float
    quantity: float
    margin: float
    leverage: float
    gross_pnl: float
    fees: float
    net_pnl: float
    result: str
    exit_reason: str
    signal_score: int
    tp_price: float
    sl_price: float
    atr_at_entry: float
    rsi_at_entry: float
    htf_trend: str


def iso_ts(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def candle_hour_utc(c: Candle) -> int:
    return datetime.fromtimestamp(c.open_time / 1000, tz=timezone.utc).hour


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_klines(symbol: str, interval: str, limit: int) -> List[Candle]:
    symbol = symbol.upper()
    chunks: List[List[Candle]] = []
    remaining = limit
    end_time: Optional[int] = None

    while remaining > 0:
        batch = min(remaining, MAX_PER_FETCH)
        params: Dict = {"symbol": symbol, "interval": interval, "limit": batch}
        if end_time is not None:
            params["endTime"] = end_time
        url = f"{KLINES_URL}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=30) as r:
            payload = json.load(r)
        if not payload:
            break
        chunk = [Candle(int(x[0]), float(x[1]), float(x[2]), float(x[3]),
                        float(x[4]), float(x[5]), int(x[6])) for x in payload]
        chunks.insert(0, chunk)
        end_time   = chunk[0].open_time - 1
        remaining -= len(chunk)
        if len(chunk) < batch:
            break

    all_c: List[Candle] = []
    for ch in chunks:
        all_c.extend(ch)

    seen: set = set()
    unique = []
    for c in sorted(all_c, key=lambda x: x.open_time):
        if c.open_time not in seen:
            seen.add(c.open_time)
            unique.append(c)
    return unique[-limit:]


# ── Indicators ────────────────────────────────────────────────────────────────

def fmean(v: Iterable[float]) -> float:
    items = list(v)
    return statistics.fmean(items) if items else 0.0


def ema_of(values: List[float], period: int) -> float:
    if not values or len(values) < period:
        return values[-1] if values else 0.0
    k = 2.0 / (period + 1)
    e = fmean(values[:period])
    for v in values[period:]:
        e = v * k + e * (1 - k)
    return e


def calc_atr(candles: List[Candle], period: int = 14) -> float:
    if len(candles) < 2:
        return 0.0
    trs = []
    for i in range(1, len(candles)):
        pc = candles[i - 1].close
        trs.append(max(
            candles[i].high - candles[i].low,
            abs(candles[i].high - pc),
            abs(candles[i].low  - pc),
        ))
    return fmean(trs[-period:])


def calc_rsi(candles: List[Candle], period: int = 14) -> float:
    if len(candles) < period + 1:
        return 50.0
    closes = [c.close for c in candles[-(period + 1):]]
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    ag = fmean(gains)
    al = fmean(losses)
    if al == 0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + ag / al)


def choppiness_index(candles: List[Candle], period: int = 14) -> float:
    if len(candles) < period + 2:
        return 61.8
    window = candles[-(period + 1):]
    atr_sum = sum(
        max(window[i].high - window[i].low,
            abs(window[i].high - window[i - 1].close),
            abs(window[i].low  - window[i - 1].close))
        for i in range(1, len(window))
    )
    hi = max(c.high for c in window)
    lo = min(c.low  for c in window)
    if math.isclose(hi, lo):
        return 100.0
    return 100.0 * math.log10(atr_sum / (hi - lo)) / math.log10(period)


def build_volume_profile(candles: List[Candle], bins: int = 36
                          ) -> Tuple[float, float, float]:
    lo = min(c.low  for c in candles)
    hi = max(c.high for c in candles)
    if math.isclose(lo, hi):
        return lo, lo, hi
    step    = (hi - lo) / bins
    vols    = [0.0] * bins
    centers = [lo + (i + 0.5) * step for i in range(bins)]
    for c in candles:
        tp  = (c.high + c.low + c.close) / 3.0
        idx = min(bins - 1, max(0, int((tp - lo) / step)))
        vols[idx] += c.volume
    total = sum(vols)
    poc_i = max(range(bins), key=lambda i: vols[i])
    incl  = {poc_i}
    cov   = vols[poc_i]
    li, ri = poc_i - 1, poc_i + 1
    while total and cov / total < 0.70 and (li >= 0 or ri < bins):
        lv = vols[li] if li >= 0   else -1.0
        rv = vols[ri] if ri < bins else -1.0
        if rv >= lv:
            if ri < bins: incl.add(ri); cov += vols[ri]; ri += 1
            else:         incl.add(li); cov += vols[li]; li -= 1
        else:
            if li >= 0:   incl.add(li); cov += vols[li]; li -= 1
            else:         incl.add(ri); cov += vols[ri]; ri += 1
    return centers[poc_i], min(centers[i] for i in incl), max(centers[i] for i in incl)


# ── HTF analysis ──────────────────────────────────────────────────────────────

def htf_analysis(htf_candles: List[Candle], timestamp_ms: int,
                 fast_p: int = 21, slow_p: int = 55, ultra_p: int = 200,
                 ) -> Tuple[str, bool]:
    rel = [c for c in htf_candles if c.close_time < timestamp_ms]
    if len(rel) < ultra_p + 5:
        return "neutral", True
    closes     = [c.close for c in rel]
    e21        = ema_of(closes, fast_p)
    e21_p      = ema_of(closes[:-1], fast_p)
    e55        = ema_of(closes, slow_p)
    e200       = ema_of(closes, ultra_p)
    last       = rel[-1].close
    above_200  = last > e200
    rising_21  = e21 > e21_p * 1.0002
    falling_21 = e21 < e21_p * 0.9998
    if last > e21 and e21 > e55 * 0.998 and rising_21:
        return "up", above_200
    if last < e21 and e21 < e55 * 1.002 and falling_21:
        return "down", above_200
    return "neutral", above_200


# ── Structure helpers ─────────────────────────────────────────────────────────

def has_higher_highs_and_lows(candles: List[Candle], n: int = 4) -> bool:
    if len(candles) < n * 2:
        return False
    highs = [c.high for c in candles[-n:]]
    lows  = [c.low  for c in candles[-n:]]
    hh = all(highs[i] > highs[i - 1] for i in range(1, len(highs)))
    hl = all(lows[i]  > lows[i - 1]  for i in range(1, len(lows)))
    return hh or hl


def has_lower_lows_and_highs(candles: List[Candle], n: int = 4) -> bool:
    if len(candles) < n * 2:
        return False
    highs = [c.high for c in candles[-n:]]
    lows  = [c.low  for c in candles[-n:]]
    ll = all(lows[i]  < lows[i - 1]  for i in range(1, len(lows)))
    lh = all(highs[i] < highs[i - 1] for i in range(1, len(highs)))
    return ll or lh


def retest_of_breakout(candles: List[Candle], index: int,
                        swing_high: float, atr: float,
                        lookback: int = 4) -> bool:
    start = max(0, index - lookback)
    if not any(c.close > swing_high for c in candles[start:index]):
        return False
    cur = candles[index]
    return abs(cur.low - swing_high) < atr * 0.5


def retest_of_breakdown(candles: List[Candle], index: int,
                         swing_low: float, atr: float,
                         lookback: int = 4) -> bool:
    start = max(0, index - lookback)
    if not any(c.close < swing_low for c in candles[start:index]):
        return False
    cur = candles[index]
    return abs(cur.high - swing_low) < atr * 0.5


# ── Signal generation ─────────────────────────────────────────────────────────

def generate_signal(
    candles: List[Candle],
    index: int,
    htf_candles: List[Candle],
    min_score: int        = 7,
    rsi_long_min: float   = 45.0,
    rsi_long_max: float   = 75.0,
    rsi_short_max: float  = 55.0,
    rsi_short_min: float  = 25.0,
    vol_spike_max: float  = 5.0,
    chop_max: float       = 60.0,
    session_filter: bool  = True,
    enable_shorts: bool   = True,
) -> Optional[Tuple[str, int, float, str]]:

    PROFILE_LB = 60
    STRUCT_LB  = 20
    FAST_P, MID_P, SLOW_P = 9, 21, 50
    RSI_P = 14

    if index < max(PROFILE_LB, STRUCT_LB, SLOW_P, RSI_P) + 5:
        return None

    cur   = candles[index]
    prev  = candles[index - 1]
    prev2 = candles[index - 2]

    # Session filter: skip 00:00-05:59 UTC
    if session_filter and 0 <= candle_hour_utc(cur) < 6:
        return None

    # Choppiness
    ci = choppiness_index(candles[index - 15 : index + 1], 14)
    if ci > chop_max:
        return None

    # HTF
    trend, above_200 = htf_analysis(htf_candles, cur.open_time)

    # Volume
    window  = candles[index - PROFILE_LB : index]
    avg_vol = fmean(c.volume for c in window)
    vol_r   = cur.volume / avg_vol if avg_vol > 0 else 0.0
    if vol_r > vol_spike_max:
        return None

    # Indicators
    rsi = calc_rsi(candles[index - RSI_P - 1 : index + 1], RSI_P)

    closes   = [c.close for c in candles[index - SLOW_P : index + 1]]
    ema_fast = ema_of(closes, FAST_P)
    ema_mid  = ema_of(closes, MID_P)
    ema_slow = ema_of(closes, SLOW_P)
    bull_stack = ema_fast > ema_mid * 0.999 and ema_mid > ema_slow * 0.999
    bear_stack = ema_fast < ema_mid * 1.001 and ema_mid < ema_slow * 1.001

    # Volume profile
    poc, val, vah = build_volume_profile(window)

    # Structure
    struct     = candles[index - STRUCT_LB : index]
    swing_high = max(c.high for c in struct[:-1])
    swing_low  = min(c.low  for c in struct[:-1])

    # ATR
    cur_atr = calc_atr(candles[max(0, index - 15) : index], 14)

    # Candle quality
    avg_body = fmean(abs(c.close - c.open) for c in window)
    body     = abs(cur.close - cur.open)
    body_r   = body / avg_body if avg_body > 0 else 0.0

    rng         = max(cur.high - cur.low, 1e-9)
    close_pos   = (cur.close - cur.low) / rng
    strong_bull = close_pos >= 0.65
    strong_bear = close_pos <= 0.35

    upper_wick = cur.high - max(cur.open, cur.close)
    lower_wick = min(cur.open, cur.close) - cur.low
    no_up_wick = upper_wick < body * 0.35
    no_lo_wick = lower_wick < body * 0.35

    prev_body   = abs(prev.close - prev.open)
    prev_body_r = prev_body / avg_body if avg_body > 0 else 0.0
    prev_bull   = prev.close > prev.open
    prev_bear   = prev.close < prev.open
    exhaustion_bull = prev.close < prev.open and prev_body_r > 2.0
    exhaustion_bear = prev.close > prev.open and prev_body_r > 2.0

    clean_up  = all(c.close <= swing_high for c in struct[-6:-1])
    clean_dn  = all(c.close >= swing_low  for c in struct[-6:-1])
    trend_up   = has_higher_highs_and_lows(struct)
    trend_down = has_lower_lows_and_highs(struct)

    is_retest_long  = retest_of_breakout(candles, index, swing_high, cur_atr)
    is_retest_short = retest_of_breakdown(candles, index, swing_low, cur_atr)

    # ── LONG ─────────────────────────────────────────────────────────────
    long_gate = (
        trend != "down"
        and rsi >= rsi_long_min
        and rsi <= rsi_long_max
        and cur.close > cur.open
        and vol_r  >= 1.30
        and body_r >= 1.10
        and cur.close >= poc
        and not exhaustion_bull
        and (
            (cur.close > swing_high and prev.close <= swing_high and clean_up)
            or
            (is_retest_long and strong_bull)
        )
    )

    if long_gate:
        s = 0
        if trend == "up":         s += 3
        if trend == "neutral":    s += 1
        if above_200:             s += 2
        if bull_stack:            s += 2
        if trend_up:              s += 2
        if is_retest_long:        s += 2
        if vol_r  >= 1.60:        s += 1
        if vol_r  >= 2.20:        s += 1
        if body_r >= 1.40:        s += 1
        if cur.close > vah:       s += 1
        if strong_bull:           s += 1
        if no_up_wick:            s += 1
        if rsi >= 52:             s += 1
        if prev_bull:             s += 1
        if cur.low > prev2.low:   s += 1
        if s >= min_score:
            return "long", s, rsi, trend

    # ── SHORT ────────────────────────────────────────────────────────────
    short_gate = (
        enable_shorts
        and trend != "up"
        and rsi <= rsi_short_max
        and rsi >= rsi_short_min
        and cur.close < cur.open
        and vol_r  >= 1.30
        and body_r >= 1.10
        and cur.close <= poc
        and not exhaustion_bear
        and (
            (cur.close < swing_low and prev.close >= swing_low and clean_dn)
            or
            (is_retest_short and strong_bear)
        )
    )

    if short_gate:
        s = 0
        if trend == "down":       s += 3
        if trend == "neutral":    s += 1
        if not above_200:         s += 2
        if bear_stack:            s += 2
        if trend_down:            s += 2
        if is_retest_short:       s += 2
        if vol_r  >= 1.60:        s += 1
        if vol_r  >= 2.20:        s += 1
        if body_r >= 1.40:        s += 1
        if cur.close < val:       s += 1
        if strong_bear:           s += 1
        if no_lo_wick:            s += 1
        if rsi <= 48:             s += 1
        if prev_bear:             s += 1
        if cur.high < prev2.high: s += 1
        if s >= min_score:
            return "short", s, rsi, trend

    return None


# ── Backtester ────────────────────────────────────────────────────────────────

def backtest(
    candles: List[Candle],
    htf_candles: List[Candle],
    initial_balance: float,
    risk_pct: float              = 0.10,
    margin_per_trade: Optional[float] = None,
    leverage: float              = 10.0,
    tp_atr_mult: float           = 2.5,   # ↑ from 2.0 → bigger wins
    tp_atr_mult_high: float      = 3.0,   # for score≥10: extended TP
    high_score_threshold: int    = 10,    # score needed for extended TP
    sl_atr_mult: float           = 0.8,   # tighter base risk
    atr_period: int              = 14,
    fee_rate: float              = 0.0004,
    trail_activation_atr: float  = 1.3,   # start protecting profits sooner
    trail_distance_atr: float    = 0.4,   # tighter trail after activation
    trail_min_profit_atr: float  = 0.7,   # lock some profit after trail starts
    max_hold_candles: int        = 20,    # ↑ from 16 → TP has more time
    max_consecutive_losses: int  = 3,
    cooldown_candles: int        = 5,
    min_score: int               = 7,
    rsi_long_min: float          = 45.0,
    rsi_long_max: float          = 75.0,
    rsi_short_max: float         = 55.0,
    rsi_short_min: float         = 25.0,
    vol_spike_max: float         = 5.0,
    chop_max: float              = 60.0,
    session_filter: bool         = True,
    enable_shorts: bool          = True,
) -> Tuple[dict, List[Trade]]:

    balance       = initial_balance
    trades: List[Trade] = []
    index         = 100
    consec_losses = 0
    cooldown_left = 0

    while index < len(candles) - 1:
        if cooldown_left > 0:
            cooldown_left -= 1
            index += 1
            continue

        sig = generate_signal(
            candles, index, htf_candles, min_score,
            rsi_long_min, rsi_long_max, rsi_short_max, rsi_short_min,
            vol_spike_max, chop_max, session_filter, enable_shorts,
        )
        if sig is None:
            index += 1
            continue

        direction, score, rsi_val, htf_tr = sig
        entry_c     = candles[index]
        entry_price = entry_c.close

        cur_atr = calc_atr(candles[max(0, index - atr_period - 1) : index], atr_period)
        if cur_atr <= 0:
            index += 1
            continue

        if margin_per_trade is not None:
            margin_trade = min(balance, max(0.50, margin_per_trade))
        else:
            margin_trade = max(0.50, balance * risk_pct)
        notional     = margin_trade * leverage
        qty          = notional / entry_price

        # High-confidence trades get extended TP
        tp_mult = tp_atr_mult_high if score >= high_score_threshold else tp_atr_mult

        if direction == "long":
            tp_price = entry_price + tp_mult  * cur_atr
            sl_price = entry_price - sl_atr_mult * cur_atr
        else:
            tp_price = entry_price - tp_mult  * cur_atr
            sl_price = entry_price + sl_atr_mult * cur_atr

        peak_fav    = entry_price
        trail_sl: Optional[float] = None
        exit_price  = candles[-1].close
        exit_reason = "end_of_data"
        exit_index  = len(candles) - 1
        max_exit    = min(index + max_hold_candles, len(candles) - 1)

        for ei in range(index + 1, max_exit + 1):
            c       = candles[ei]
            is_last = (ei == max_exit)

            if direction == "long":
                if c.high > peak_fav:
                    peak_fav = c.high
                gain_atr = (peak_fav - entry_price) / cur_atr
                if gain_atr >= trail_activation_atr:
                    # Trail locks in at least (gain - trail_distance) ATR
                    profit_floor = entry_price + trail_min_profit_atr * cur_atr
                    cand = max(
                        peak_fav - trail_distance_atr * cur_atr,
                        profit_floor,
                    )
                    if trail_sl is None or cand > trail_sl:
                        trail_sl = cand
                eff_sl = max(sl_price, trail_sl) if trail_sl else sl_price
                hit_tp = c.high >= tp_price
                hit_sl = c.low  <= eff_sl
                if hit_tp:
                    exit_price  = tp_price
                    exit_reason = "take_profit"
                    exit_index  = ei; break
                elif hit_sl:
                    exit_price  = eff_sl
                    exit_reason = "trailing_stop" if (trail_sl and eff_sl > sl_price) else "stop_loss"
                    exit_index  = ei; break
                elif is_last:
                    exit_price  = c.close
                    exit_reason = "time_exit"
                    exit_index  = ei; break
            else:
                if c.low < peak_fav:
                    peak_fav = c.low
                gain_atr = (entry_price - peak_fav) / cur_atr
                if gain_atr >= trail_activation_atr:
                    profit_floor = entry_price - trail_min_profit_atr * cur_atr
                    cand = min(
                        peak_fav + trail_distance_atr * cur_atr,
                        profit_floor,
                    )
                    if trail_sl is None or cand < trail_sl:
                        trail_sl = cand
                eff_sl = min(sl_price, trail_sl) if trail_sl else sl_price
                hit_tp = c.low  <= tp_price
                hit_sl = c.high >= eff_sl
                if hit_tp:
                    exit_price  = tp_price
                    exit_reason = "take_profit"
                    exit_index  = ei; break
                elif hit_sl:
                    exit_price  = eff_sl
                    exit_reason = "trailing_stop" if (trail_sl and eff_sl < sl_price) else "stop_loss"
                    exit_index  = ei; break
                elif is_last:
                    exit_price  = c.close
                    exit_reason = "time_exit"
                    exit_index  = ei; break

        gross   = ((exit_price - entry_price) * qty if direction == "long"
                   else (entry_price - exit_price) * qty)
        fees    = notional * fee_rate * 2.0
        net_pnl = gross - fees
        balance = max(0.01, balance + net_pnl)

        if net_pnl <= 0:
            consec_losses += 1
            if consec_losses >= max_consecutive_losses:
                cooldown_left = cooldown_candles
                consec_losses = 0
        else:
            consec_losses = 0

        trades.append(Trade(
            side=direction,
            entry_time=entry_c.close_time,
            exit_time=candles[exit_index].close_time,
            entry_price=round(entry_price, 6),
            exit_price=round(exit_price, 6),
            quantity=round(qty, 6),
            margin=round(margin_trade, 4),
            leverage=leverage,
            gross_pnl=round(gross, 6),
            fees=round(fees, 6),
            net_pnl=round(net_pnl, 6),
            result="win" if net_pnl > 0 else "loss",
            exit_reason=exit_reason,
            signal_score=score,
            tp_price=round(tp_price, 4),
            sl_price=round(sl_price, 4),
            atr_at_entry=round(cur_atr, 4),
            rsi_at_entry=round(rsi_val, 2),
            htf_trend=htf_tr,
        ))
        index = exit_index + 1

    wins   = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl <= 0]
    longs  = [t for t in trades if t.side == "long"]
    shorts = [t for t in trades if t.side == "short"]

    by_reason: Dict[str, int] = {}
    for t in trades:
        by_reason[t.exit_reason] = by_reason.get(t.exit_reason, 0) + 1

    tw = sum(t.net_pnl  for t in wins)
    tl = sum(-t.net_pnl for t in losses)

    def wr_str(ts: List[Trade]) -> str:
        if not ts: return "n/a"
        w = sum(1 for t in ts if t.net_pnl > 0)
        return f"{w}/{len(ts)} ({w/len(ts)*100:.0f}%)"

    months = max(
        (candles[-1].close_time - candles[100].close_time) / (1000 * 3600 * 24 * 30.44), 1
    )

    peak_bal = initial_balance
    max_dd   = 0.0
    running  = initial_balance
    for t in trades:
        running += t.net_pnl
        if running > peak_bal:
            peak_bal = running
        dd = (peak_bal - running) / peak_bal * 100
        if dd > max_dd:
            max_dd = dd

    summary = {
        "trade_count":        len(trades),
        "trades_per_month":   round(len(trades) / months, 1),
        "win_count":          len(wins),
        "loss_count":         len(losses),
        "win_rate_pct":       round(len(wins) / len(trades) * 100, 2) if trades else 0,
        "long_wr":            wr_str(longs),
        "short_wr":           wr_str(shorts),
        "exit_breakdown":     by_reason,
        "avg_win":            round(fmean(t.net_pnl  for t in wins)   if wins   else 0, 4),
        "avg_loss":           round(fmean(-t.net_pnl for t in losses) if losses else 0, 4),
        "profit_factor":      round(tw / tl, 3) if tl else float("inf"),
        "win_amount":         round(tw, 4),
        "loss_amount":        round(tl, 4),
        "net_profit":         round(tw - tl, 4),
        "net_profit_pm":      round((tw - tl) / months, 4),
        "final_balance":      round(balance, 4),
        "starting_balance":   round(initial_balance, 4),
        "return_pct":         round((balance - initial_balance) / initial_balance * 100, 2),
        "monthly_return_pct": round((balance - initial_balance) / initial_balance * 100 / months, 2),
        "max_drawdown_pct":   round(max_dd, 2),
    }
    return summary, trades


def write_trades_csv(path: str, trades: List[Trade]) -> None:
    if not trades:
        print("  No trades to write.")
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(asdict(trades[0]).keys()))
        w.writeheader()
        for t in trades:
            w.writerow(asdict(t))


def print_report(s: dict, symbol: str, interval: str, candles: List[Candle]) -> None:
    W = 60
    print("=" * W)
    print("  Backtest Report v11 — Fixed $1 Sizing + Profit Lock Trail")
    print("=" * W)
    print(f"  Symbol          : {symbol}")
    print(f"  Interval        : {interval} (HTF: 4h)")
    print(f"  Candles         : {len(candles)}")
    print(f"  Data start      : {iso_ts(candles[0].open_time)}")
    print(f"  Data end        : {iso_ts(candles[-1].close_time)}")
    print("-" * W)
    tpm      = s["trades_per_month"]
    tpm_icon = "✅" if tpm >= 8 else ("⚠️" if tpm >= 5 else "❌")
    print(f"  Trades total    : {s['trade_count']}  (~{tpm}/month) {tpm_icon}")
    print(f"  Wins            : {s['win_count']}  (${s['win_amount']:.4f})")
    print(f"  Losses          : {s['loss_count']}  (${s['loss_amount']:.4f})")
    wr       = s["win_rate_pct"]
    wr_icon  = "✅" if wr >= 45 else ("⚠️" if wr >= 38 else "❌")
    print(f"  Win rate        : {wr:.1f}%  {wr_icon}")
    print(f"  Long  WR        : {s['long_wr']}")
    print(f"  Short WR        : {s['short_wr']}")
    exits = s["exit_breakdown"]
    tp = exits.get("take_profit",   0)
    tr = exits.get("trailing_stop", 0)
    sl = exits.get("stop_loss",     0)
    te = exits.get("time_exit",     0)
    tp_icon  = "✅" if tp >= 5 else ("⚠️" if tp >= 1 else "❌")
    print(f"  Exits TP/Trail/SL/Time: {tp}/{tr}/{sl}/{te}  {tp_icon}")
    print(f"  Avg win         : ${s['avg_win']:.4f}")
    print(f"  Avg loss        : ${s['avg_loss']:.4f}")
    rr = s["avg_win"] / s["avg_loss"] if s["avg_loss"] else 0
    rr_icon  = "✅" if rr >= 1.8 else ("⚠️" if rr >= 1.3 else "❌")
    print(f"  Actual R:R      : {rr:.2f}:1  {rr_icon}")
    pf       = s["profit_factor"]
    pf_icon  = "✅" if pf >= 1.5 else ("⚠️" if pf >= 1.1 else "❌")
    print(f"  Profit factor   : {pf:.3f}  {pf_icon}")
    print(f"  Max drawdown    : {s['max_drawdown_pct']:.1f}%")
    print("-" * W)
    goal_ok  = "✅" if s["net_profit_pm"] >= 1.0 else ("⚠️" if s["net_profit_pm"] >= 0.50 else "❌")
    print(f"  Net profit      : ${s['net_profit']:.4f}  total")
    print(f"  Avg / month     : ${s['net_profit_pm']:.4f}  {goal_ok} (goal: $1.00)")
    print(f"  Monthly return  : {s['monthly_return_pct']:.2f}%")
    print(f"  Total return    : {s['return_pct']:.2f}%")
    print(f"  Final balance   : ${s['final_balance']:.4f}")
    print("=" * W)
    # Smart diagnosis
    if rr < 1.5:
        print("  ⚠  R:R low  → try --tp-atr-mult 3.0 --sl-atr-mult 0.8")
    if wr < 40:
        print("  ⚠  WR low   → try --min-score 8 --chop-max 55")
    if tp == 0:
        print("  ⚠  TP=0     → try --tp-atr-mult 1.8 --max-hold-candles 24")
    if te > s["trade_count"] * 0.3:
        print("  ⚠  Many time exits → try --max-hold-candles 28")
    if s["net_profit_pm"] >= 1.0:
        print("  🎯 GOAL REACHED — consider running on BTCUSDT to validate")
    else:
        gap = max(0.0, 1.0 - s["net_profit_pm"])
        print(f"  ⏳ Gap to goal : ${gap:.4f}/month still needed")


def main() -> None:
    p = argparse.ArgumentParser(
        description="PA + VP backtester v11 — fixed sizing, profit lock trail, score-based TP"
    )
    p.add_argument("--symbol",                   default="SOLUSDT")
    p.add_argument("--interval",                 default="1h")
    p.add_argument("--htf-interval",             default="4h")
    p.add_argument("--limit",         type=int,  default=5000)
    p.add_argument("--initial-balance",          type=float, default=10.0)
    p.add_argument("--risk-pct",                 type=float, default=0.10)
    p.add_argument("--margin-per-trade",         type=float, default=1.0,
                   help="Fixed margin to allocate per trade. Set 0 or negative to disable and use --risk-pct.")
    p.add_argument("--leverage",                 type=float, default=10.0)
    p.add_argument("--tp-atr-mult",              type=float, default=2.5)
    p.add_argument("--tp-atr-mult-high",         type=float, default=3.0,
                   help="TP multiplier for high-score trades")
    p.add_argument("--high-score-threshold",     type=int,   default=10,
                   help="Score needed to use extended TP")
    p.add_argument("--sl-atr-mult",              type=float, default=0.8)
    p.add_argument("--atr-period",               type=int,   default=14)
    p.add_argument("--fee-rate",                 type=float, default=0.0004)
    p.add_argument("--trail-activation-atr",     type=float, default=1.3)
    p.add_argument("--trail-distance-atr",       type=float, default=0.40)
    p.add_argument("--trail-min-profit-atr",     type=float, default=0.70,
                   help="Minimum ATR profit to lock once trailing activates")
    p.add_argument("--max-hold-candles",         type=int,   default=20)
    p.add_argument("--max-consecutive-losses",   type=int,   default=3)
    p.add_argument("--cooldown-candles",         type=int,   default=5)
    p.add_argument("--min-score",                type=int,   default=7)
    p.add_argument("--rsi-long-min",             type=float, default=45.0)
    p.add_argument("--rsi-long-max",             type=float, default=75.0)
    p.add_argument("--rsi-short-max",            type=float, default=55.0)
    p.add_argument("--rsi-short-min",            type=float, default=25.0)
    p.add_argument("--vol-spike-max",            type=float, default=5.0)
    p.add_argument("--chop-max",                 type=float, default=60.0)
    p.add_argument("--no-session-filter",        action="store_true")
    p.add_argument("--no-shorts",                action="store_true")
    p.add_argument("--trades-output",            default="trades_v11.csv")
    args = p.parse_args()

    print(f"  Fetching {args.limit} × {args.interval} candles...")
    candles = fetch_klines(args.symbol, args.interval, args.limit)

    htf_limit = max(600, args.limit // 4 + 200)
    print(f"  Fetching {htf_limit} × {args.htf_interval} candles (HTF)...")
    htf_candles = fetch_klines(args.symbol, args.htf_interval, htf_limit)

    print(f"  LTF candles : {len(candles)}")
    print(f"  HTF candles : {len(htf_candles)}")
    print(f"  Session flt : {'OFF' if args.no_session_filter else 'ON (skip 00-05 UTC)'}")
    print(f"  Shorts      : {'disabled' if args.no_shorts else 'enabled'}")
    margin_mode = (
        f"${args.margin_per_trade:.2f} fixed margin/trade"
        if args.margin_per_trade and args.margin_per_trade > 0
        else f"{args.risk_pct * 100:.1f}% of balance/trade"
    )
    print(f"  Sizing      : {margin_mode} @ {args.leverage:.1f}x")
    print(f"  TP          : {args.tp_atr_mult} ATR  (score≥{args.high_score_threshold}: {args.tp_atr_mult_high} ATR)")
    print(f"  SL          : {args.sl_atr_mult} ATR   Max hold: {args.max_hold_candles}h")

    summary, trades = backtest(
        candles=candles,
        htf_candles=htf_candles,
        initial_balance=args.initial_balance,
        risk_pct=args.risk_pct,
        margin_per_trade=(args.margin_per_trade if args.margin_per_trade and args.margin_per_trade > 0 else None),
        leverage=args.leverage,
        tp_atr_mult=args.tp_atr_mult,
        tp_atr_mult_high=args.tp_atr_mult_high,
        high_score_threshold=args.high_score_threshold,
        sl_atr_mult=args.sl_atr_mult,
        atr_period=args.atr_period,
        fee_rate=args.fee_rate,
        trail_activation_atr=args.trail_activation_atr,
        trail_distance_atr=args.trail_distance_atr,
        trail_min_profit_atr=args.trail_min_profit_atr,
        max_hold_candles=args.max_hold_candles,
        max_consecutive_losses=args.max_consecutive_losses,
        cooldown_candles=args.cooldown_candles,
        min_score=args.min_score,
        rsi_long_min=args.rsi_long_min,
        rsi_long_max=args.rsi_long_max,
        rsi_short_max=args.rsi_short_max,
        rsi_short_min=args.rsi_short_min,
        vol_spike_max=args.vol_spike_max,
        chop_max=args.chop_max,
        session_filter=not args.no_session_filter,
        enable_shorts=not args.no_shorts,
    )

    write_trades_csv(args.trades_output, trades)
    print(f"  Trade log   : {args.trades_output}")
    print_report(summary, args.symbol, args.interval, candles)


if __name__ == "__main__":
    main()
