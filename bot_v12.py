#!/usr/bin/env python3
"""
Price Action + Volume Profile Futures Backtester — v12

v12 goals:
  - Target >=$10/month profit on SOLUSDT at 10x leverage
  - Switch to 15m entries with 1h HTF for ~4x more trade opportunities
  - Add multiple entry patterns: breakout, retest, EMA pullback, engulfing,
    momentum, trend-follow continuation, mean-reversion bounce
  - Breakeven-stop mechanism to convert losing trades into scratch trades
  - Compound position sizing (% of balance) with score-based scaling
  - Optimised scalp parameters: tight TP (0.8 ATR), BE trigger at 0.2 ATR
  - Adaptive trailing stop that tightens as profit grows

Key changes from v11:
  1. 15m default timeframe (1h HTF) for more entries
  2. Added EMA pullback, engulfing, trend-follow, and mean-reversion patterns
  3. Breakeven stop: once +0.2 ATR, SL moves to entry+fees (scratch trade)
  4. Compound sizing: margin = % of balance (grows with profits)
  5. Score-based scaling: 2x margin on score>=14, 1.5x on score>=10
  6. Tighter scalp exits: 0.8 ATR TP, 0.6 ATR SL, trail at 0.4 ATR
  7. Relaxed entry filters for more volume (wider RSI, chop, session filter off)
  8. ADX-based trend strength filter
  9. Reduced cooldown (2 candles after 5 consecutive losses)
  10. Two sizing modes: compound (default) or fixed margin (--margin-per-trade)

Backtested results (SOLUSDT, 15m, ~83 days of data):
  - Fixed $1 margin:       ~463 trades, 69% WR, PF 1.74, ~$1.1/month
  - 25% compound+scale:    ~463 trades, 69% WR, PF 1.63, ~$4-5/month
  - 30% compound+scale:    ~463 trades, 69% WR, PF 1.61, ~$5-6/month
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


# Use data-api.binance.vision (globally accessible spot data endpoint).
# Price action is identical to futures for backtesting purposes.
KLINES_URL         = "https://data-api.binance.vision/api/v3/klines"
KLINES_URL_FUTURES = "https://fapi.binance.com/fapi/v1/klines"
MAX_PER_FETCH      = 1000


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
    pattern_type: str
    tp_price: float
    sl_price: float
    atr_at_entry: float
    rsi_at_entry: float
    htf_trend: str


def iso_ts(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def candle_hour_utc(c: Candle) -> int:
    return datetime.fromtimestamp(c.open_time / 1000, tz=timezone.utc).hour


# -- Data fetching -------------------------------------------------------------

def fetch_klines(symbol: str, interval: str, limit: int,
                 use_futures: bool = False) -> List[Candle]:
    symbol = symbol.upper()
    base_url = KLINES_URL_FUTURES if use_futures else KLINES_URL
    chunks: List[List[Candle]] = []
    remaining = limit
    end_time: Optional[int] = None

    while remaining > 0:
        batch = min(remaining, MAX_PER_FETCH)
        params: Dict[str, object] = {
            "symbol": symbol, "interval": interval, "limit": batch,
        }
        if end_time is not None:
            params["endTime"] = end_time
        url = f"{base_url}?{urllib.parse.urlencode(params)}"
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                payload = json.load(r)
        except Exception:
            if use_futures:
                url = f"{KLINES_URL}?{urllib.parse.urlencode(params)}"
                with urllib.request.urlopen(url, timeout=30) as r:
                    payload = json.load(r)
            else:
                raise
        if not payload:
            break
        chunk = [
            Candle(
                int(x[0]), float(x[1]), float(x[2]), float(x[3]),
                float(x[4]), float(x[5]), int(x[6]),
            )
            for x in payload
        ]
        chunks.insert(0, chunk)
        end_time   = chunk[0].open_time - 1
        remaining -= len(chunk)
        if len(chunk) < batch:
            break

    all_c: List[Candle] = []
    for ch in chunks:
        all_c.extend(ch)

    seen: set = set()
    unique: List[Candle] = []
    for c in sorted(all_c, key=lambda x: x.open_time):
        if c.open_time not in seen:
            seen.add(c.open_time)
            unique.append(c)
    return unique[-limit:]


# -- Indicators ----------------------------------------------------------------

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
    trs: List[float] = []
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
    gains: List[float] = []
    losses: List[float] = []
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


def calc_adx(candles: List[Candle], period: int = 14) -> float:
    """Simplified ADX for trend-strength measurement."""
    if len(candles) < period + 2:
        return 25.0
    plus_dm_list: List[float] = []
    minus_dm_list: List[float] = []
    tr_list: List[float] = []
    for i in range(1, len(candles)):
        hi_diff = candles[i].high - candles[i - 1].high
        lo_diff = candles[i - 1].low - candles[i].low
        plus_dm_list.append(hi_diff if hi_diff > lo_diff and hi_diff > 0 else 0)
        minus_dm_list.append(lo_diff if lo_diff > hi_diff and lo_diff > 0 else 0)
        tr_list.append(max(
            candles[i].high - candles[i].low,
            abs(candles[i].high - candles[i - 1].close),
            abs(candles[i].low - candles[i - 1].close),
        ))
    n = min(period, len(tr_list))
    if n == 0:
        return 25.0
    atr_val = fmean(tr_list[-n:])
    if atr_val == 0:
        return 25.0
    plus_di = 100.0 * fmean(plus_dm_list[-n:]) / atr_val
    minus_di = 100.0 * fmean(minus_dm_list[-n:]) / atr_val
    di_sum = plus_di + minus_di
    if di_sum == 0:
        return 0.0
    return 100.0 * abs(plus_di - minus_di) / di_sum


def build_volume_profile(candles: List[Candle], bins: int = 36,
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
    incl: set = {poc_i}
    cov   = vols[poc_i]
    li, ri = poc_i - 1, poc_i + 1
    while total and cov / total < 0.70 and (li >= 0 or ri < bins):
        lv = vols[li] if li >= 0   else -1.0
        rv = vols[ri] if ri < bins else -1.0
        if rv >= lv:
            if ri < bins:
                incl.add(ri); cov += vols[ri]; ri += 1
            else:
                incl.add(li); cov += vols[li]; li -= 1
        else:
            if li >= 0:
                incl.add(li); cov += vols[li]; li -= 1
            else:
                incl.add(ri); cov += vols[ri]; ri += 1
    return centers[poc_i], min(centers[i] for i in incl), max(centers[i] for i in incl)


# -- HTF analysis --------------------------------------------------------------

def htf_analysis(htf_candles: List[Candle], timestamp_ms: int,
                 fast_p: int = 21, slow_p: int = 55, ultra_p: int = 200,
                 ) -> Tuple[str, bool, float]:
    """Returns (trend_direction, above_200_ema, trend_strength 0-1)."""
    rel = [c for c in htf_candles if c.close_time < timestamp_ms]
    if len(rel) < ultra_p + 5:
        return "neutral", True, 0.0
    closes     = [c.close for c in rel]
    e21        = ema_of(closes, fast_p)
    e21_p      = ema_of(closes[:-1], fast_p)
    e55        = ema_of(closes, slow_p)
    e200       = ema_of(closes, ultra_p)
    last       = rel[-1].close
    above_200  = last > e200
    rising_21  = e21 > e21_p * 1.0002
    falling_21 = e21 < e21_p * 0.9998

    ema_spread = abs(e21 - e55) / last if last > 0 else 0
    strength = min(1.0, ema_spread * 100)

    if last > e21 and e21 > e55 * 0.998 and rising_21:
        return "up", above_200, strength
    if last < e21 and e21 < e55 * 1.002 and falling_21:
        return "down", above_200, strength
    return "neutral", above_200, strength


# -- Structure helpers ---------------------------------------------------------

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
                       lookback: int = 6) -> bool:
    start = max(0, index - lookback)
    if not any(c.close > swing_high for c in candles[start:index]):
        return False
    cur = candles[index]
    return abs(cur.low - swing_high) < atr * 0.6


def retest_of_breakdown(candles: List[Candle], index: int,
                        swing_low: float, atr: float,
                        lookback: int = 6) -> bool:
    start = max(0, index - lookback)
    if not any(c.close < swing_low for c in candles[start:index]):
        return False
    cur = candles[index]
    return abs(cur.high - swing_low) < atr * 0.6


def is_bullish_engulfing(cur: Candle, prev: Candle) -> bool:
    if prev.close >= prev.open or cur.close <= cur.open:
        return False
    return cur.close > prev.open and cur.open <= prev.close


def is_bearish_engulfing(cur: Candle, prev: Candle) -> bool:
    if prev.close <= prev.open or cur.close >= cur.open:
        return False
    return cur.close < prev.open and cur.open >= prev.close


# -- Signal generation ---------------------------------------------------------

def generate_signal(
    candles: List[Candle],
    index: int,
    htf_candles: List[Candle],
    min_score: int        = 5,
    rsi_long_min: float   = 32.0,
    rsi_long_max: float   = 82.0,
    rsi_short_max: float  = 68.0,
    rsi_short_min: float  = 18.0,
    vol_spike_max: float  = 6.0,
    chop_max: float       = 65.0,
    session_filter: bool  = False,
    enable_shorts: bool   = True,
) -> Optional[Tuple[str, int, float, str, str]]:
    """Returns (direction, score, rsi, htf_trend, pattern_type) or None."""

    PROFILE_LB = 60
    STRUCT_LB  = 24
    FAST_P, MID_P, SLOW_P = 9, 21, 50
    RSI_P = 14

    if index < max(PROFILE_LB, STRUCT_LB, SLOW_P, RSI_P) + 5:
        return None

    cur   = candles[index]
    prev  = candles[index - 1]
    prev2 = candles[index - 2]

    # Session filter: skip 01:00-04:59 UTC (low-liquidity window)
    if session_filter and 1 <= candle_hour_utc(cur) < 5:
        return None

    # Choppiness -- reject ranging markets
    ci = choppiness_index(candles[index - 15 : index + 1], 14)
    if ci > chop_max:
        return None

    # HTF trend
    trend, above_200, trend_strength = htf_analysis(htf_candles, cur.open_time)

    # ADX for LTF trend strength
    adx = calc_adx(candles[max(0, index - 20) : index + 1], 14)

    # Volume
    window  = candles[index - PROFILE_LB : index]
    avg_vol = fmean(c.volume for c in window)
    vol_r   = cur.volume / avg_vol if avg_vol > 0 else 0.0
    if vol_r > vol_spike_max:
        return None

    # RSI
    rsi = calc_rsi(candles[index - RSI_P - 1 : index + 1], RSI_P)

    # EMAs
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
    if cur_atr <= 0:
        return None

    # Candle quality metrics
    avg_body = fmean(abs(c.close - c.open) for c in window)
    body     = abs(cur.close - cur.open)
    body_r   = body / avg_body if avg_body > 0 else 0.0

    rng         = max(cur.high - cur.low, 1e-9)
    close_pos   = (cur.close - cur.low) / rng
    strong_bull = close_pos >= 0.60
    strong_bear = close_pos <= 0.40

    upper_wick = cur.high - max(cur.open, cur.close)
    lower_wick = min(cur.open, cur.close) - cur.low
    no_up_wick = upper_wick < body * 0.40
    no_lo_wick = lower_wick < body * 0.40

    prev_body   = abs(prev.close - prev.open)
    prev_body_r = prev_body / avg_body if avg_body > 0 else 0.0
    prev_bull   = prev.close > prev.open
    prev_bear   = prev.close < prev.open
    exhaustion_bull = prev.close < prev.open and prev_body_r > 2.5
    exhaustion_bear = prev.close > prev.open and prev_body_r > 2.5

    clean_up  = all(c.close <= swing_high for c in struct[-6:-1])
    clean_dn  = all(c.close >= swing_low  for c in struct[-6:-1])
    trend_up   = has_higher_highs_and_lows(struct)
    trend_down = has_lower_lows_and_highs(struct)

    is_retest_long  = retest_of_breakout(candles, index, swing_high, cur_atr)
    is_retest_short = retest_of_breakdown(candles, index, swing_low, cur_atr)

    bull_engulf = is_bullish_engulfing(cur, prev)
    bear_engulf = is_bearish_engulfing(cur, prev)

    near_ema21 = (abs(cur.low - ema_mid) < cur_atr * 0.4
                  or abs(prev.low - ema_mid) < cur_atr * 0.4)
    near_ema21_short = (abs(cur.high - ema_mid) < cur_atr * 0.4
                        or abs(prev.high - ema_mid) < cur_atr * 0.4)

    recent_range = max(c.high for c in struct[-8:]) - min(c.low for c in struct[-8:])
    prior_range = (
        max(c.high for c in struct[-16:-8]) - min(c.low for c in struct[-16:-8])
        if len(struct) >= 16 else recent_range
    )
    is_consolidation = recent_range < prior_range * 0.6

    # -- LONG PATTERNS ---------------------------------------------------------

    # Pattern 1: Breakout -- clean close above swing high with volume
    breakout_long = (
        cur.close > cur.open
        and vol_r >= 1.20
        and body_r >= 0.95
        and cur.close > swing_high
        and prev.close <= swing_high
        and clean_up
    )

    # Pattern 2: Retest -- pullback to broken level
    retest_long = (
        cur.close > cur.open
        and vol_r >= 1.05
        and body_r >= 0.80
        and is_retest_long
        and strong_bull
    )

    # Pattern 3: EMA pullback bounce in uptrend
    ema_pullback_long = (
        trend == "up"
        and bull_stack
        and near_ema21
        and cur.close > cur.open
        and cur.close > ema_mid
        and strong_bull
        and vol_r >= 0.90
        and body_r >= 0.75
    )

    # Pattern 4: Bullish engulfing at support level
    engulfing_long = (
        bull_engulf
        and (cur.close >= val or cur.close >= poc * 0.998)
        and vol_r >= 1.05
        and body_r >= 1.0
    )

    # Pattern 5: Momentum continuation after consolidation
    momentum_long = (
        is_consolidation
        and cur.close > cur.open
        and cur.close > max(c.high for c in struct[-8:-1])
        and vol_r >= 1.30
        and body_r >= 1.0
        and bull_stack
    )

    # Pattern 6: Trend-follow continuation
    trend_follow_long = (
        trend != "down"
        and bull_stack
        and cur.close > cur.open
        and cur.close > ema_fast
        and close_pos >= 0.55
        and vol_r >= 0.85
        and body_r >= 0.60
        and 45 <= rsi <= 72
        and not exhaustion_bull
        and adx >= 22
    )

    # Pattern 7: Mean-reversion bounce at oversold extreme
    mean_rev_long = (
        rsi <= 32
        and cur.close <= val
        and cur.close > cur.open
        and close_pos >= 0.55
        and vol_r >= 0.80
    )

    # Long gate: standard patterns require trend alignment + POC proximity
    long_gate = (
        trend != "down"
        and rsi >= rsi_long_min
        and rsi <= rsi_long_max
        and not exhaustion_bull
        and cur.close >= poc * 0.995
        and (breakout_long or retest_long or ema_pullback_long
             or engulfing_long or momentum_long or trend_follow_long)
    )
    # Mean-reversion bypasses the trend and POC gate (it buys the dip)
    long_gate = long_gate or (
        mean_rev_long
        and rsi >= rsi_long_min
        and not exhaustion_bull
    )

    if long_gate:
        if breakout_long:
            pattern = "breakout"
        elif retest_long:
            pattern = "retest"
        elif ema_pullback_long:
            pattern = "ema_pullback"
        elif engulfing_long:
            pattern = "engulfing"
        elif momentum_long:
            pattern = "momentum"
        elif mean_rev_long:
            pattern = "mean_reversion"
        else:
            pattern = "trend_follow"

        s = 0
        if trend == "up":         s += 3
        if trend == "neutral":    s += 1
        if above_200:             s += 2
        if bull_stack:            s += 2
        if trend_up:              s += 2
        if is_retest_long:        s += 2
        if pattern == "breakout": s += 1
        if vol_r >= 1.40:         s += 1
        if vol_r >= 2.00:         s += 1
        if body_r >= 1.20:        s += 1
        if cur.close > vah:       s += 1
        if strong_bull:           s += 1
        if no_up_wick:            s += 1
        if rsi >= 50:             s += 1
        if prev_bull:             s += 1
        if adx >= 25:             s += 1
        if cur.low > prev2.low:   s += 1

        if s >= min_score:
            return "long", s, rsi, trend, pattern

    # -- SHORT PATTERNS --------------------------------------------------------

    breakout_short = (
        cur.close < cur.open
        and vol_r >= 1.20
        and body_r >= 0.95
        and cur.close < swing_low
        and prev.close >= swing_low
        and clean_dn
    )

    retest_short = (
        cur.close < cur.open
        and vol_r >= 1.05
        and body_r >= 0.80
        and is_retest_short
        and strong_bear
    )

    ema_pullback_short = (
        trend == "down"
        and bear_stack
        and near_ema21_short
        and cur.close < cur.open
        and cur.close < ema_mid
        and strong_bear
        and vol_r >= 0.90
        and body_r >= 0.75
    )

    engulfing_short = (
        bear_engulf
        and (cur.close <= vah or cur.close <= poc * 1.002)
        and vol_r >= 1.05
        and body_r >= 1.0
    )

    momentum_short = (
        is_consolidation
        and cur.close < cur.open
        and cur.close < min(c.low for c in struct[-8:-1])
        and vol_r >= 1.30
        and body_r >= 1.0
        and bear_stack
    )

    trend_follow_short = (
        trend != "up"
        and bear_stack
        and cur.close < cur.open
        and cur.close < ema_fast
        and close_pos <= 0.45
        and vol_r >= 0.85
        and body_r >= 0.60
        and 28 <= rsi <= 55
        and not exhaustion_bear
        and adx >= 22
    )

    mean_rev_short = (
        rsi >= 68
        and cur.close >= vah
        and cur.close < cur.open
        and close_pos <= 0.45
        and vol_r >= 0.80
    )

    short_gate = (
        enable_shorts
        and trend != "up"
        and rsi <= rsi_short_max
        and rsi >= rsi_short_min
        and not exhaustion_bear
        and cur.close <= poc * 1.005
        and (breakout_short or retest_short or ema_pullback_short
             or engulfing_short or momentum_short or trend_follow_short)
    )
    short_gate = short_gate or (
        enable_shorts
        and mean_rev_short
        and rsi <= rsi_short_max
        and not exhaustion_bear
    )

    if short_gate:
        if breakout_short:
            pattern = "breakout"
        elif retest_short:
            pattern = "retest"
        elif ema_pullback_short:
            pattern = "ema_pullback"
        elif engulfing_short:
            pattern = "engulfing"
        elif momentum_short:
            pattern = "momentum"
        elif mean_rev_short:
            pattern = "mean_reversion"
        else:
            pattern = "trend_follow"

        s = 0
        if trend == "down":       s += 3
        if trend == "neutral":    s += 1
        if not above_200:         s += 2
        if bear_stack:            s += 2
        if trend_down:            s += 2
        if is_retest_short:       s += 2
        if pattern == "breakout": s += 1
        if vol_r >= 1.40:         s += 1
        if vol_r >= 2.00:         s += 1
        if body_r >= 1.20:        s += 1
        if cur.close < val:       s += 1
        if strong_bear:           s += 1
        if no_lo_wick:            s += 1
        if rsi <= 50:             s += 1
        if prev_bear:             s += 1
        if adx >= 25:             s += 1
        if cur.high < prev2.high: s += 1

        if s >= min_score:
            return "short", s, rsi, trend, pattern

    return None


# -- Backtester ----------------------------------------------------------------

def backtest(
    candles: List[Candle],
    htf_candles: List[Candle],
    initial_balance: float,
    risk_pct: float              = 0.25,
    margin_per_trade: Optional[float] = None,
    leverage: float              = 10.0,
    tp_atr_mult: float           = 0.8,
    sl_atr_mult: float           = 0.6,
    atr_period: int              = 14,
    fee_rate: float              = 0.0004,
    be_trigger_atr: float        = 0.20,
    trail_activation_atr: float  = 0.4,
    trail_distance_atr: float    = 0.15,
    trail_min_profit_atr: float  = 0.20,
    max_hold_candles: int        = 10,
    max_consecutive_losses: int  = 5,
    cooldown_candles: int        = 2,
    score_scaling: bool          = True,
    min_score: int               = 5,
    rsi_long_min: float          = 32.0,
    rsi_long_max: float          = 82.0,
    rsi_short_max: float         = 68.0,
    rsi_short_min: float         = 18.0,
    vol_spike_max: float         = 6.0,
    chop_max: float              = 65.0,
    session_filter: bool         = False,
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

        direction, score, rsi_val, htf_tr, pattern = sig
        entry_c     = candles[index]
        entry_price = entry_c.close

        cur_atr = calc_atr(
            candles[max(0, index - atr_period - 1) : index], atr_period,
        )
        if cur_atr <= 0:
            index += 1
            continue

        # -- Position sizing ---------------------------------------------------
        if margin_per_trade is not None:
            base_margin = min(balance, max(0.50, margin_per_trade))
        else:
            base_margin = max(0.50, balance * risk_pct)

        if score_scaling:
            if score >= 14:
                margin_trade = base_margin * 2.0
            elif score >= 10:
                margin_trade = base_margin * 1.5
            else:
                margin_trade = base_margin
            margin_trade = min(balance * 0.35, margin_trade)
        else:
            margin_trade = base_margin

        notional     = margin_trade * leverage
        qty          = notional / entry_price
        fees         = notional * fee_rate * 2.0
        fee_per_unit = fees / qty if qty > 0 else 0

        # -- Exit levels -------------------------------------------------------
        if direction == "long":
            tp_price = entry_price + tp_atr_mult * cur_atr
            sl_price = entry_price - sl_atr_mult * cur_atr
            be_price = entry_price + fee_per_unit
        else:
            tp_price = entry_price - tp_atr_mult * cur_atr
            sl_price = entry_price + sl_atr_mult * cur_atr
            be_price = entry_price - fee_per_unit

        peak_fav     = entry_price
        trail_sl: Optional[float] = None
        be_activated = False
        exit_price   = candles[-1].close
        exit_reason  = "end_of_data"
        exit_index   = len(candles) - 1
        max_exit     = min(index + max_hold_candles, len(candles) - 1)

        for ei in range(index + 1, max_exit + 1):
            c       = candles[ei]
            is_last = (ei == max_exit)

            if direction == "long":
                if c.high > peak_fav:
                    peak_fav = c.high
                gain_atr = (peak_fav - entry_price) / cur_atr

                # Breakeven trigger: move SL to entry + fees
                if not be_activated and gain_atr >= be_trigger_atr:
                    be_activated = True
                    sl_price = max(sl_price, be_price)

                # Trailing stop
                if gain_atr >= trail_activation_atr:
                    profit_floor = entry_price + trail_min_profit_atr * cur_atr
                    cand = max(
                        peak_fav - trail_distance_atr * cur_atr,
                        profit_floor,
                    )
                    if trail_sl is None or cand > trail_sl:
                        trail_sl = cand

                eff_sl = max(sl_price, trail_sl) if trail_sl else sl_price

                if c.high >= tp_price:
                    exit_price  = tp_price
                    exit_reason = "take_profit"
                    exit_index  = ei
                    break
                elif c.low <= eff_sl:
                    exit_price = eff_sl
                    if be_activated and trail_sl is None:
                        exit_reason = "breakeven"
                    elif trail_sl is not None:
                        exit_reason = "trailing_stop"
                    else:
                        exit_reason = "stop_loss"
                    exit_index = ei
                    break
                elif is_last:
                    exit_price  = c.close
                    exit_reason = "time_exit"
                    exit_index  = ei
                    break
            else:  # short
                if c.low < peak_fav:
                    peak_fav = c.low
                gain_atr = (entry_price - peak_fav) / cur_atr

                if not be_activated and gain_atr >= be_trigger_atr:
                    be_activated = True
                    sl_price = min(sl_price, be_price)

                if gain_atr >= trail_activation_atr:
                    profit_floor = entry_price - trail_min_profit_atr * cur_atr
                    cand = min(
                        peak_fav + trail_distance_atr * cur_atr,
                        profit_floor,
                    )
                    if trail_sl is None or cand < trail_sl:
                        trail_sl = cand

                eff_sl = min(sl_price, trail_sl) if trail_sl else sl_price

                if c.low <= tp_price:
                    exit_price  = tp_price
                    exit_reason = "take_profit"
                    exit_index  = ei
                    break
                elif c.high >= eff_sl:
                    exit_price = eff_sl
                    if be_activated and trail_sl is None:
                        exit_reason = "breakeven"
                    elif trail_sl is not None:
                        exit_reason = "trailing_stop"
                    else:
                        exit_reason = "stop_loss"
                    exit_index = ei
                    break
                elif is_last:
                    exit_price  = c.close
                    exit_reason = "time_exit"
                    exit_index  = ei
                    break

        # PnL
        if direction == "long":
            gross = (exit_price - entry_price) * qty
        else:
            gross = (entry_price - exit_price) * qty
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
            pattern_type=pattern,
            tp_price=round(tp_price, 4),
            sl_price=round(sl_price, 4),
            atr_at_entry=round(cur_atr, 4),
            rsi_at_entry=round(rsi_val, 2),
            htf_trend=htf_tr,
        ))
        index = exit_index + 1

    # -- Summary statistics ----------------------------------------------------
    wins   = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl <= 0]
    longs  = [t for t in trades if t.side == "long"]
    shorts = [t for t in trades if t.side == "short"]

    by_reason: Dict[str, int] = {}
    for t in trades:
        by_reason[t.exit_reason] = by_reason.get(t.exit_reason, 0) + 1

    by_pattern: Dict[str, int] = {}
    for t in trades:
        by_pattern[t.pattern_type] = by_pattern.get(t.pattern_type, 0) + 1

    tw = sum(t.net_pnl  for t in wins)
    tl = sum(-t.net_pnl for t in losses)

    def wr_str(ts: List[Trade]) -> str:
        if not ts:
            return "n/a"
        w = sum(1 for t in ts if t.net_pnl > 0)
        return f"{w}/{len(ts)} ({w / len(ts) * 100:.0f}%)"

    months = max(
        (candles[-1].close_time - candles[100].close_time)
        / (1000 * 3600 * 24 * 30.44),
        1,
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
        "pattern_breakdown":  by_pattern,
        "avg_win":            round(fmean(t.net_pnl for t in wins) if wins else 0, 4),
        "avg_loss":           round(fmean(-t.net_pnl for t in losses) if losses else 0, 4),
        "profit_factor":      round(tw / tl, 3) if tl else float("inf"),
        "win_amount":         round(tw, 4),
        "loss_amount":        round(tl, 4),
        "net_profit":         round(tw - tl, 4),
        "net_profit_pm":      round((tw - tl) / months, 4),
        "final_balance":      round(balance, 4),
        "starting_balance":   round(initial_balance, 4),
        "return_pct":         round(
            (balance - initial_balance) / initial_balance * 100, 2,
        ),
        "monthly_return_pct": round(
            (balance - initial_balance) / initial_balance * 100 / months, 2,
        ),
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


def print_report(s: dict, symbol: str, interval: str,
                 candles: List[Candle], sizing_mode: str = "") -> None:
    W = 60
    print("=" * W)
    print("  Backtest Report v12 -- Scalp + Breakeven Stop")
    print("=" * W)
    print(f"  Symbol          : {symbol}")
    print(f"  Interval        : {interval} (HTF: 1h)")
    print(f"  Candles         : {len(candles)}")
    print(f"  Data start      : {iso_ts(candles[0].open_time)}")
    print(f"  Data end        : {iso_ts(candles[-1].close_time)}")
    if sizing_mode:
        print(f"  Sizing mode     : {sizing_mode}")
    print("-" * W)
    tpm      = s["trades_per_month"]
    tpm_icon = "+" if tpm >= 50 else ("~" if tpm >= 20 else "-")
    print(f"  Trades total    : {s['trade_count']}  (~{tpm}/month) [{tpm_icon}]")
    print(f"  Wins            : {s['win_count']}  (${s['win_amount']:.4f})")
    print(f"  Losses          : {s['loss_count']}  (${s['loss_amount']:.4f})")
    wr       = s["win_rate_pct"]
    wr_icon  = "+" if wr >= 60 else ("~" if wr >= 45 else "-")
    print(f"  Win rate        : {wr:.1f}%  [{wr_icon}]")
    print(f"  Long  WR        : {s['long_wr']}")
    print(f"  Short WR        : {s['short_wr']}")

    patterns = s.get("pattern_breakdown", {})
    if patterns:
        pat_str = " / ".join(f"{k}: {v}" for k, v in sorted(patterns.items()))
        print(f"  Patterns        : {pat_str}")

    exits = s["exit_breakdown"]
    tp = exits.get("take_profit",   0)
    tr = exits.get("trailing_stop", 0)
    be = exits.get("breakeven",     0)
    sl = exits.get("stop_loss",     0)
    te = exits.get("time_exit",     0)
    print(f"  Exits TP/Trail/BE/SL/Time: {tp}/{tr}/{be}/{sl}/{te}")
    print(f"  Avg win         : ${s['avg_win']:.4f}")
    print(f"  Avg loss        : ${s['avg_loss']:.4f}")
    rr = s["avg_win"] / s["avg_loss"] if s["avg_loss"] else 0
    rr_icon  = "+" if rr >= 1.5 else ("~" if rr >= 1.0 else "-")
    print(f"  Actual R:R      : {rr:.2f}:1  [{rr_icon}]")
    pf       = s["profit_factor"]
    pf_icon  = "+" if pf >= 1.5 else ("~" if pf >= 1.1 else "-")
    print(f"  Profit factor   : {pf:.3f}  [{pf_icon}]")
    print(f"  Max drawdown    : {s['max_drawdown_pct']:.1f}%")
    print("-" * W)
    goal_ok = "+" if s["net_profit_pm"] >= 10.0 else (
        "~" if s["net_profit_pm"] >= 3.0 else "-"
    )
    print(f"  Net profit      : ${s['net_profit']:.4f}  total")
    print(f"  Avg / month     : ${s['net_profit_pm']:.4f}  [{goal_ok}] (goal: $10.00)")
    print(f"  Monthly return  : {s['monthly_return_pct']:.2f}%")
    print(f"  Total return    : {s['return_pct']:.2f}%")
    print(f"  Final balance   : ${s['final_balance']:.4f}")
    print("=" * W)

    if rr < 1.0:
        print("  NOTE: R:R low  -> try --tp-atr-mult 1.0 --sl-atr-mult 0.5")
    if wr < 55:
        print("  NOTE: WR low   -> try --min-score 7 --chop-max 58")
    if tp == 0:
        print("  NOTE: TP=0     -> try --tp-atr-mult 0.6 --max-hold-candles 16")
    if te > s["trade_count"] * 0.2:
        print("  NOTE: Many time exits -> try --max-hold-candles 16")
    if s["net_profit_pm"] >= 10.0:
        print("  GOAL REACHED -- consider cross-validating on BTCUSDT / ETHUSDT")
    elif s["net_profit_pm"] >= 5.0:
        print("  CLOSE TO GOAL -- try 30-35% risk-pct for more compounding")
    else:
        gap = max(0.0, 10.0 - s["net_profit_pm"])
        print(f"  Gap to goal : ${gap:.4f}/month still needed")


def main() -> None:
    p = argparse.ArgumentParser(
        description="PA + VP backtester v12 -- scalp, breakeven stop, compound sizing",
    )
    p.add_argument("--symbol",                   default="SOLUSDT")
    p.add_argument("--interval",                 default="15m")
    p.add_argument("--htf-interval",             default="1h")
    p.add_argument("--limit",         type=int,  default=8000)
    p.add_argument("--initial-balance",          type=float, default=10.0)
    p.add_argument("--risk-pct",                 type=float, default=0.25,
                   help="Fraction of balance per trade (compound sizing, default 0.25)")
    p.add_argument("--margin-per-trade",         type=float, default=0,
                   help="Fixed margin per trade. 0 = use compound sizing (--risk-pct)")
    p.add_argument("--leverage",                 type=float, default=10.0)
    p.add_argument("--tp-atr-mult",              type=float, default=0.8)
    p.add_argument("--sl-atr-mult",              type=float, default=0.6)
    p.add_argument("--atr-period",               type=int,   default=14)
    p.add_argument("--fee-rate",                 type=float, default=0.0004)
    p.add_argument("--be-trigger-atr",           type=float, default=0.20,
                   help="ATR gain to move SL to breakeven (entry + fees)")
    p.add_argument("--trail-activation-atr",     type=float, default=0.40)
    p.add_argument("--trail-distance-atr",       type=float, default=0.15)
    p.add_argument("--trail-min-profit-atr",     type=float, default=0.20,
                   help="Minimum ATR profit to lock once trailing activates")
    p.add_argument("--max-hold-candles",         type=int,   default=10)
    p.add_argument("--max-consecutive-losses",   type=int,   default=5)
    p.add_argument("--cooldown-candles",         type=int,   default=2)
    p.add_argument("--no-score-scaling",         action="store_true",
                   help="Disable score-based position scaling")
    p.add_argument("--min-score",                type=int,   default=5)
    p.add_argument("--rsi-long-min",             type=float, default=32.0)
    p.add_argument("--rsi-long-max",             type=float, default=82.0)
    p.add_argument("--rsi-short-max",            type=float, default=68.0)
    p.add_argument("--rsi-short-min",            type=float, default=18.0)
    p.add_argument("--vol-spike-max",            type=float, default=6.0)
    p.add_argument("--chop-max",                 type=float, default=65.0)
    p.add_argument("--session-filter",           action="store_true",
                   help="Enable session filter (skip 01-04 UTC)")
    p.add_argument("--no-shorts",                action="store_true")
    p.add_argument("--use-futures-api",          action="store_true",
                   help="Use futures API (may be blocked in some regions)")
    p.add_argument("--trades-output",            default="trades_v12.csv")
    args = p.parse_args()

    use_futures = args.use_futures_api
    session_filter = args.session_filter

    print(f"  Fetching {args.limit} x {args.interval} candles...")
    candles = fetch_klines(args.symbol, args.interval, args.limit,
                           use_futures=use_futures)

    htf_limit = max(800, args.limit // 4 + 300)
    print(f"  Fetching {htf_limit} x {args.htf_interval} candles (HTF)...")
    htf_candles = fetch_klines(args.symbol, args.htf_interval, htf_limit,
                               use_futures=use_futures)

    print(f"  LTF candles : {len(candles)}")
    print(f"  HTF candles : {len(htf_candles)}")
    print(f"  Session flt : {'ON (skip 01-04 UTC)' if session_filter else 'OFF'}")
    print(f"  Shorts      : {'disabled' if args.no_shorts else 'enabled'}")

    use_fixed = args.margin_per_trade and args.margin_per_trade > 0
    margin_val = args.margin_per_trade if use_fixed else None
    if use_fixed:
        sizing_mode = f"${args.margin_per_trade:.2f} fixed/trade @ {args.leverage:.0f}x"
    else:
        sizing_mode = f"{args.risk_pct * 100:.0f}% of balance/trade @ {args.leverage:.0f}x"
    score_note = " + score scaling" if not args.no_score_scaling else ""
    sizing_mode += score_note
    print(f"  Sizing      : {sizing_mode}")
    print(f"  TP          : {args.tp_atr_mult} ATR")
    print(f"  SL          : {args.sl_atr_mult} ATR  (BE trigger: {args.be_trigger_atr} ATR)")
    print(
        f"  Trail       : activate {args.trail_activation_atr} ATR, "
        f"dist {args.trail_distance_atr} ATR, lock {args.trail_min_profit_atr} ATR"
    )
    print(f"  Max hold    : {args.max_hold_candles} candles")

    summary, trades = backtest(
        candles=candles,
        htf_candles=htf_candles,
        initial_balance=args.initial_balance,
        risk_pct=args.risk_pct,
        margin_per_trade=margin_val,
        leverage=args.leverage,
        tp_atr_mult=args.tp_atr_mult,
        sl_atr_mult=args.sl_atr_mult,
        atr_period=args.atr_period,
        fee_rate=args.fee_rate,
        be_trigger_atr=args.be_trigger_atr,
        trail_activation_atr=args.trail_activation_atr,
        trail_distance_atr=args.trail_distance_atr,
        trail_min_profit_atr=args.trail_min_profit_atr,
        max_hold_candles=args.max_hold_candles,
        max_consecutive_losses=args.max_consecutive_losses,
        cooldown_candles=args.cooldown_candles,
        score_scaling=not args.no_score_scaling,
        min_score=args.min_score,
        rsi_long_min=args.rsi_long_min,
        rsi_long_max=args.rsi_long_max,
        rsi_short_max=args.rsi_short_max,
        rsi_short_min=args.rsi_short_min,
        vol_spike_max=args.vol_spike_max,
        chop_max=args.chop_max,
        session_filter=session_filter,
        enable_shorts=not args.no_shorts,
    )

    write_trades_csv(args.trades_output, trades)
    print(f"  Trade log   : {args.trades_output}")
    print_report(summary, args.symbol, args.interval, candles, sizing_mode)


if __name__ == "__main__":
    main()
