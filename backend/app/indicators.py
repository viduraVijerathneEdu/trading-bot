"""Technical indicators for feature engineering."""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple


@dataclass
class Candle:
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int


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


def ema_series(values: List[float], period: int) -> List[float]:
    """Return full EMA series."""
    if not values:
        return []
    result: List[float] = []
    k = 2.0 / (period + 1)
    e = values[0]
    for i, v in enumerate(values):
        if i < period:
            e = fmean(values[: i + 1])
        else:
            e = v * k + e * (1 - k)
        result.append(e)
    return result


def calc_atr(candles: List[Candle], period: int = 14) -> float:
    if len(candles) < 2:
        return 0.0
    trs: List[float] = []
    for i in range(1, len(candles)):
        pc = candles[i - 1].close
        trs.append(max(
            candles[i].high - candles[i].low,
            abs(candles[i].high - pc),
            abs(candles[i].low - pc),
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


def rsi_series(candles: List[Candle], period: int = 14) -> List[float]:
    """Return RSI value for each candle position."""
    result: List[float] = []
    for i in range(len(candles)):
        subset = candles[max(0, i - period): i + 1]
        result.append(calc_rsi(subset, period))
    return result


def choppiness_index(candles: List[Candle], period: int = 14) -> float:
    if len(candles) < period + 2:
        return 61.8
    window = candles[-(period + 1):]
    atr_sum = sum(
        max(window[i].high - window[i].low,
            abs(window[i].high - window[i - 1].close),
            abs(window[i].low - window[i - 1].close))
        for i in range(1, len(window))
    )
    hi = max(c.high for c in window)
    lo = min(c.low for c in window)
    if math.isclose(hi, lo):
        return 100.0
    return 100.0 * math.log10(atr_sum / (hi - lo)) / math.log10(period)


def calc_adx(candles: List[Candle], period: int = 14) -> float:
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


def build_volume_profile(candles: List[Candle], bins: int = 36) -> Tuple[float, float, float]:
    """Returns (POC, VAL, VAH)."""
    lo = min(c.low for c in candles)
    hi = max(c.high for c in candles)
    if math.isclose(lo, hi):
        return lo, lo, hi
    step = (hi - lo) / bins
    vols = [0.0] * bins
    centers = [lo + (i + 0.5) * step for i in range(bins)]
    for c in candles:
        tp = (c.high + c.low + c.close) / 3.0
        idx = min(bins - 1, max(0, int((tp - lo) / step)))
        vols[idx] += c.volume
    total = sum(vols)
    poc_i = max(range(bins), key=lambda i: vols[i])
    incl: set = {poc_i}
    cov = vols[poc_i]
    li, ri = poc_i - 1, poc_i + 1
    while total and cov / total < 0.70 and (li >= 0 or ri < bins):
        lv = vols[li] if li >= 0 else -1.0
        rv = vols[ri] if ri < bins else -1.0
        if rv >= lv:
            if ri < bins:
                incl.add(ri)
                cov += vols[ri]
                ri += 1
            else:
                incl.add(li)
                cov += vols[li]
                li -= 1
        else:
            if li >= 0:
                incl.add(li)
                cov += vols[li]
                li -= 1
            else:
                incl.add(ri)
                cov += vols[ri]
                ri += 1
    return centers[poc_i], min(centers[i] for i in incl), max(centers[i] for i in incl)


def calc_macd(candles: List[Candle], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
    """Returns (MACD line, signal line, histogram)."""
    closes = [c.close for c in candles]
    if len(closes) < slow + signal:
        return 0.0, 0.0, 0.0
    fast_ema = ema_series(closes, fast)
    slow_ema = ema_series(closes, slow)
    macd_line = [f - s for f, s in zip(fast_ema, slow_ema)]
    signal_line = ema_series(macd_line[slow - 1:], signal)
    if not signal_line:
        return 0.0, 0.0, 0.0
    macd_val = macd_line[-1]
    sig_val = signal_line[-1]
    return macd_val, sig_val, macd_val - sig_val


def calc_bollinger(candles: List[Candle], period: int = 20, std_dev: float = 2.0) -> Tuple[float, float, float]:
    """Returns (upper, middle, lower) band."""
    if len(candles) < period:
        close = candles[-1].close if candles else 0.0
        return close, close, close
    closes = [c.close for c in candles[-period:]]
    middle = statistics.fmean(closes)
    sd = statistics.stdev(closes) if len(closes) > 1 else 0.0
    return middle + std_dev * sd, middle, middle - std_dev * sd


def calc_stochastic(candles: List[Candle], k_period: int = 14, d_period: int = 3) -> Tuple[float, float]:
    """Returns (%K, %D)."""
    if len(candles) < k_period:
        return 50.0, 50.0
    window = candles[-k_period:]
    highest = max(c.high for c in window)
    lowest = min(c.low for c in window)
    if math.isclose(highest, lowest):
        return 50.0, 50.0
    k_val = 100.0 * (candles[-1].close - lowest) / (highest - lowest)
    k_values: List[float] = []
    for i in range(max(0, len(candles) - d_period), len(candles)):
        w = candles[max(0, i - k_period + 1): i + 1]
        h = max(c.high for c in w)
        l_val = min(c.low for c in w)
        if math.isclose(h, l_val):
            k_values.append(50.0)
        else:
            k_values.append(100.0 * (candles[i].close - l_val) / (h - l_val))
    d_val = fmean(k_values) if k_values else k_val
    return k_val, d_val


def calc_obv(candles: List[Candle]) -> float:
    """On-Balance Volume."""
    if len(candles) < 2:
        return 0.0
    obv = 0.0
    for i in range(1, len(candles)):
        if candles[i].close > candles[i - 1].close:
            obv += candles[i].volume
        elif candles[i].close < candles[i - 1].close:
            obv -= candles[i].volume
    return obv


def htf_analysis(htf_candles: List[Candle], timestamp_ms: int) -> Tuple[str, bool, float]:
    """Returns (trend_direction, above_200_ema, trend_strength)."""
    rel = [c for c in htf_candles if c.close_time < timestamp_ms]
    if len(rel) < 205:
        return "neutral", True, 0.0
    closes = [c.close for c in rel]
    e21 = ema_of(closes, 21)
    e21_p = ema_of(closes[:-1], 21)
    e55 = ema_of(closes, 55)
    e200 = ema_of(closes, 200)
    last = rel[-1].close
    above_200 = last > e200
    rising = e21 > e21_p * 1.0002
    falling = e21 < e21_p * 0.9998
    strength = min(1.0, abs(e21 - e55) / last * 100) if last > 0 else 0
    if last > e21 and e21 > e55 * 0.998 and rising:
        return "up", above_200, strength
    if last < e21 and e21 < e55 * 1.002 and falling:
        return "down", above_200, strength
    return "neutral", above_200, strength


def compute_features(candles: List[Candle], index: int, htf_candles: Optional[List[Candle]] = None) -> Optional[dict]:
    """Compute ML features for a given candle index."""
    if index < 65:
        return None

    cur = candles[index]
    prev = candles[index - 1]
    window = candles[max(0, index - 60): index]
    short_window = candles[max(0, index - 15): index + 1]

    closes = [c.close for c in candles[max(0, index - 50): index + 1]]
    if len(closes) < 10:
        return None

    ema9 = ema_of(closes, 9)
    ema21 = ema_of(closes, 21)
    ema50 = ema_of(closes, min(50, len(closes)))

    rsi = calc_rsi(candles[max(0, index - 15): index + 1], 14)
    atr = calc_atr(candles[max(0, index - 15): index], 14)
    adx = calc_adx(candles[max(0, index - 20): index + 1], 14)
    ci = choppiness_index(short_window, 14)

    avg_vol = fmean(c.volume for c in window) if window else 1.0
    vol_ratio = cur.volume / avg_vol if avg_vol > 0 else 0.0

    poc, val, vah = build_volume_profile(window) if len(window) >= 5 else (cur.close, cur.close, cur.close)

    macd_val, macd_sig, macd_hist = calc_macd(candles[max(0, index - 40): index + 1])
    bb_upper, bb_middle, bb_lower = calc_bollinger(candles[max(0, index - 25): index + 1])
    stoch_k, stoch_d = calc_stochastic(candles[max(0, index - 20): index + 1])
    obv = calc_obv(candles[max(0, index - 20): index + 1])

    rng = max(cur.high - cur.low, 1e-9)
    close_pos = (cur.close - cur.low) / rng
    body = abs(cur.close - cur.open)
    avg_body = fmean(abs(c.close - c.open) for c in window) if window else 1.0
    body_ratio = body / avg_body if avg_body > 0 else 0.0
    upper_wick = cur.high - max(cur.open, cur.close)
    lower_wick = min(cur.open, cur.close) - cur.low
    wick_ratio = (upper_wick - lower_wick) / rng if rng > 0 else 0.0

    # Price relative to EMAs
    price_to_ema9 = (cur.close - ema9) / cur.close * 100 if cur.close > 0 else 0
    price_to_ema21 = (cur.close - ema21) / cur.close * 100 if cur.close > 0 else 0
    price_to_ema50 = (cur.close - ema50) / cur.close * 100 if cur.close > 0 else 0

    # EMA alignment
    ema_bull_stack = 1.0 if ema9 > ema21 > ema50 else 0.0
    ema_bear_stack = 1.0 if ema9 < ema21 < ema50 else 0.0

    # Bollinger band position
    bb_width = (bb_upper - bb_lower) / bb_middle if bb_middle > 0 else 0
    bb_pos = (cur.close - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5

    # Price relative to POC
    price_to_poc = (cur.close - poc) / cur.close * 100 if cur.close > 0 else 0

    # Momentum
    returns_1 = (cur.close - prev.close) / prev.close * 100 if prev.close > 0 else 0
    if index >= 5:
        returns_5 = (cur.close - candles[index - 5].close) / candles[index - 5].close * 100
    else:
        returns_5 = 0
    if index >= 10:
        returns_10 = (cur.close - candles[index - 10].close) / candles[index - 10].close * 100
    else:
        returns_10 = 0

    # Consecutive candle direction
    consec_bull = 0
    consec_bear = 0
    for j in range(index, max(index - 10, -1), -1):
        if candles[j].close > candles[j].open:
            consec_bull += 1
        else:
            break
    for j in range(index, max(index - 10, -1), -1):
        if candles[j].close < candles[j].open:
            consec_bear += 1
        else:
            break

    # ATR normalized
    atr_pct = atr / cur.close * 100 if cur.close > 0 else 0

    # HTF features
    htf_trend_val = 0.0
    htf_above_200 = 1.0
    htf_strength = 0.0
    if htf_candles and len(htf_candles) > 205:
        trend_dir, above200, strength = htf_analysis(htf_candles, cur.open_time)
        htf_trend_val = 1.0 if trend_dir == "up" else (-1.0 if trend_dir == "down" else 0.0)
        htf_above_200 = 1.0 if above200 else 0.0
        htf_strength = strength

    features = {
        "rsi": rsi,
        "atr_pct": atr_pct,
        "adx": adx,
        "choppiness": ci,
        "vol_ratio": vol_ratio,
        "close_pos": close_pos,
        "body_ratio": body_ratio,
        "wick_ratio": wick_ratio,
        "price_to_ema9": price_to_ema9,
        "price_to_ema21": price_to_ema21,
        "price_to_ema50": price_to_ema50,
        "ema_bull_stack": ema_bull_stack,
        "ema_bear_stack": ema_bear_stack,
        "macd_hist": macd_hist,
        "macd_val": macd_val,
        "bb_width": bb_width,
        "bb_pos": bb_pos,
        "stoch_k": stoch_k,
        "stoch_d": stoch_d,
        "price_to_poc": price_to_poc,
        "returns_1": returns_1,
        "returns_5": returns_5,
        "returns_10": returns_10,
        "consec_bull": float(consec_bull),
        "consec_bear": float(consec_bear),
        "htf_trend": htf_trend_val,
        "htf_above_200": htf_above_200,
        "htf_strength": htf_strength,
    }
    return features


FEATURE_NAMES = [
    "rsi", "atr_pct", "adx", "choppiness", "vol_ratio",
    "close_pos", "body_ratio", "wick_ratio",
    "price_to_ema9", "price_to_ema21", "price_to_ema50",
    "ema_bull_stack", "ema_bear_stack",
    "macd_hist", "macd_val", "bb_width", "bb_pos",
    "stoch_k", "stoch_d", "price_to_poc",
    "returns_1", "returns_5", "returns_10",
    "consec_bull", "consec_bear",
    "htf_trend", "htf_above_200", "htf_strength",
]
