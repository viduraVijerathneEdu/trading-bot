#!/usr/bin/env python3
"""
Real-time Trading Bot Server — v12 Dashboard Edition
Runs the v12 signal logic on live Binance data and streams
trade events to a browser dashboard via WebSocket.

Usage:
    python3 trading_server.py [--symbol SOLUSDT] [--interval 15m] [--port 8765]

Then open dashboard.html in your browser.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Set, Tuple

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
except ImportError:
    print("Run:  pip install websockets")
    raise

# ── Constants ─────────────────────────────────────────────────────────────────
KLINES_URL  = "https://data-api.binance.vision/api/v3/klines"
BINANCE_WS  = "wss://stream.binance.com:9443/ws"
MAX_FETCH   = 1000
WARMUP_BARS = 200          # bars needed before signals can fire


# ── Data classes ──────────────────────────────────────────────────────────────
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
    id: int
    side: str
    entry_time: int
    exit_time: Optional[int]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    margin: float
    leverage: float
    tp_price: float
    sl_price: float
    atr_at_entry: float
    rsi_at_entry: float
    htf_trend: str
    signal_score: int
    pattern_type: str
    gross_pnl: float
    fees: float
    net_pnl: float
    result: str           # "open" | "win" | "loss"
    exit_reason: str


def iso_ts(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def now_ms() -> int:
    return int(time.time() * 1000)


# ── Indicators (identical to v12) ─────────────────────────────────────────────
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
            abs(candles[i].low - pc),
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
    ag, al = fmean(gains), fmean(losses)
    return 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)


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
    plus_dm, minus_dm, trs = [], [], []
    for i in range(1, len(candles)):
        hd = candles[i].high - candles[i - 1].high
        ld = candles[i - 1].low - candles[i].low
        plus_dm.append(hd if hd > ld and hd > 0 else 0)
        minus_dm.append(ld if ld > hd and ld > 0 else 0)
        trs.append(max(
            candles[i].high - candles[i].low,
            abs(candles[i].high - candles[i - 1].close),
            abs(candles[i].low - candles[i - 1].close),
        ))
    n = min(period, len(trs))
    atr_val = fmean(trs[-n:])
    if atr_val == 0:
        return 25.0
    pdi = 100.0 * fmean(plus_dm[-n:]) / atr_val
    mdi = 100.0 * fmean(minus_dm[-n:]) / atr_val
    s = pdi + mdi
    return 0.0 if s == 0 else 100.0 * abs(pdi - mdi) / s


def build_volume_profile(candles: List[Candle], bins: int = 36) -> Tuple[float, float, float]:
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
            if ri < bins: incl.add(ri); cov += vols[ri]; ri += 1
            else: incl.add(li); cov += vols[li]; li -= 1
        else:
            if li >= 0: incl.add(li); cov += vols[li]; li -= 1
            else: incl.add(ri); cov += vols[ri]; ri += 1
    return centers[poc_i], min(centers[i] for i in incl), max(centers[i] for i in incl)


def htf_analysis(htf: List[Candle], ts_ms: int) -> Tuple[str, bool, float]:
    rel = [c for c in htf if c.close_time < ts_ms]
    if len(rel) < 205:
        return "neutral", True, 0.0
    closes = [c.close for c in rel]
    e21 = ema_of(closes, 21)
    e21p = ema_of(closes[:-1], 21)
    e55 = ema_of(closes, 55)
    e200 = ema_of(closes, 200)
    last = rel[-1].close
    above_200 = last > e200
    rising = e21 > e21p * 1.0002
    falling = e21 < e21p * 0.9998
    strength = min(1.0, abs(e21 - e55) / last * 100) if last > 0 else 0
    if last > e21 and e21 > e55 * 0.998 and rising:
        return "up", above_200, strength
    if last < e21 and e21 < e55 * 1.002 and falling:
        return "down", above_200, strength
    return "neutral", above_200, strength


def has_higher_highs_and_lows(candles: List[Candle], n: int = 4) -> bool:
    if len(candles) < n * 2: return False
    highs = [c.high for c in candles[-n:]]
    lows = [c.low for c in candles[-n:]]
    return (all(highs[i] > highs[i-1] for i in range(1, len(highs))) or
            all(lows[i] > lows[i-1] for i in range(1, len(lows))))


def has_lower_lows_and_highs(candles: List[Candle], n: int = 4) -> bool:
    if len(candles) < n * 2: return False
    highs = [c.high for c in candles[-n:]]
    lows = [c.low for c in candles[-n:]]
    return (all(lows[i] < lows[i-1] for i in range(1, len(lows))) or
            all(highs[i] < highs[i-1] for i in range(1, len(highs))))


def retest_of_breakout(candles, index, swing_high, atr, lookback=6):
    start = max(0, index - lookback)
    if not any(c.close > swing_high for c in candles[start:index]): return False
    return abs(candles[index].low - swing_high) < atr * 0.6


def retest_of_breakdown(candles, index, swing_low, atr, lookback=6):
    start = max(0, index - lookback)
    if not any(c.close < swing_low for c in candles[start:index]): return False
    return abs(candles[index].high - swing_low) < atr * 0.6


def is_bullish_engulfing(cur, prev):
    if prev.close >= prev.open or cur.close <= cur.open: return False
    return cur.close > prev.open and cur.open <= prev.close


def is_bearish_engulfing(cur, prev):
    if prev.close <= prev.open or cur.close >= cur.open: return False
    return cur.close < prev.open and cur.open >= prev.close


def generate_signal(
    candles, index, htf_candles,
    min_score=5, rsi_long_min=32.0, rsi_long_max=82.0,
    rsi_short_max=68.0, rsi_short_min=18.0,
    vol_spike_max=6.0, chop_max=65.0, enable_shorts=True,
):
    PROFILE_LB, STRUCT_LB = 60, 24
    FAST_P, MID_P, SLOW_P = 9, 21, 50
    RSI_P = 14

    if index < max(PROFILE_LB, STRUCT_LB, SLOW_P, RSI_P) + 5:
        return None

    cur = candles[index]
    prev = candles[index - 1]
    prev2 = candles[index - 2]

    ci = choppiness_index(candles[index - 15: index + 1], 14)
    if ci > chop_max:
        return None

    trend, above_200, trend_strength = htf_analysis(htf_candles, cur.open_time)
    adx = calc_adx(candles[max(0, index - 20): index + 1], 14)

    window = candles[index - PROFILE_LB: index]
    avg_vol = fmean(c.volume for c in window)
    vol_r = cur.volume / avg_vol if avg_vol > 0 else 0.0
    if vol_r > vol_spike_max:
        return None

    rsi = calc_rsi(candles[index - RSI_P - 1: index + 1], RSI_P)
    closes = [c.close for c in candles[index - SLOW_P: index + 1]]
    ema_fast = ema_of(closes, FAST_P)
    ema_mid = ema_of(closes, MID_P)
    ema_slow = ema_of(closes, SLOW_P)
    bull_stack = ema_fast > ema_mid * 0.999 and ema_mid > ema_slow * 0.999
    bear_stack = ema_fast < ema_mid * 1.001 and ema_mid < ema_slow * 1.001

    poc, val, vah = build_volume_profile(window)
    struct = candles[index - STRUCT_LB: index]
    swing_high = max(c.high for c in struct[:-1])
    swing_low = min(c.low for c in struct[:-1])

    cur_atr = calc_atr(candles[max(0, index - 15): index], 14)
    if cur_atr <= 0:
        return None

    avg_body = fmean(abs(c.close - c.open) for c in window)
    body = abs(cur.close - cur.open)
    body_r = body / avg_body if avg_body > 0 else 0.0
    rng = max(cur.high - cur.low, 1e-9)
    close_pos = (cur.close - cur.low) / rng
    strong_bull = close_pos >= 0.60
    strong_bear = close_pos <= 0.40
    upper_wick = cur.high - max(cur.open, cur.close)
    lower_wick = min(cur.open, cur.close) - cur.low
    no_up_wick = upper_wick < body * 0.40
    no_lo_wick = lower_wick < body * 0.40
    prev_body = abs(prev.close - prev.open)
    prev_body_r = prev_body / avg_body if avg_body > 0 else 0.0
    prev_bull = prev.close > prev.open
    prev_bear = prev.close < prev.open
    exhaustion_bull = prev.close < prev.open and prev_body_r > 2.5
    exhaustion_bear = prev.close > prev.open and prev_body_r > 2.5
    clean_up = all(c.close <= swing_high for c in struct[-6:-1])
    clean_dn = all(c.close >= swing_low for c in struct[-6:-1])
    trend_up = has_higher_highs_and_lows(struct)
    trend_down = has_lower_lows_and_highs(struct)
    is_retest_long = retest_of_breakout(candles, index, swing_high, cur_atr)
    is_retest_short = retest_of_breakdown(candles, index, swing_low, cur_atr)
    bull_engulf = is_bullish_engulfing(cur, prev)
    bear_engulf = is_bearish_engulfing(cur, prev)
    near_ema21 = (abs(cur.low - ema_mid) < cur_atr * 0.4 or abs(prev.low - ema_mid) < cur_atr * 0.4)
    near_ema21_short = (abs(cur.high - ema_mid) < cur_atr * 0.4 or abs(prev.high - ema_mid) < cur_atr * 0.4)
    recent_range = max(c.high for c in struct[-8:]) - min(c.low for c in struct[-8:])
    prior_range = (max(c.high for c in struct[-16:-8]) - min(c.low for c in struct[-16:-8])
                   if len(struct) >= 16 else recent_range)
    is_consolidation = recent_range < prior_range * 0.6

    breakout_long = (cur.close > cur.open and vol_r >= 1.20 and body_r >= 0.95 and
                     cur.close > swing_high and prev.close <= swing_high and clean_up)
    retest_long = (cur.close > cur.open and vol_r >= 1.05 and body_r >= 0.80 and is_retest_long and strong_bull)
    ema_pullback_long = (trend == "up" and bull_stack and near_ema21 and cur.close > cur.open and
                         cur.close > ema_mid and strong_bull and vol_r >= 0.90 and body_r >= 0.75)
    engulfing_long = (bull_engulf and (cur.close >= val or cur.close >= poc * 0.998) and
                      vol_r >= 1.05 and body_r >= 1.0)
    momentum_long = (is_consolidation and cur.close > cur.open and
                     cur.close > max(c.high for c in struct[-8:-1]) and
                     vol_r >= 1.30 and body_r >= 1.0 and bull_stack)
    trend_follow_long = (trend != "down" and bull_stack and cur.close > cur.open and
                         cur.close > ema_fast and close_pos >= 0.55 and vol_r >= 0.85 and
                         body_r >= 0.60 and 45 <= rsi <= 72 and not exhaustion_bull and adx >= 22)
    mean_rev_long = (rsi <= 32 and cur.close <= val and cur.close > cur.open and
                     close_pos >= 0.55 and vol_r >= 0.80)

    long_gate = (trend != "down" and rsi >= rsi_long_min and rsi <= rsi_long_max and
                 not exhaustion_bull and cur.close >= poc * 0.995 and
                 (breakout_long or retest_long or ema_pullback_long or engulfing_long or
                  momentum_long or trend_follow_long))
    long_gate = long_gate or (mean_rev_long and rsi >= rsi_long_min and not exhaustion_bull)

    if long_gate:
        pattern = ("breakout" if breakout_long else "retest" if retest_long else
                   "ema_pullback" if ema_pullback_long else "engulfing" if engulfing_long else
                   "momentum" if momentum_long else "mean_reversion" if mean_rev_long else "trend_follow")
        s = 0
        if trend == "up": s += 3
        if trend == "neutral": s += 1
        if above_200: s += 2
        if bull_stack: s += 2
        if trend_up: s += 2
        if is_retest_long: s += 2
        if pattern == "breakout": s += 1
        if vol_r >= 1.40: s += 1
        if vol_r >= 2.00: s += 1
        if body_r >= 1.20: s += 1
        if cur.close > vah: s += 1
        if strong_bull: s += 1
        if no_up_wick: s += 1
        if rsi >= 50: s += 1
        if prev_bull: s += 1
        if adx >= 25: s += 1
        if cur.low > prev2.low: s += 1
        if s >= min_score:
            return "long", s, rsi, trend, pattern, cur_atr, poc, val, vah

    breakout_short = (cur.close < cur.open and vol_r >= 1.20 and body_r >= 0.95 and
                      cur.close < swing_low and prev.close >= swing_low and clean_dn)
    retest_short = (cur.close < cur.open and vol_r >= 1.05 and body_r >= 0.80 and is_retest_short and strong_bear)
    ema_pullback_short = (trend == "down" and bear_stack and near_ema21_short and cur.close < cur.open and
                          cur.close < ema_mid and strong_bear and vol_r >= 0.90 and body_r >= 0.75)
    engulfing_short = (bear_engulf and (cur.close <= vah or cur.close <= poc * 1.002) and
                       vol_r >= 1.05 and body_r >= 1.0)
    momentum_short = (is_consolidation and cur.close < cur.open and
                      cur.close < min(c.low for c in struct[-8:-1]) and
                      vol_r >= 1.30 and body_r >= 1.0 and bear_stack)
    trend_follow_short = (trend != "up" and bear_stack and cur.close < cur.open and
                          cur.close < ema_fast and close_pos <= 0.45 and vol_r >= 0.85 and
                          body_r >= 0.60 and 28 <= rsi <= 55 and not exhaustion_bear and adx >= 22)
    mean_rev_short = (rsi >= 68 and cur.close >= vah and cur.close < cur.open and
                      close_pos <= 0.45 and vol_r >= 0.80)

    short_gate = (enable_shorts and trend != "up" and rsi <= rsi_short_max and rsi >= rsi_short_min and
                  not exhaustion_bear and cur.close <= poc * 1.005 and
                  (breakout_short or retest_short or ema_pullback_short or
                   engulfing_short or momentum_short or trend_follow_short))
    short_gate = short_gate or (enable_shorts and mean_rev_short and rsi <= rsi_short_max and not exhaustion_bear)

    if short_gate:
        pattern = ("breakout" if breakout_short else "retest" if retest_short else
                   "ema_pullback" if ema_pullback_short else "engulfing" if engulfing_short else
                   "momentum" if momentum_short else "mean_reversion" if mean_rev_short else "trend_follow")
        s = 0
        if trend == "down": s += 3
        if trend == "neutral": s += 1
        if not above_200: s += 2
        if bear_stack: s += 2
        if trend_down: s += 2
        if is_retest_short: s += 2
        if pattern == "breakout": s += 1
        if vol_r >= 1.40: s += 1
        if vol_r >= 2.00: s += 1
        if body_r >= 1.20: s += 1
        if cur.close < val: s += 1
        if strong_bear: s += 1
        if no_lo_wick: s += 1
        if rsi <= 50: s += 1
        if prev_bear: s += 1
        if adx >= 25: s += 1
        if cur.high < prev2.high: s += 1
        if s >= min_score:
            return "short", s, rsi, trend, pattern, cur_atr, poc, val, vah

    return None


# ── REST data fetch ────────────────────────────────────────────────────────────
def fetch_klines(symbol: str, interval: str, limit: int) -> List[Candle]:
    symbol = symbol.upper()
    chunks: List[List[Candle]] = []
    remaining = limit
    end_time = None
    while remaining > 0:
        batch = min(remaining, MAX_FETCH)
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
        end_time = chunk[0].open_time - 1
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


# ── Trading Engine ────────────────────────────────────────────────────────────
class TradingEngine:
    def __init__(self, symbol: str, interval: str, htf_interval: str,
                 initial_balance: float, risk_pct: float, leverage: float,
                 tp_atr: float, sl_atr: float, be_trigger_atr: float,
                 trail_activation_atr: float, trail_distance_atr: float,
                 trail_min_profit_atr: float, max_hold_candles: int,
                 min_score: int, enable_shorts: bool, fee_rate: float):
        self.symbol = symbol.upper()
        self.interval = interval
        self.htf_interval = htf_interval
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.risk_pct = risk_pct
        self.leverage = leverage
        self.tp_atr = tp_atr
        self.sl_atr = sl_atr
        self.be_trigger_atr = be_trigger_atr
        self.trail_activation_atr = trail_activation_atr
        self.trail_distance_atr = trail_distance_atr
        self.trail_min_profit_atr = trail_min_profit_atr
        self.max_hold_candles = max_hold_candles
        self.min_score = min_score
        self.enable_shorts = enable_shorts
        self.fee_rate = fee_rate

        self.ltf_candles: List[Candle] = []
        self.htf_candles: List[Candle] = []
        self.open_trade: Optional[Trade] = None
        self.trades: List[Trade] = []
        self.trade_id = 0
        self.hold_count = 0
        self.current_candle: Optional[Candle] = None
        self.current_price: float = 0.0
        self.last_signal_candle: int = 0
        self.be_activated = False
        self.trail_sl: Optional[float] = None
        self.peak_fav: float = 0.0
        self.clients: Set = set()
        self.log: List[str] = []

    def _log(self, msg: str):
        ts = datetime.now(tz=timezone.utc).strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self.log.append(line)
        if len(self.log) > 200:
            self.log = self.log[-200:]
        print(line)

    async def load_history(self):
        self._log(f"Loading {self.symbol} history...")
        self.ltf_candles = fetch_klines(self.symbol, self.interval, 500)
        self.htf_candles = fetch_klines(self.symbol, self.htf_interval, 800)
        self.current_price = self.ltf_candles[-1].close
        self._log(f"Loaded {len(self.ltf_candles)} LTF + {len(self.htf_candles)} HTF candles")

    def _make_state(self) -> dict:
        wins = [t for t in self.trades if t.result == "win"]
        losses = [t for t in self.trades if t.result == "loss"]
        total = len(self.trades)
        net = sum(t.net_pnl for t in self.trades)
        open_pnl = 0.0
        if self.open_trade and self.current_price:
            t = self.open_trade
            if t.side == "long":
                open_pnl = (self.current_price - t.entry_price) * t.quantity - t.fees
            else:
                open_pnl = (t.entry_price - self.current_price) * t.quantity - t.fees

        return {
            "type": "state",
            "symbol": self.symbol,
            "interval": self.interval,
            "current_price": round(self.current_price, 4),
            "balance": round(self.balance + open_pnl, 4),
            "initial_balance": round(self.initial_balance, 4),
            "net_pnl": round(net + open_pnl, 4),
            "return_pct": round((self.balance + open_pnl - self.initial_balance) / self.initial_balance * 100, 2),
            "win_count": len(wins),
            "loss_count": len(losses),
            "total_trades": total,
            "win_rate": round(len(wins) / total * 100, 1) if total else 0,
            "open_trade": asdict(self.open_trade) if self.open_trade else None,
            "open_pnl": round(open_pnl, 4),
            "trades": [asdict(t) for t in self.trades[-50:]],
            "log": self.log[-30:],
            "leverage": self.leverage,
        }

    async def broadcast(self, data: dict):
        if not self.clients:
            return
        msg = json.dumps(data)
        dead = set()
        for ws in self.clients:
            try:
                await ws.send(msg)
            except Exception:
                dead.add(ws)
        self.clients -= dead

    def _check_exit(self, c: Candle) -> Optional[Tuple[str, float]]:
        """Check if open trade should exit. Returns (reason, price) or None."""
        if not self.open_trade:
            return None
        t = self.open_trade
        cur_atr = t.atr_at_entry

        if t.side == "long":
            if c.high > self.peak_fav:
                self.peak_fav = c.high
            gain_atr = (self.peak_fav - t.entry_price) / cur_atr
            if not self.be_activated and gain_atr >= self.be_trigger_atr:
                self.be_activated = True
                fee_pu = t.fees / t.quantity
                t.sl_price = max(t.sl_price, t.entry_price + fee_pu)
                self._log(f"  🔒 Breakeven activated @ {t.sl_price:.4f}")
            if gain_atr >= self.trail_activation_atr:
                pf = t.entry_price + self.trail_min_profit_atr * cur_atr
                cand = max(self.peak_fav - self.trail_distance_atr * cur_atr, pf)
                if self.trail_sl is None or cand > self.trail_sl:
                    self.trail_sl = cand
            eff_sl = max(t.sl_price, self.trail_sl) if self.trail_sl else t.sl_price
            if c.high >= t.tp_price:
                return "take_profit", t.tp_price
            if c.low <= eff_sl:
                reason = ("breakeven" if self.be_activated and self.trail_sl is None
                          else "trailing_stop" if self.trail_sl else "stop_loss")
                return reason, eff_sl
        else:
            if c.low < self.peak_fav:
                self.peak_fav = c.low
            gain_atr = (t.entry_price - self.peak_fav) / cur_atr
            if not self.be_activated and gain_atr >= self.be_trigger_atr:
                self.be_activated = True
                fee_pu = t.fees / t.quantity
                t.sl_price = min(t.sl_price, t.entry_price - fee_pu)
                self._log(f"  🔒 Breakeven activated @ {t.sl_price:.4f}")
            if gain_atr >= self.trail_activation_atr:
                pf = t.entry_price - self.trail_min_profit_atr * cur_atr
                cand = min(self.peak_fav + self.trail_distance_atr * cur_atr, pf)
                if self.trail_sl is None or cand < self.trail_sl:
                    self.trail_sl = cand
            eff_sl = min(t.sl_price, self.trail_sl) if self.trail_sl else t.sl_price
            if c.low <= t.tp_price:
                return "take_profit", t.tp_price
            if c.high >= eff_sl:
                reason = ("breakeven" if self.be_activated and self.trail_sl is None
                          else "trailing_stop" if self.trail_sl else "stop_loss")
                return reason, eff_sl
        return None

    def _close_trade(self, exit_price: float, reason: str, exit_time: int):
        t = self.open_trade
        if t.side == "long":
            gross = (exit_price - t.entry_price) * t.quantity
        else:
            gross = (t.entry_price - exit_price) * t.quantity
        net = gross - t.fees
        t.exit_price = round(exit_price, 6)
        t.exit_time = exit_time
        t.gross_pnl = round(gross, 6)
        t.net_pnl = round(net, 6)
        t.result = "win" if net > 0 else "loss"
        t.exit_reason = reason
        self.balance = max(0.01, self.balance + net)
        emoji = "✅" if net > 0 else "❌"
        self._log(f"{emoji} Trade #{t.id} CLOSED [{reason}] "
                  f"P&L: ${net:+.4f} | Balance: ${self.balance:.4f}")
        self.trades.append(t)
        self.open_trade = None
        self.be_activated = False
        self.trail_sl = None
        self.peak_fav = 0.0
        self.hold_count = 0

    def process_candle(self, c: Candle):
        """Process a closed candle: check exits, check entries."""
        # Update HTF if needed (simple check by interval size)
        self.ltf_candles.append(c)
        if len(self.ltf_candles) > 2000:
            self.ltf_candles = self.ltf_candles[-2000:]

        # Exit check
        if self.open_trade:
            self.hold_count += 1
            exit_info = self._check_exit(c)
            if exit_info:
                reason, price = exit_info
                self._close_trade(price, reason, c.close_time)
            elif self.hold_count >= self.max_hold_candles:
                self._close_trade(c.close, "time_exit", c.close_time)
            return  # one trade at a time

        # Entry check
        if len(self.ltf_candles) < WARMUP_BARS:
            return
        index = len(self.ltf_candles) - 1
        sig = generate_signal(
            self.ltf_candles, index, self.htf_candles,
            min_score=self.min_score, enable_shorts=self.enable_shorts,
        )
        if sig is None:
            return

        direction, score, rsi_val, htf_tr, pattern, cur_atr, poc, val, vah = sig
        entry_price = c.close
        margin = max(0.50, self.balance * self.risk_pct)
        if score >= 14:
            margin = min(self.balance * 0.35, margin * 2.0)
        elif score >= 10:
            margin = min(self.balance * 0.35, margin * 1.5)

        notional = margin * self.leverage
        qty = notional / entry_price
        fees = notional * self.fee_rate * 2.0

        if direction == "long":
            tp = entry_price + self.tp_atr * cur_atr
            sl = entry_price - self.sl_atr * cur_atr
        else:
            tp = entry_price - self.tp_atr * cur_atr
            sl = entry_price + self.sl_atr * cur_atr

        self.trade_id += 1
        self.open_trade = Trade(
            id=self.trade_id,
            side=direction,
            entry_time=c.close_time,
            exit_time=None,
            entry_price=round(entry_price, 6),
            exit_price=None,
            quantity=round(qty, 6),
            margin=round(margin, 4),
            leverage=self.leverage,
            tp_price=round(tp, 6),
            sl_price=round(sl, 6),
            atr_at_entry=round(cur_atr, 6),
            rsi_at_entry=round(rsi_val, 2),
            htf_trend=htf_tr,
            signal_score=score,
            pattern_type=pattern,
            gross_pnl=0.0,
            fees=round(fees, 6),
            net_pnl=0.0,
            result="open",
            exit_reason="",
        )
        self.peak_fav = entry_price
        self.be_activated = False
        self.trail_sl = None
        self.hold_count = 0
        arrow = "🟢 LONG" if direction == "long" else "🔴 SHORT"
        self._log(f"{arrow} Trade #{self.trade_id} | {pattern} | score={score} "
                  f"| entry={entry_price:.4f} TP={tp:.4f} SL={sl:.4f}")

    def process_tick(self, price: float):
        """Update current price and check live exit for open trade."""
        self.current_price = price
        if not self.open_trade:
            return
        t = self.open_trade
        # Live exit check with current tick
        fake_c = Candle(
            open_time=now_ms(), open=price, high=price, low=price,
            close=price, volume=0.0, close_time=now_ms(),
        )
        exit_info = self._check_exit(fake_c)
        if exit_info:
            reason, ep = exit_info
            self._close_trade(ep, reason, now_ms())


# ── WebSocket server ──────────────────────────────────────────────────────────
class Server:
    def __init__(self, engine: TradingEngine, host: str = "0.0.0.0", port: int = 8765):
        self.engine = engine
        self.host = host
        self.port = port

    async def handler(self, ws):
        self.engine.clients.add(ws)
        self.engine._log(f"Dashboard connected ({len(self.engine.clients)} clients)")
        try:
            await ws.send(json.dumps(self.engine._make_state()))
            async for _ in ws:
                pass  # ignore incoming messages
        except Exception:
            pass
        finally:
            self.engine.clients.discard(ws)

    async def stream_binance(self):
        """Connect to Binance WebSocket for live kline + trade streams."""
        stream = f"{self.engine.symbol.lower()}@kline_{self.engine.interval}"
        tick_stream = f"{self.engine.symbol.lower()}@aggTrade"
        url = f"{BINANCE_WS}/{stream}/{tick_stream}"
        self.engine._log(f"Connecting to Binance WS: {url}")
        reconnect_delay = 5
        while True:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    self.engine._log("✅ Binance WS connected")
                    reconnect_delay = 5
                    async for raw in ws:
                        data = json.loads(raw)
                        # Aggregate trade (tick price update)
                        if data.get("e") == "aggTrade":
                            price = float(data["p"])
                            self.engine.process_tick(price)
                            state = self.engine._make_state()
                            await self.engine.broadcast(state)
                        # Kline (candle close)
                        elif data.get("e") == "kline":
                            k = data["k"]
                            if k["x"]:  # candle is closed
                                c = Candle(
                                    open_time=int(k["t"]),
                                    open=float(k["o"]),
                                    high=float(k["h"]),
                                    low=float(k["l"]),
                                    close=float(k["c"]),
                                    volume=float(k["v"]),
                                    close_time=int(k["T"]),
                                )
                                self.engine.process_candle(c)
                                state = self.engine._make_state()
                                await self.engine.broadcast(state)
            except Exception as e:
                self.engine._log(f"⚠️  WS error: {e} — reconnecting in {reconnect_delay}s")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(60, reconnect_delay * 2)

    async def heartbeat(self):
        """Send state every 2 seconds so the dashboard stays fresh."""
        while True:
            await asyncio.sleep(2)
            if self.engine.clients:
                await self.engine.broadcast(self.engine._make_state())

    async def run(self):
        await self.engine.load_history()
        async with websockets.serve(self.handler, self.host, self.port):
            self.engine._log(f"🚀 Dashboard WS server on ws://{self.host}:{self.port}")
            self.engine._log(f"   Open dashboard.html in your browser")
            await asyncio.gather(
                self.stream_binance(),
                self.heartbeat(),
            )


def main():
    p = argparse.ArgumentParser(description="v12 Real-time Trading Dashboard Server")
    p.add_argument("--symbol",              default="SOLUSDT")
    p.add_argument("--interval",            default="15m")
    p.add_argument("--htf-interval",        default="1h")
    p.add_argument("--initial-balance",     type=float, default=10.0)
    p.add_argument("--risk-pct",            type=float, default=0.25)
    p.add_argument("--leverage",            type=float, default=10.0)
    p.add_argument("--tp-atr-mult",         type=float, default=1.2)
    p.add_argument("--sl-atr-mult",         type=float, default=0.6)
    p.add_argument("--be-trigger-atr",      type=float, default=0.20)
    p.add_argument("--trail-activation",    type=float, default=0.40)
    p.add_argument("--trail-distance",      type=float, default=0.15)
    p.add_argument("--trail-min-profit",    type=float, default=0.20)
    p.add_argument("--max-hold-candles",    type=int,   default=10)
    p.add_argument("--min-score",           type=int,   default=5)
    p.add_argument("--fee-rate",            type=float, default=0.0004)
    p.add_argument("--no-shorts",           action="store_true")
    p.add_argument("--port",                type=int,   default=8765)
    p.add_argument("--host",                default="0.0.0.0")
    a = p.parse_args()

    engine = TradingEngine(
        symbol=a.symbol, interval=a.interval, htf_interval=a.htf_interval,
        initial_balance=a.initial_balance, risk_pct=a.risk_pct, leverage=a.leverage,
        tp_atr=a.tp_atr_mult, sl_atr=a.sl_atr_mult, be_trigger_atr=a.be_trigger_atr,
        trail_activation_atr=a.trail_activation, trail_distance_atr=a.trail_distance,
        trail_min_profit_atr=a.trail_min_profit, max_hold_candles=a.max_hold_candles,
        min_score=a.min_score, enable_shorts=not a.no_shorts, fee_rate=a.fee_rate,
    )
    server = Server(engine, host=a.host, port=a.port)
    asyncio.run(server.run())


if __name__ == "__main__":
    main()