"""Fetch historical kline data from Binance."""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Dict, List, Optional

from app.indicators import Candle

KLINES_URL = "https://data-api.binance.vision/api/v3/klines"
KLINES_URL_FUTURES = "https://fapi.binance.com/fapi/v1/klines"
MAX_PER_FETCH = 1000


def fetch_klines(
    symbol: str,
    interval: str,
    limit: int,
    use_futures: bool = False,
) -> List[Candle]:
    """Fetch kline/candlestick data from Binance."""
    symbol = symbol.upper()
    base_url = KLINES_URL_FUTURES if use_futures else KLINES_URL
    chunks: List[List[Candle]] = []
    remaining = limit
    end_time: Optional[int] = None

    while remaining > 0:
        batch = min(remaining, MAX_PER_FETCH)
        params: Dict[str, object] = {
            "symbol": symbol,
            "interval": interval,
            "limit": batch,
        }
        if end_time is not None:
            params["endTime"] = end_time
        url = f"{base_url}?{urllib.parse.urlencode(params)}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "TradingBot/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                payload = json.load(r)
        except Exception:
            if use_futures:
                url = f"{KLINES_URL}?{urllib.parse.urlencode(params)}"
                req = urllib.request.Request(url, headers={"User-Agent": "TradingBot/1.0"})
                with urllib.request.urlopen(req, timeout=30) as r:
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
        end_time = chunk[0].open_time - 1
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
