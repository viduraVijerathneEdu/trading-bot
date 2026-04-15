"""Configuration for the trading bot."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


TRADING_PAIRS: List[str] = [
    "XRPUSDT", "DOGEUSDT", "ADAUSDT", "SOLUSDT", "SHIBUSDT",
    "PEPEUSDT", "LINKUSDT", "MATICUSDT", "AVAXUSDT", "ARBUSDT",
    "OPUSDT", "SUIUSDT", "APTUSDT", "NEARUSDT", "FTMUSDT",
    "DOTUSDT", "ATOMUSDT",
]


@dataclass
class TradingConfig:
    margin_per_trade: float = 1.0
    leverage: int = 20
    tp_pct: float = 50.0  # take profit percentage of margin
    sl_pct: float = 50.0  # stop loss percentage of margin
    min_signal_confidence: float = 0.60
    pairs: List[str] = field(default_factory=lambda: TRADING_PAIRS.copy())
    timeframe: str = "15m"
    htf_timeframe: str = "1h"
    warmup_bars: int = 300
    max_open_trades: int = 5
    min_trades_per_month: int = 30


@dataclass
class BinanceConfig:
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True
    testnet_base_url: str = "https://testnet.binancefuture.com"
    mainnet_base_url: str = "https://fapi.binance.com"

    @property
    def base_url(self) -> str:
        return self.testnet_base_url if self.testnet else self.mainnet_base_url


def get_binance_config() -> BinanceConfig:
    return BinanceConfig(
        api_key=os.environ.get("BINANCE_API_KEY", ""),
        api_secret=os.environ.get("BINANCE_API_SECRET", ""),
        testnet=os.environ.get("BINANCE_TESTNET", "true").lower() == "true",
    )
