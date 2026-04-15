"""Trading engine for executing trades on Binance Futures."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.config import BinanceConfig, TradingConfig
from app.data_fetcher import fetch_klines
from app.indicators import Candle, compute_features
from app.ml_model import TradingModel

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    id: str
    symbol: str
    side: str  # LONG or SHORT
    entry_time: str
    exit_time: Optional[str] = None
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    quantity: float = 0.0
    margin: float = 0.0
    leverage: int = 20
    tp_price: float = 0.0
    sl_price: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    status: str = "OPEN"  # OPEN, WIN, LOSS, CLOSED
    confidence: float = 0.0
    exit_reason: str = ""
    is_custom: bool = False
    order_id: Optional[str] = None


class BinanceClient:
    """HTTP client for Binance Futures API."""

    def __init__(self, config: BinanceConfig) -> None:
        self.config = config
        self.base_url = config.base_url

    def _sign(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params["timestamp"] = int(time.time() * 1000)
        query_string = urllib.parse.urlencode(params)
        signature = hmac.new(
            self.config.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        params["signature"] = signature
        return params

    def _request(self, method: str, endpoint: str, params: Optional[Dict] = None, signed: bool = True) -> Any:
        if params is None:
            params = {}
        if signed:
            params = self._sign(params)

        url = f"{self.base_url}{endpoint}"
        if method in ("GET", "DELETE"):
            if params:
                url += "?" + urllib.parse.urlencode(params)
            data = None
        else:
            data = urllib.parse.urlencode(params).encode("utf-8")

        headers = {
            "X-MBX-APIKEY": self.config.api_key,
            "Content-Type": "application/x-www-form-urlencoded",
        }

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            logger.error(f"Binance API error {e.code}: {error_body}")
            raise RuntimeError(f"Binance API error {e.code}: {error_body}") from e

    def set_leverage(self, symbol: str, leverage: int) -> Dict:
        return self._request("POST", "/fapi/v1/leverage", {
            "symbol": symbol,
            "leverage": leverage,
        })

    def set_margin_type(self, symbol: str, margin_type: str = "ISOLATED") -> Dict:
        try:
            return self._request("POST", "/fapi/v1/marginType", {
                "symbol": symbol,
                "marginType": margin_type,
            })
        except RuntimeError as e:
            if "No need to change margin type" in str(e):
                return {"msg": "already set"}
            raise

    def place_market_order(self, symbol: str, side: str, quantity: float) -> Dict:
        return self._request("POST", "/fapi/v1/order", {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": f"{quantity:.8f}".rstrip("0").rstrip("."),
        })

    def place_limit_order(self, symbol: str, side: str, quantity: float, price: float) -> Dict:
        return self._request("POST", "/fapi/v1/order", {
            "symbol": symbol,
            "side": side,
            "type": "LIMIT",
            "timeInForce": "GTC",
            "quantity": f"{quantity:.8f}".rstrip("0").rstrip("."),
            "price": f"{price:.8f}".rstrip("0").rstrip("."),
        })

    def place_stop_market(self, symbol: str, side: str, stop_price: float, close_position: bool = True) -> Dict:
        params: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": "STOP_MARKET",
            "stopPrice": f"{stop_price:.8f}".rstrip("0").rstrip("."),
        }
        if close_position:
            params["closePosition"] = "true"
        return self._request("POST", "/fapi/v1/order", params)

    def place_take_profit_market(self, symbol: str, side: str, stop_price: float, close_position: bool = True) -> Dict:
        params: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": "TAKE_PROFIT_MARKET",
            "stopPrice": f"{stop_price:.8f}".rstrip("0").rstrip("."),
        }
        if close_position:
            params["closePosition"] = "true"
        return self._request("POST", "/fapi/v1/order", params)

    def cancel_all_orders(self, symbol: str) -> Dict:
        return self._request("DELETE", "/fapi/v1/allOpenOrders", {
            "symbol": symbol,
        })

    def get_account(self) -> Dict:
        return self._request("GET", "/fapi/v2/account")

    def get_positions(self) -> List[Dict]:
        account = self.get_account()
        return [p for p in account.get("positions", []) if float(p.get("positionAmt", 0)) != 0]

    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        info = self._request("GET", "/fapi/v1/exchangeInfo", signed=False)
        for s in info.get("symbols", []):
            if s["symbol"] == symbol:
                return s
        return None

    def get_price(self, symbol: str) -> float:
        result = self._request("GET", "/fapi/v1/ticker/price", {"symbol": symbol}, signed=False)
        return float(result["price"])


class TradingEngine:
    """Main trading engine that coordinates ML predictions and order execution."""

    def __init__(
        self,
        model: TradingModel,
        trading_config: TradingConfig,
        binance_config: BinanceConfig,
    ) -> None:
        self.model = model
        self.config = trading_config
        self.binance_config = binance_config
        self.client = BinanceClient(binance_config)
        self.trades: List[TradeRecord] = []
        self.is_running = False
        self._task: Optional[asyncio.Task] = None
        self._trade_counter = 0
        self.logs: List[str] = []

    def _log(self, msg: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{ts}] {msg}"
        self.logs.append(entry)
        if len(self.logs) > 1000:
            self.logs = self.logs[-500:]
        logger.info(msg)

    def _get_quantity(self, symbol: str, margin: float, leverage: int, price: float) -> float:
        """Calculate order quantity based on margin, leverage, and current price."""
        notional = margin * leverage
        qty = notional / price

        # Get symbol precision
        try:
            info = self.client.get_symbol_info(symbol)
            if info:
                for f in info.get("filters", []):
                    if f["filterType"] == "LOT_SIZE":
                        step = float(f["stepSize"])
                        precision = len(str(step).rstrip("0").split(".")[-1]) if "." in str(step) else 0
                        qty = round(qty - (qty % step), precision)
                        break
        except Exception as e:
            self._log(f"Warning: Could not get symbol info for {symbol}: {e}")

        return max(qty, 0.0)

    def _get_price_precision(self, symbol: str) -> int:
        """Get price precision for a symbol."""
        try:
            info = self.client.get_symbol_info(symbol)
            if info:
                for f in info.get("filters", []):
                    if f["filterType"] == "PRICE_FILTER":
                        tick = float(f["tickSize"])
                        return len(str(tick).rstrip("0").split(".")[-1]) if "." in str(tick) else 0
        except Exception:
            pass
        return 4

    def open_trade(
        self,
        symbol: str,
        side: str,
        confidence: float = 0.0,
        is_custom: bool = False,
    ) -> Optional[TradeRecord]:
        """Open a new trade."""
        try:
            # Set leverage and margin type
            try:
                self.client.set_margin_type(symbol, "ISOLATED")
            except Exception:
                pass
            self.client.set_leverage(symbol, self.config.leverage)

            price = self.client.get_price(symbol)
            qty = self._get_quantity(symbol, self.config.margin_per_trade, self.config.leverage, price)

            if qty <= 0:
                self._log(f"Quantity too small for {symbol} at price {price}")
                return None

            # Calculate TP and SL prices
            tp_move = self.config.tp_pct / self.config.leverage / 100.0
            sl_move = self.config.sl_pct / self.config.leverage / 100.0

            price_prec = self._get_price_precision(symbol)

            if side == "LONG":
                order_side = "BUY"
                close_side = "SELL"
                tp_price = round(price * (1 + tp_move), price_prec)
                sl_price = round(price * (1 - sl_move), price_prec)
            else:
                order_side = "SELL"
                close_side = "BUY"
                tp_price = round(price * (1 - tp_move), price_prec)
                sl_price = round(price * (1 + sl_move), price_prec)

            # Place market order
            order = self.client.place_market_order(symbol, order_side, qty)
            order_id = str(order.get("orderId", ""))

            # Place TP and SL
            try:
                self.client.place_take_profit_market(symbol, close_side, tp_price)
            except Exception as e:
                self._log(f"Warning: Could not place TP for {symbol}: {e}")

            try:
                self.client.place_stop_market(symbol, close_side, sl_price)
            except Exception as e:
                self._log(f"Warning: Could not place SL for {symbol}: {e}")

            self._trade_counter += 1
            trade = TradeRecord(
                id=f"T{self._trade_counter:06d}",
                symbol=symbol,
                side=side,
                entry_time=datetime.now(timezone.utc).isoformat(),
                entry_price=price,
                quantity=qty,
                margin=self.config.margin_per_trade,
                leverage=self.config.leverage,
                tp_price=tp_price,
                sl_price=sl_price,
                confidence=confidence,
                is_custom=is_custom,
                order_id=order_id,
            )
            self.trades.append(trade)
            self._log(f"OPENED {side} {symbol} @ {price:.6f} qty={qty:.6f} TP={tp_price:.6f} SL={sl_price:.6f}")
            return trade

        except Exception as e:
            self._log(f"ERROR opening trade {side} {symbol}: {e}")
            return None

    def close_trade(self, trade_id: str, reason: str = "manual") -> bool:
        """Close a specific trade."""
        trade = next((t for t in self.trades if t.id == trade_id and t.status == "OPEN"), None)
        if not trade:
            return False

        try:
            # Cancel all open orders for this symbol
            self.client.cancel_all_orders(trade.symbol)

            # Close position
            price = self.client.get_price(trade.symbol)
            close_side = "SELL" if trade.side == "LONG" else "BUY"
            self.client.place_market_order(trade.symbol, close_side, trade.quantity)

            # Calculate PnL
            if trade.side == "LONG":
                pnl = (price - trade.entry_price) / trade.entry_price * trade.margin * trade.leverage
            else:
                pnl = (trade.entry_price - price) / trade.entry_price * trade.margin * trade.leverage

            trade.exit_price = price
            trade.exit_time = datetime.now(timezone.utc).isoformat()
            trade.pnl = round(pnl, 4)
            trade.pnl_pct = round(pnl / trade.margin * 100, 2)
            trade.status = "WIN" if pnl > 0 else "LOSS"
            trade.exit_reason = reason

            self._log(f"CLOSED {trade.symbol} {trade.side} @ {price:.6f} PnL={pnl:.4f} ({reason})")
            return True
        except Exception as e:
            self._log(f"ERROR closing trade {trade_id}: {e}")
            return False

    def check_open_trades(self) -> None:
        """Check and update status of open trades."""
        for trade in self.trades:
            if trade.status != "OPEN":
                continue
            try:
                price = self.client.get_price(trade.symbol)

                if trade.side == "LONG":
                    if price >= trade.tp_price:
                        pnl = (trade.tp_price - trade.entry_price) / trade.entry_price * trade.margin * trade.leverage
                        trade.exit_price = trade.tp_price
                        trade.status = "WIN"
                        trade.exit_reason = "TP"
                    elif price <= trade.sl_price:
                        pnl = (trade.sl_price - trade.entry_price) / trade.entry_price * trade.margin * trade.leverage
                        trade.exit_price = trade.sl_price
                        trade.status = "LOSS"
                        trade.exit_reason = "SL"
                    else:
                        continue
                else:
                    if price <= trade.tp_price:
                        pnl = (trade.entry_price - trade.tp_price) / trade.entry_price * trade.margin * trade.leverage
                        trade.exit_price = trade.tp_price
                        trade.status = "WIN"
                        trade.exit_reason = "TP"
                    elif price >= trade.sl_price:
                        pnl = (trade.entry_price - trade.sl_price) / trade.entry_price * trade.margin * trade.leverage
                        trade.exit_price = trade.sl_price
                        trade.status = "LOSS"
                        trade.exit_reason = "SL"
                    else:
                        continue

                trade.exit_time = datetime.now(timezone.utc).isoformat()
                trade.pnl = round(pnl, 4)
                trade.pnl_pct = round(pnl / trade.margin * 100, 2)
                self._log(f"TRADE {trade.status}: {trade.symbol} {trade.side} PnL={trade.pnl:.4f} ({trade.exit_reason})")

            except Exception as e:
                self._log(f"Error checking {trade.symbol}: {e}")

    def scan_and_trade(self) -> List[TradeRecord]:
        """Scan all pairs for ML signals and execute trades."""
        new_trades: List[TradeRecord] = []
        open_count = sum(1 for t in self.trades if t.status == "OPEN")

        if open_count >= self.config.max_open_trades:
            self._log(f"Max open trades ({self.config.max_open_trades}) reached, skipping scan")
            return new_trades

        for pair in self.config.pairs:
            if open_count >= self.config.max_open_trades:
                break

            # Skip if already have an open trade on this pair
            if any(t.symbol == pair and t.status == "OPEN" for t in self.trades):
                continue

            try:
                candles = fetch_klines(pair, self.config.timeframe, 300)
                htf_candles = fetch_klines(pair, self.config.htf_timeframe, 300)

                if len(candles) < 100:
                    continue

                features = compute_features(candles, len(candles) - 1, htf_candles)
                if features is None:
                    continue

                direction, confidence = self.model.predict(features)

                if direction == "SKIP" or confidence < self.config.min_signal_confidence:
                    continue

                self._log(f"SIGNAL: {pair} {direction} confidence={confidence:.4f}")
                trade = self.open_trade(pair, direction, confidence)
                if trade:
                    new_trades.append(trade)
                    open_count += 1

            except Exception as e:
                self._log(f"Error scanning {pair}: {e}")

        return new_trades

    async def run_loop(self) -> None:
        """Main trading loop."""
        self._log("Trading bot started")
        self.is_running = True

        while self.is_running:
            try:
                # Check open trades
                self.check_open_trades()

                # Scan for new opportunities
                self.scan_and_trade()

                # Wait for next candle (15 minutes)
                self._log("Waiting for next scan cycle (15 minutes)...")
                for _ in range(90):  # 90 * 10s = 15 minutes
                    if not self.is_running:
                        break
                    await asyncio.sleep(10)

            except Exception as e:
                self._log(f"Error in trading loop: {e}")
                await asyncio.sleep(60)

        self._log("Trading bot stopped")

    def start(self) -> None:
        """Start the trading bot in background."""
        if self.is_running:
            return
        self._task = asyncio.create_task(self.run_loop())

    def stop(self) -> None:
        """Stop the trading bot."""
        self.is_running = False
        if self._task:
            self._task.cancel()
            self._task = None
        self._log("Bot stop requested")

    def get_stats(self) -> Dict:
        """Get trading statistics."""
        closed = [t for t in self.trades if t.status in ("WIN", "LOSS")]
        wins = [t for t in closed if t.status == "WIN"]
        losses = [t for t in closed if t.status == "LOSS"]
        open_trades = [t for t in self.trades if t.status == "OPEN"]

        total_pnl = sum(t.pnl for t in closed)
        win_rate = len(wins) / len(closed) * 100 if closed else 0.0

        return {
            "is_running": self.is_running,
            "total_trades": len(closed),
            "open_trades": len(open_trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(win_rate, 2),
            "total_pnl": round(total_pnl, 4),
            "avg_pnl": round(total_pnl / len(closed), 4) if closed else 0.0,
        }
