"""Backtesting engine for the ML trading model."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from app.config import TradingConfig
from app.data_fetcher import fetch_klines
from app.indicators import FEATURE_NAMES, Candle, calc_atr, compute_features
from app.ml_model import TradingModel

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    symbol: str
    side: str
    entry_index: int
    entry_price: float
    tp_price: float
    sl_price: float
    exit_price: float = 0.0
    exit_index: int = 0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    result: str = ""
    confidence: float = 0.0
    exit_reason: str = ""


@dataclass
class BacktestResult:
    symbol: str
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    max_drawdown: float
    profit_factor: float
    sharpe_ratio: float
    trades: List[Dict]
    equity_curve: List[float]


def run_backtest(
    model: TradingModel,
    config: TradingConfig,
    symbol: str,
    num_candles: int = 3000,
    test_ratio: float = 0.3,
) -> BacktestResult:
    """Run backtest on historical data for a single symbol."""
    logger.info(f"Backtesting {symbol}...")

    candles = fetch_klines(symbol, config.timeframe, num_candles)
    htf_candles = fetch_klines(symbol, config.htf_timeframe, num_candles)

    if len(candles) < 200:
        return BacktestResult(
            symbol=symbol, total_trades=0, wins=0, losses=0,
            win_rate=0, total_pnl=0, avg_pnl=0, max_drawdown=0,
            profit_factor=0, sharpe_ratio=0, trades=[], equity_curve=[],
        )

    # Use last portion for testing
    test_start = int(len(candles) * (1 - test_ratio))
    trades: List[BacktestTrade] = []
    balance = 100.0  # Starting balance
    equity_curve = [balance]
    peak_balance = balance

    tp_move = config.tp_pct / config.leverage / 100.0
    sl_move = config.sl_pct / config.leverage / 100.0

    open_trade: Optional[BacktestTrade] = None
    cooldown = 0

    for i in range(test_start, len(candles) - 1):
        cur = candles[i]

        # Check open trade
        if open_trade is not None:
            next_candle = candles[i]
            hit_tp = False
            hit_sl = False

            if open_trade.side == "LONG":
                if next_candle.high >= open_trade.tp_price:
                    hit_tp = True
                if next_candle.low <= open_trade.sl_price:
                    hit_sl = True
            else:
                if next_candle.low <= open_trade.tp_price:
                    hit_tp = True
                if next_candle.high >= open_trade.sl_price:
                    hit_sl = True

            if hit_sl:
                open_trade.exit_price = open_trade.sl_price
                open_trade.exit_index = i
                open_trade.result = "LOSS"
                open_trade.exit_reason = "SL"
                pnl = -config.margin_per_trade * (config.sl_pct / 100.0)
                open_trade.pnl = round(pnl, 4)
                open_trade.pnl_pct = round(-config.sl_pct, 2)
                balance += pnl
                trades.append(open_trade)
                open_trade = None
                cooldown = 2
            elif hit_tp:
                open_trade.exit_price = open_trade.tp_price
                open_trade.exit_index = i
                open_trade.result = "WIN"
                open_trade.exit_reason = "TP"
                pnl = config.margin_per_trade * (config.tp_pct / 100.0)
                open_trade.pnl = round(pnl, 4)
                open_trade.pnl_pct = round(config.tp_pct, 2)
                balance += pnl
                trades.append(open_trade)
                open_trade = None

            equity_curve.append(balance)
            continue

        if cooldown > 0:
            cooldown -= 1
            equity_curve.append(balance)
            continue

        # Get ML prediction
        features = compute_features(candles, i, htf_candles)
        if features is None:
            equity_curve.append(balance)
            continue

        direction, confidence = model.predict(features)

        if direction == "SKIP" or confidence < config.min_signal_confidence:
            equity_curve.append(balance)
            continue

        entry_price = cur.close

        if direction == "LONG":
            tp_price = entry_price * (1 + tp_move)
            sl_price = entry_price * (1 - sl_move)
        else:
            tp_price = entry_price * (1 - tp_move)
            sl_price = entry_price * (1 + sl_move)

        open_trade = BacktestTrade(
            symbol=symbol,
            side=direction,
            entry_index=i,
            entry_price=entry_price,
            tp_price=tp_price,
            sl_price=sl_price,
            confidence=confidence,
        )

        equity_curve.append(balance)

    # Close any remaining open trade
    if open_trade is not None:
        last_price = candles[-1].close
        if open_trade.side == "LONG":
            pnl = (last_price - open_trade.entry_price) / open_trade.entry_price * config.margin_per_trade * config.leverage
        else:
            pnl = (open_trade.entry_price - last_price) / open_trade.entry_price * config.margin_per_trade * config.leverage
        open_trade.exit_price = last_price
        open_trade.exit_index = len(candles) - 1
        open_trade.pnl = round(pnl, 4)
        open_trade.pnl_pct = round(pnl / config.margin_per_trade * 100, 2)
        open_trade.result = "WIN" if pnl > 0 else "LOSS"
        open_trade.exit_reason = "END"
        balance += pnl
        trades.append(open_trade)

    # Calculate metrics
    wins = [t for t in trades if t.result == "WIN"]
    losses = [t for t in trades if t.result == "LOSS"]
    total_pnl = sum(t.pnl for t in trades)
    win_rate = len(wins) / len(trades) * 100 if trades else 0

    gross_profit = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0

    # Max drawdown
    peak = equity_curve[0]
    max_dd = 0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    # Sharpe ratio (simplified)
    if len(trades) > 1:
        returns = [t.pnl_pct for t in trades]
        avg_ret = np.mean(returns)
        std_ret = np.std(returns)
        sharpe = avg_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0
    else:
        sharpe = 0

    trade_dicts = []
    for t in trades:
        trade_dicts.append({
            "symbol": t.symbol,
            "side": t.side,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "tp_price": t.tp_price,
            "sl_price": t.sl_price,
            "pnl": t.pnl,
            "pnl_pct": t.pnl_pct,
            "result": t.result,
            "confidence": round(t.confidence, 4),
            "exit_reason": t.exit_reason,
        })

    return BacktestResult(
        symbol=symbol,
        total_trades=len(trades),
        wins=len(wins),
        losses=len(losses),
        win_rate=round(win_rate, 2),
        total_pnl=round(total_pnl, 4),
        avg_pnl=round(total_pnl / len(trades), 4) if trades else 0,
        max_drawdown=round(max_dd, 2),
        profit_factor=round(profit_factor, 2),
        sharpe_ratio=round(float(sharpe), 2),
        trades=trade_dicts,
        equity_curve=equity_curve,
    )


def run_full_backtest(
    model: TradingModel,
    config: TradingConfig,
    num_candles: int = 3000,
) -> Dict:
    """Run backtest across all configured pairs."""
    results = {}
    total_trades = 0
    total_wins = 0
    total_losses = 0
    total_pnl = 0.0

    for pair in config.pairs:
        try:
            result = run_backtest(model, config, pair, num_candles)
            results[pair] = {
                "total_trades": result.total_trades,
                "wins": result.wins,
                "losses": result.losses,
                "win_rate": result.win_rate,
                "total_pnl": result.total_pnl,
                "avg_pnl": result.avg_pnl,
                "max_drawdown": result.max_drawdown,
                "profit_factor": result.profit_factor,
                "sharpe_ratio": result.sharpe_ratio,
                "trades": result.trades,
                "equity_curve": result.equity_curve,
            }
            total_trades += result.total_trades
            total_wins += result.wins
            total_losses += result.losses
            total_pnl += result.total_pnl
        except Exception as e:
            logger.error(f"Backtest failed for {pair}: {e}")
            results[pair] = {"error": str(e)}

    overall_win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0

    return {
        "pair_results": results,
        "summary": {
            "total_trades": total_trades,
            "total_wins": total_wins,
            "total_losses": total_losses,
            "overall_win_rate": round(overall_win_rate, 2),
            "total_pnl": round(total_pnl, 4),
            "avg_pnl_per_trade": round(total_pnl / total_trades, 4) if total_trades > 0 else 0,
        },
    }
