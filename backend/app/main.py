"""Main FastAPI application for the ML Trading Bot."""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.config import BinanceConfig, TradingConfig, TRADING_PAIRS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Lazy imports for heavy ML modules (saves ~200MB at startup)
_TradingModel = None
_TradingEngine = None
_TradeRecord = None
_run_backtest = None
_run_full_backtest = None
_prepare_training_data = None


def _lazy_ml():
    global _TradingModel, _prepare_training_data
    if _TradingModel is None:
        from app.ml_model import TradingModel as _TM, prepare_training_data as _ptd
        _TradingModel = _TM
        _prepare_training_data = _ptd
    return _TradingModel, _prepare_training_data


def _lazy_engine():
    global _TradingEngine, _TradeRecord
    if _TradingEngine is None:
        from app.trading_engine import TradingEngine as _TE, TradeRecord as _TR
        _TradingEngine = _TE
        _TradeRecord = _TR
    return _TradingEngine, _TradeRecord


def _lazy_backtest():
    global _run_backtest, _run_full_backtest
    if _run_backtest is None:
        from app.backtester import run_backtest as _rb, run_full_backtest as _rfb
        _run_backtest = _rb
        _run_full_backtest = _rfb
    return _run_backtest, _run_full_backtest


# Global state
model: Optional[object] = None
engine: Optional[object] = None
training_status: Dict = {"status": "idle", "progress": "", "metrics": {}}


def _get_model():
    global model
    if model is None:
        TradingModel, _ = _lazy_ml()
        model = TradingModel()
        if model.load():
            logger.info("Pre-trained model loaded successfully")
    return model


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Application lifespan. Model loading is deferred to first use to save memory."""
    logger.info("Trading bot API starting up (model loads on first use)")
    yield
    if engine and hasattr(engine, 'is_running') and engine.is_running:
        engine.stop()


app = FastAPI(title="ML Crypto Trading Bot", lifespan=lifespan)

# Disable CORS. Do not remove this for full-stack development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# ── Pydantic Models ──────────────────────────────────────────────────────────


class ConfigUpdate(BaseModel):
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True
    margin_per_trade: float = 1.0
    leverage: int = 20
    tp_pct: float = 50.0
    sl_pct: float = 50.0
    min_signal_confidence: float = 0.60
    max_open_trades: int = 5
    pairs: List[str] = TRADING_PAIRS.copy()


class CustomTradeRequest(BaseModel):
    symbol: str
    side: str  # LONG or SHORT


class TrainRequest(BaseModel):
    pairs: Optional[List[str]] = None
    num_candles: int = 3000


class BacktestRequest(BaseModel):
    symbol: Optional[str] = None
    num_candles: int = 3000
    pairs: Optional[List[str]] = None


# ── Helper ───────────────────────────────────────────────────────────────────

_current_config = ConfigUpdate()


def _get_binance_config() -> BinanceConfig:
    return BinanceConfig(
        api_key=_current_config.api_key or os.environ.get("BINANCE_API_KEY", ""),
        api_secret=_current_config.api_secret or os.environ.get("BINANCE_API_SECRET", ""),
        testnet=_current_config.testnet,
    )


def _get_trading_config() -> TradingConfig:
    return TradingConfig(
        margin_per_trade=_current_config.margin_per_trade,
        leverage=_current_config.leverage,
        tp_pct=_current_config.tp_pct,
        sl_pct=_current_config.sl_pct,
        min_signal_confidence=_current_config.min_signal_confidence,
        max_open_trades=_current_config.max_open_trades,
        pairs=_current_config.pairs,
    )


def _ensure_engine():
    global engine
    if engine is None:
        TradingEngine, _ = _lazy_engine()
        engine = TradingEngine(_get_model(), _get_trading_config(), _get_binance_config())
    return engine


# ── Health ───────────────────────────────────────────────────────────────────


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


# ── Config ───────────────────────────────────────────────────────────────────


@app.get("/api/config")
async def get_config():
    return {
        "testnet": _current_config.testnet,
        "margin_per_trade": _current_config.margin_per_trade,
        "leverage": _current_config.leverage,
        "tp_pct": _current_config.tp_pct,
        "sl_pct": _current_config.sl_pct,
        "min_signal_confidence": _current_config.min_signal_confidence,
        "max_open_trades": _current_config.max_open_trades,
        "pairs": _current_config.pairs,
        "has_api_key": bool(_current_config.api_key or os.environ.get("BINANCE_API_KEY")),
    }


@app.post("/api/config")
async def update_config(config: ConfigUpdate):
    global _current_config, engine
    _current_config = config
    # Recreate engine with new config
    if engine:
        was_running = engine.is_running
        if was_running:
            engine.stop()
        TE, _ = _lazy_engine()
        engine = TE(_get_model(), _get_trading_config(), _get_binance_config())
        if was_running:
            engine.start()
    return {"status": "ok", "config": config.model_dump()}


# ── Model Training ───────────────────────────────────────────────────────────


@app.get("/api/model/status")
async def model_status():
    m = _get_model()
    return {
        "is_trained": m.is_trained,
        "accuracy": m.accuracy,
        "metrics": m.metrics,
        "training_status": training_status,
    }


def _train_model_task(pairs: List[str], num_candles: int) -> None:
    global training_status
    try:
        training_status = {"status": "collecting_data", "progress": "Fetching historical data...", "metrics": {}}
        config = _get_trading_config()
        config.pairs = pairs

        _, prepare_training_data = _lazy_ml()
        X, y = prepare_training_data(pairs, config, num_candles)

        training_status = {"status": "training", "progress": f"Training model on {X.shape[0]} samples...", "metrics": {}}

        m = _get_model()
        metrics = m.train(X, y)
        m.save()

        training_status = {
            "status": "completed",
            "progress": f"Training complete! Accuracy: {m.accuracy:.4f}",
            "metrics": metrics,
        }
        logger.info(f"Model training completed: {metrics}")

    except Exception as e:
        training_status = {"status": "error", "progress": f"Error: {str(e)}", "metrics": {}}
        logger.error(f"Model training failed: {e}")


@app.post("/api/model/train")
async def train_model(req: TrainRequest, background_tasks: BackgroundTasks):
    if training_status.get("status") == "training" or training_status.get("status") == "collecting_data":
        raise HTTPException(400, "Training already in progress")

    pairs = req.pairs or _current_config.pairs
    background_tasks.add_task(_train_model_task, pairs, req.num_candles)
    return {"status": "started", "pairs": pairs, "num_candles": req.num_candles}



# ── Trading Bot Control ─────────────────────────────────────────────────────


@app.post("/api/bot/start")
async def start_bot():
    m = _get_model()
    if not m.is_trained:
        raise HTTPException(400, "Model not trained yet. Train the model first.")

    binance_cfg = _get_binance_config()
    if not binance_cfg.api_key or not binance_cfg.api_secret:
        raise HTTPException(400, "Binance API credentials not configured.")

    eng = _ensure_engine()
    if eng.is_running:
        return {"status": "already_running"}

    eng.start()
    return {"status": "started", "testnet": binance_cfg.testnet}


@app.post("/api/bot/stop")
async def stop_bot():
    eng = _ensure_engine()
    eng.stop()
    return {"status": "stopped"}


@app.get("/api/bot/status")
async def bot_status():
    eng = _ensure_engine()
    stats = eng.get_stats()
    m = _get_model()
    return {
        **stats,
        "testnet": _get_binance_config().testnet,
        "model_accuracy": m.accuracy,
        "model_trained": m.is_trained,
    }


@app.post("/api/bot/scan")
async def manual_scan():
    """Manually trigger a scan for trade signals."""
    if not _get_model().is_trained:
        raise HTTPException(400, "Model not trained yet.")

    eng = _ensure_engine()
    new_trades = eng.scan_and_trade()
    return {
        "new_trades": len(new_trades),
        "trades": [_trade_to_dict(t) for t in new_trades],
    }


# ── Trades ───────────────────────────────────────────────────────────────────


def _trade_to_dict(t) -> Dict:
    return {
        "id": t.id,
        "symbol": t.symbol,
        "side": t.side,
        "entry_time": t.entry_time,
        "exit_time": t.exit_time,
        "entry_price": t.entry_price,
        "exit_price": t.exit_price,
        "quantity": t.quantity,
        "margin": t.margin,
        "leverage": t.leverage,
        "tp_price": t.tp_price,
        "sl_price": t.sl_price,
        "pnl": t.pnl,
        "pnl_pct": t.pnl_pct,
        "status": t.status,
        "confidence": t.confidence,
        "exit_reason": t.exit_reason,
        "is_custom": t.is_custom,
    }


@app.get("/api/trades")
async def get_trades(status: Optional[str] = None):
    eng = _ensure_engine()
    trades = eng.trades
    if status:
        trades = [t for t in trades if t.status == status.upper()]
    return {"trades": [_trade_to_dict(t) for t in reversed(trades)]}


@app.get("/api/trades/{trade_id}")
async def get_trade(trade_id: str):
    eng = _ensure_engine()
    trade = next((t for t in eng.trades if t.id == trade_id), None)
    if not trade:
        raise HTTPException(404, "Trade not found")
    return _trade_to_dict(trade)


@app.post("/api/trades/custom")
async def custom_trade(req: CustomTradeRequest):
    if not _get_binance_config().api_key:
        raise HTTPException(400, "Binance API credentials not configured.")

    symbol = req.symbol.upper()
    side = req.side.upper()
    if side not in ("LONG", "SHORT"):
        raise HTTPException(400, "Side must be LONG or SHORT")

    eng = _ensure_engine()
    trade = eng.open_trade(symbol, side, confidence=0.0, is_custom=True)
    if not trade:
        raise HTTPException(500, "Failed to open trade")
    return _trade_to_dict(trade)


@app.post("/api/trades/{trade_id}/close")
async def close_trade(trade_id: str):
    eng = _ensure_engine()
    success = eng.close_trade(trade_id, reason="manual")
    if not success:
        raise HTTPException(404, "Trade not found or already closed")
    trade = next((t for t in eng.trades if t.id == trade_id), None)
    if trade:
        return _trade_to_dict(trade)
    return {"status": "closed"}


# ── Backtest ─────────────────────────────────────────────────────────────────


@app.post("/api/backtest")
async def run_backtest_endpoint(req: BacktestRequest):
    m = _get_model()
    if not m.is_trained:
        raise HTTPException(400, "Model not trained yet. Train the model first.")

    config = _get_trading_config()
    rb, rfb = _lazy_backtest()

    if req.symbol:
        result = rb(m, config, req.symbol.upper(), req.num_candles)
        return {
            "symbol": result.symbol,
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
    else:
        pairs = req.pairs or config.pairs
        config.pairs = pairs
        return rfb(m, config, req.num_candles)


# ── Logs ─────────────────────────────────────────────────────────────────────


@app.get("/api/logs")
async def get_logs(limit: int = 100):
    eng = _ensure_engine()
    return {"logs": eng.logs[-limit:]}


# ── Prices ───────────────────────────────────────────────────────────────────


@app.get("/api/prices")
async def get_prices():
    """Get current prices for all configured pairs."""
    from app.data_fetcher import fetch_klines

    prices = {}
    for pair in _current_config.pairs:
        try:
            candles = fetch_klines(pair, "1m", 1)
            if candles:
                prices[pair] = {
                    "price": candles[-1].close,
                    "high_24h": candles[-1].high,
                    "low_24h": candles[-1].low,
                    "volume": candles[-1].volume,
                }
        except Exception:
            prices[pair] = {"price": 0, "error": "Failed to fetch"}
    return {"prices": prices}


# ── Signals ──────────────────────────────────────────────────────────────────


@app.get("/api/signals")
async def get_signals():
    """Get current ML signals for all pairs."""
    m = _get_model()
    if not m.is_trained:
        raise HTTPException(400, "Model not trained yet.")

    from app.data_fetcher import fetch_klines
    from app.indicators import compute_features

    config = _get_trading_config()
    signals = []

    for pair in config.pairs:
        try:
            candles = fetch_klines(pair, config.timeframe, 300)
            htf_candles = fetch_klines(pair, config.htf_timeframe, 300)
            if len(candles) < 100:
                continue
            features = compute_features(candles, len(candles) - 1, htf_candles)
            if features is None:
                continue
            direction, confidence = m.predict(features)
            signals.append({
                "symbol": pair,
                "direction": direction,
                "confidence": round(confidence, 4),
                "price": candles[-1].close,
                "rsi": round(features["rsi"], 2),
                "adx": round(features["adx"], 2),
                "volume_ratio": round(features["vol_ratio"], 2),
            })
        except Exception as e:
            logger.warning(f"Error getting signal for {pair}: {e}")

    return {"signals": sorted(signals, key=lambda s: s["confidence"], reverse=True)}


# ── Static Frontend Serving (must be AFTER all API routes) ─────────────────

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

if _STATIC_DIR.is_dir():
    @app.get("/")
    async def serve_index():
        return FileResponse(_STATIC_DIR / "index.html")

    app.mount("/assets", StaticFiles(directory=_STATIC_DIR / "assets"), name="static-assets")

    @app.get("/{path:path}")
    async def serve_spa(path: str):
        """Catch-all for SPA client-side routing."""
        file_path = (_STATIC_DIR / path).resolve()
        if file_path.is_relative_to(_STATIC_DIR.resolve()) and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(_STATIC_DIR / "index.html")
