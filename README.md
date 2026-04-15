# ML Crypto Futures Trading Bot

An ML-powered cryptocurrency futures trading bot for Binance with a web dashboard.

## Features

- **ML Model**: Gradient Boosting + Random Forest ensemble trained on 28 technical indicators across 17 crypto pairs
- **Automated Trading**: Executes trades on Binance Futures with configurable parameters
- **Backtesting**: Full backtesting engine with equity curves and per-pair analysis
- **Web Dashboard**: Real-time dashboard with bot control, trade history, signals view, and backtest visualization
- **Testnet/Real Toggle**: Switch between Binance Futures Testnet and Mainnet

## Trading Parameters

| Parameter | Value |
|-----------|-------|
| Margin per trade | $1.00 |
| Leverage | 20x |
| Take Profit | 50% of margin |
| Stop Loss | 50% of margin |
| Min trades/month | 30+ |

## Supported Pairs

XRP, DOGE, ADA, SOL, SHIB, PEPE, LINK, MATIC, AVAX, ARB, OP, SUI, APT, NEAR, FTM, DOT, ATOM (all /USDT)

## ML Model Features (28 indicators)

RSI, ATR%, ADX, Choppiness Index, Volume Ratio, Candlestick patterns (close position, body ratio, wick ratio), EMA alignment (9/21/50), MACD, Bollinger Bands, Stochastic, Volume Profile (POC), Momentum (1/5/10 bar returns), Consecutive candle direction, HTF trend analysis

## Quick Start

### Backend

```bash
cd backend
poetry install
# Set environment variables or use the web dashboard settings
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Then open http://localhost:5173 in your browser.

### Usage

1. Go to **Settings** tab and enter your Binance API credentials
2. Toggle between **Testnet** and **Real** mode
3. Click **Train Model** on the Dashboard to train the ML model on historical data
4. Once trained, click **Start Bot** to begin automated trading
5. Use **Backtest** tab to validate model performance
6. Use **Custom Trade** in the Trades tab to place manual trades

## Architecture

```
backend/
  app/
    main.py          # FastAPI endpoints
    ml_model.py      # ML model (GradientBoosting + RandomForest ensemble)
    trading_engine.py # Binance Futures order execution
    backtester.py    # Historical backtesting
    indicators.py    # 28 technical indicators
    data_fetcher.py  # Binance kline data fetcher
    config.py        # Configuration
frontend/
  src/
    App.tsx          # React dashboard with 5 tabs
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/bot/status | Bot status and stats |
| POST | /api/bot/start | Start trading bot |
| POST | /api/bot/stop | Stop trading bot |
| POST | /api/bot/scan | Manual signal scan |
| POST | /api/model/train | Train ML model |
| GET | /api/model/status | Model training status |
| GET | /api/trades | Trade history |
| POST | /api/trades/custom | Place custom trade |
| POST | /api/trades/{id}/close | Close a trade |
| POST | /api/backtest | Run backtest |
| GET | /api/signals | Current ML signals |
| GET/POST | /api/config | Get/update configuration |
