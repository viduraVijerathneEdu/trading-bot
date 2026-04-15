"""ML model for trade signal prediction."""
from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from app.config import TradingConfig
from app.data_fetcher import fetch_klines
from app.indicators import FEATURE_NAMES, Candle, calc_atr, compute_features

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


def _label_candle(
    candles: List[Candle],
    index: int,
    leverage: int = 20,
    tp_pct: float = 50.0,
    sl_pct: float = 50.0,
) -> Optional[int]:
    """
    Label: 1 = LONG profitable, 2 = SHORT profitable, 0 = no clear signal.
    At 20x leverage, 50% TP/SL means price must move 2.5% for TP or SL.
    """
    if index >= len(candles) - 1:
        return None

    entry_price = candles[index].close
    tp_move = tp_pct / leverage / 100.0  # 0.025 for 50%/20x
    sl_move = sl_pct / leverage / 100.0

    long_tp = entry_price * (1 + tp_move)
    long_sl = entry_price * (1 - sl_move)
    short_tp = entry_price * (1 - tp_move)
    short_sl = entry_price * (1 + sl_move)

    # Look forward up to 96 candles (24 hours on 15m)
    lookahead = min(96, len(candles) - index - 1)

    long_hit_tp = False
    long_hit_sl = False
    short_hit_tp = False
    short_hit_sl = False

    for j in range(1, lookahead + 1):
        future = candles[index + j]
        if not long_hit_tp and not long_hit_sl:
            if future.high >= long_tp:
                long_hit_tp = True
            if future.low <= long_sl:
                long_hit_sl = True
        if not short_hit_tp and not short_hit_sl:
            if future.low <= short_tp:
                short_hit_tp = True
            if future.high >= short_sl:
                short_hit_sl = True

    # Label based on which hit TP first without hitting SL
    long_win = long_hit_tp and not long_hit_sl
    short_win = short_hit_tp and not short_hit_sl

    if long_win and not short_win:
        return 1
    if short_win and not long_win:
        return 2
    return 0


def prepare_training_data(
    pairs: List[str],
    config: TradingConfig,
    limit: int = 3000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fetch data and prepare features/labels for all pairs."""
    all_features: List[List[float]] = []
    all_labels: List[int] = []

    for pair in pairs:
        logger.info(f"Fetching data for {pair}...")
        try:
            candles = fetch_klines(pair, config.timeframe, limit)
            htf_candles = fetch_klines(pair, config.htf_timeframe, limit)
        except Exception as e:
            logger.warning(f"Failed to fetch {pair}: {e}")
            continue

        if len(candles) < 200:
            logger.warning(f"Not enough data for {pair}: {len(candles)} candles")
            continue

        logger.info(f"Processing {pair}: {len(candles)} candles")

        for i in range(100, len(candles) - 96):
            features = compute_features(candles, i, htf_candles)
            if features is None:
                continue
            label = _label_candle(candles, i, config.leverage, config.tp_pct, config.sl_pct)
            if label is None:
                continue

            feature_vec = [features[name] for name in FEATURE_NAMES]
            all_features.append(feature_vec)
            all_labels.append(label)

    X = np.array(all_features, dtype=np.float64)
    y = np.array(all_labels, dtype=np.int32)
    logger.info(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Label distribution: 0={np.sum(y==0)}, 1(LONG)={np.sum(y==1)}, 2(SHORT)={np.sum(y==2)}")
    return X, y


class TradingModel:
    """Ensemble ML model for trade direction prediction."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.model: Optional[VotingClassifier] = None
        self.is_trained = False
        self.accuracy: float = 0.0
        self.metrics: Dict = {}

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train the ensemble model with time-series cross-validation."""
        if X.shape[0] < 100:
            raise ValueError(f"Not enough training samples: {X.shape[0]}")

        # Filter out class 0 (no signal) for binary-like classification
        # But keep as multi-class: 0=skip, 1=long, 2=short
        X_scaled = self.scaler.fit_transform(X)

        gb = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=20,
            random_state=42,
        )
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=15,
            random_state=42,
            n_jobs=-1,
        )

        self.model = VotingClassifier(
            estimators=[("gb", gb), ("rf", rf)],
            voting="soft",
        )

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores: List[float] = []
        trade_accuracies: List[float] = []

        for train_idx, test_idx in tscv.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            temp_model = VotingClassifier(
                estimators=[
                    ("gb", GradientBoostingClassifier(
                        n_estimators=200, max_depth=5, learning_rate=0.05,
                        subsample=0.8, min_samples_leaf=20, random_state=42,
                    )),
                    ("rf", RandomForestClassifier(
                        n_estimators=200, max_depth=8, min_samples_leaf=15,
                        random_state=42, n_jobs=-1,
                    )),
                ],
                voting="soft",
            )
            temp_model.fit(X_train, y_train)

            predictions = temp_model.predict(X_test)
            probas = temp_model.predict_proba(X_test)

            # Overall accuracy
            accuracy = np.mean(predictions == y_test)
            cv_scores.append(accuracy)

            # Trade accuracy: among trades we would actually take (high confidence)
            trade_mask = np.max(probas, axis=1) >= 0.55
            if np.sum(trade_mask) > 0:
                trade_preds = predictions[trade_mask]
                trade_labels = y_test[trade_mask]
                # Filter to only predictions that are 1 or 2 (actual trades)
                actual_trade_mask = (trade_preds == 1) | (trade_preds == 2)
                if np.sum(actual_trade_mask) > 0:
                    trade_acc = np.mean(
                        trade_preds[actual_trade_mask] == trade_labels[actual_trade_mask]
                    )
                    trade_accuracies.append(trade_acc)

        # Train final model on all data
        self.model.fit(X_scaled, y)
        self.is_trained = True

        self.accuracy = float(np.mean(cv_scores))
        trade_acc_mean = float(np.mean(trade_accuracies)) if trade_accuracies else 0.0

        self.metrics = {
            "cv_accuracy": self.accuracy,
            "cv_scores": [float(s) for s in cv_scores],
            "trade_accuracy": trade_acc_mean,
            "trade_accuracy_scores": [float(s) for s in trade_accuracies],
            "total_samples": int(X.shape[0]),
            "feature_count": int(X.shape[1]),
            "label_distribution": {
                "no_signal": int(np.sum(y == 0)),
                "long": int(np.sum(y == 1)),
                "short": int(np.sum(y == 2)),
            },
        }

        logger.info(f"Model trained. CV accuracy: {self.accuracy:.4f}, Trade accuracy: {trade_acc_mean:.4f}")
        return self.metrics

    def predict(self, features: Dict) -> Tuple[str, float]:
        """Predict trade direction and confidence."""
        if not self.is_trained or self.model is None:
            return "SKIP", 0.0

        feature_vec = np.array([[features[name] for name in FEATURE_NAMES]])
        feature_scaled = self.scaler.transform(feature_vec)

        proba = self.model.predict_proba(feature_scaled)[0]
        classes = self.model.classes_

        class_proba = {int(c): float(p) for c, p in zip(classes, proba)}
        long_prob = class_proba.get(1, 0.0)
        short_prob = class_proba.get(2, 0.0)
        skip_prob = class_proba.get(0, 0.0)

        if long_prob > short_prob and long_prob > skip_prob and long_prob >= 0.45:
            return "LONG", long_prob
        if short_prob > long_prob and short_prob > skip_prob and short_prob >= 0.45:
            return "SHORT", short_prob
        return "SKIP", max(long_prob, short_prob)

    def save(self, path: Optional[str] = None) -> str:
        """Save model to disk."""
        if path is None:
            path = str(MODEL_DIR / "trading_model.pkl")
        data = {
            "model": self.model,
            "scaler": self.scaler,
            "is_trained": self.is_trained,
            "accuracy": self.accuracy,
            "metrics": self.metrics,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Model saved to {path}")
        return path

    def load(self, path: Optional[str] = None) -> bool:
        """Load model from disk."""
        if path is None:
            path = str(MODEL_DIR / "trading_model.pkl")
        if not os.path.exists(path):
            logger.warning(f"Model file not found: {path}")
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.is_trained = data["is_trained"]
        self.accuracy = data["accuracy"]
        self.metrics = data.get("metrics", {})
        logger.info(f"Model loaded from {path}, accuracy: {self.accuracy:.4f}")
        return True
