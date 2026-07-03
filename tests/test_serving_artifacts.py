"""The committed serving artifacts must load and predict in a fresh environment."""

import json
from pathlib import Path

import joblib
import pandas as pd

from neighbourhood_pulse.config import (
    FEATURE_COLS,
    METRICS_PATH,
    MODEL_PATH,
    VALUATION_GAP_PATH,
)
from neighbourhood_pulse.model import predict_price


def test_committed_model_loads_and_predicts():
    model = joblib.load(MODEL_PATH)
    gap = pd.read_parquet(VALUATION_GAP_PATH)
    preds = predict_price(model, gap[FEATURE_COLS].head(5))
    assert (preds > 0).all()
    assert (preds < 1e8).all()


def test_metrics_json_matches_feature_contract():
    metrics = json.loads(Path(METRICS_PATH).read_text(encoding="utf-8"))
    assert metrics["feature_cols"] == FEATURE_COLS
    assert {"r2_linear", "r2_xgboost", "backtest", "n_hexagons"} <= set(metrics)
    assert "build" in metrics
    assert metrics["build"]["versions"]["xgboost"]
