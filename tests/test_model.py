"""Model-layer tests on synthetic data: mechanics and invariants, not research metrics."""

import json

import h3
import numpy as np
import pandas as pd
import pytest

from neighbourhood_pulse.config import CX_LAT, CX_LON, FEATURE_COLS
from neighbourhood_pulse.model import (
    add_centrality,
    compute_valuation_gap,
    fit_full,
    make_model,
    predict_price,
    run_backtest,
    save_artifacts,
    train_and_evaluate,
)
from synth import synthetic_training


def test_add_centrality_haversine_known_point():
    centre_hex = h3.latlng_to_cell(CX_LAT, CX_LON, 8)
    df = add_centrality(pd.DataFrame({"h3_index": [centre_hex]}))
    # Hexagon centroid is within ~0.5 km of Charing Cross for the containing cell.
    assert df["dist_to_centre_km"].iloc[0] < 0.6


def test_make_model_hyperparameters_frozen():
    m = make_model()
    p = m.get_params()
    assert (p["n_estimators"], p["max_depth"], p["learning_rate"]) == (400, 4, 0.05)
    assert (p["subsample"], p["colsample_bytree"], p["random_state"]) == (0.8, 0.8, 42)


def test_train_and_evaluate_metrics_shape():
    metrics = train_and_evaluate(synthetic_training())
    assert set(metrics) == {"r2_linear", "r2_xgboost", "feature_importance"}
    assert -1 <= metrics["r2_xgboost"] <= 1
    assert set(metrics["feature_importance"]) == set(FEATURE_COLS)
    assert metrics["feature_importance"]  # non-empty, floats
    # Determinism: same data -> identical metrics (random_state pinned everywhere)
    assert train_and_evaluate(synthetic_training()) == metrics


def test_compute_valuation_gap_oof_and_identity():
    df = compute_valuation_gap(synthetic_training())
    assert {"pred_price", "valuation_gap"} <= set(df.columns)
    assert df["pred_price"].gt(0).all()
    # gap identity: actual/predicted - 1
    np.testing.assert_allclose(df["valuation_gap"], df["median_price"] / df["pred_price"] - 1.0)
    assert df["valuation_gap"].abs().mean() < 1.0  # log-model sanity on learnable data


def test_predict_price_roundtrip():
    train = synthetic_training()
    model = fit_full(train)
    preds = predict_price(model, train[FEATURE_COLS])
    assert preds.shape == (len(train),)
    assert (preds > 0).all()
    # In-sample predictions of a 400-tree model correlate strongly with actuals
    assert np.corrcoef(preds, train["median_price"])[0, 1] > 0.8


def test_run_backtest_structure():
    train = synthetic_training()
    lookup = pd.Series({f"PC{i}": h for i, h in enumerate(train["h3_index"])})
    rows = []
    for pc, h in lookup.items():
        base = float(train.set_index("h3_index").loc[h, "median_price"])
        for year, mult in ((2021, 0.9), (2022, 0.95), (2024, 1.05), (2025, 1.1)):
            for _ in range(16):  # >= BACKTEST_MIN_SALES per window
                rows.append({"price": base * mult, "postcode": pc, "year": year})
    sales = pd.DataFrame(rows)
    gap_df = compute_valuation_gap(train)
    result = run_backtest(gap_df, sales, lookup)
    assert set(result) == {"n_hexagons", "correlation", "quintiles"}
    assert result["n_hexagons"] == len(train)
    assert len(result["quintiles"]) == 5
    assert all(isinstance(v, float) for v in result["quintiles"].values())


def test_save_artifacts_writes_all_three(tmp_path):
    train = synthetic_training()
    gap_df = compute_valuation_gap(train)
    metrics = train_and_evaluate(train)
    metrics["backtest"] = {"n_hexagons": 0, "correlation": 0.0, "quintiles": {}}
    model = fit_full(train)
    save_artifacts(gap_df, metrics, model, artifacts_dir=str(tmp_path))
    assert (tmp_path / "hex_valuation_gap.parquet").exists()
    assert (tmp_path / "model.joblib").exists()
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["r2_xgboost"] == pytest.approx(metrics["r2_xgboost"])
    assert saved["feature_cols"] == FEATURE_COLS
    build = saved["build"]
    assert set(build) >= {"package_version", "git_sha", "trained_at", "versions"}
    assert set(build["versions"]) == {"python", "pandas", "scikit_learn", "xgboost"}
