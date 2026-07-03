"""Valuation-gap model: train, evaluate out-of-fold, back-test, persist.

Transcribed from notebook sections 14-17 (centrality-controlled final model).
Target is log(median_price) — prices are right-skewed. The gap uses OUT-OF-FOLD
predictions (5-fold cross_val_predict): no hexagon is scored by a model that
saw it, so the gap is honest. The saved model is a separate full-data fit used
only for what-if repricing (serving), never for the gap.
"""

import json
import logging
import os

import h3
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_predict, train_test_split
from xgboost import XGBRegressor

from neighbourhood_pulse.config import (
    ARTIFACTS_DIR,
    BACKTEST_EARLY_YEARS,
    BACKTEST_LATE_YEARS,
    BACKTEST_MIN_SALES,
    CX_LAT,
    CX_LON,
    FEATURE_COLS,
)
from neighbourhood_pulse.target import normalise_postcodes

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0


def add_centrality(df: pd.DataFrame) -> pd.DataFrame:
    """Haversine km from each hexagon's centroid to Charing Cross (single-centre control)."""
    df = df.copy()
    latlng = df["h3_index"].map(h3.cell_to_latlng)
    lat = latlng.map(lambda t: t[0]).astype(float)
    lon = latlng.map(lambda t: t[1]).astype(float)
    p1, p2 = np.radians(lat), np.radians(CX_LAT)
    dphi = np.radians(CX_LAT - lat)
    dlambda = np.radians(CX_LON - lon)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
    df["dist_to_centre_km"] = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))
    return df


def make_model() -> XGBRegressor:
    """Frozen hyperparameters from the validated notebook — do not tune."""
    return XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )


def _xy(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return train[FEATURE_COLS].astype(float), np.log(train["median_price"].astype(float))


def train_and_evaluate(train: pd.DataFrame) -> dict:
    """Held-out R² for linear + XGBoost, and XGBoost feature importances."""
    X, y = _xy(train)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    r2_linear = r2_score(y_te, LinearRegression().fit(X_tr, y_tr).predict(X_te))
    xgb = make_model().fit(X_tr, y_tr)
    r2_xgboost = r2_score(y_te, xgb.predict(X_te))
    importance = dict(zip(FEATURE_COLS, (float(v) for v in xgb.feature_importances_), strict=True))
    logger.info("Held-out R²: linear=%.3f xgboost=%.3f", r2_linear, r2_xgboost)
    return {
        "r2_linear": float(r2_linear),
        "r2_xgboost": float(r2_xgboost),
        "feature_importance": importance,
    }


def compute_valuation_gap(train: pd.DataFrame) -> pd.DataFrame:
    """Out-of-fold predictions -> pred_price and valuation_gap columns."""
    df = train.copy()
    X, y = _xy(df)
    oof = cross_val_predict(make_model(), X, y, cv=5)
    df["pred_price"] = np.exp(oof)
    df["valuation_gap"] = df["median_price"] / df["pred_price"] - 1.0
    return df


def fit_full(train: pd.DataFrame) -> XGBRegressor:
    """Full-data fit persisted for serving (what-if repricing) — not used for the gap."""
    X, y = _xy(train)
    return make_model().fit(X, y)


def predict_price(model: XGBRegressor, features: pd.DataFrame) -> np.ndarray:
    """£ predictions from the log-space model, column order enforced."""
    return np.exp(model.predict(features[FEATURE_COLS].astype(float)))


def run_backtest(gap_df: pd.DataFrame, sales: pd.DataFrame, lookup: pd.Series) -> dict:
    """Does EARLY undervaluation precede growth? Early-window gap vs late-window growth.

    Thesis expects NEGATIVE correlation (undervalued -> grows more) and a
    monotonic quintile gradient. Proof-of-concept, not causal proof: features
    are not strictly frozen-as-of-2021 and 2-year windows are thin.
    """
    df = sales.dropna(subset=["postcode"]).copy()
    df["pc"] = normalise_postcodes(df["postcode"])
    df["h3_index"] = df["pc"].map(lookup)
    df = df.dropna(subset=["h3_index"])

    def window_median(years):
        g = df[df["year"].isin(years)].groupby("h3_index")["price"]
        return pd.DataFrame({"median": g.median(), "n": g.size()})

    early = window_median(BACKTEST_EARLY_YEARS).add_prefix("early_")
    late = window_median(BACKTEST_LATE_YEARS).add_prefix("late_")
    px = early.join(late, how="inner")
    px = px[(px["early_n"] >= BACKTEST_MIN_SALES) & (px["late_n"] >= BACKTEST_MIN_SALES)]
    px["growth"] = px["late_median"] / px["early_median"] - 1.0

    bt = gap_df.merge(px, on="h3_index", how="inner")
    X = bt[FEATURE_COLS].astype(float)
    y = np.log(bt["early_median"].astype(float))
    bt["early_pred"] = np.exp(cross_val_predict(make_model(), X, y, cv=5))
    bt["early_gap"] = bt["early_median"] / bt["early_pred"] - 1.0

    correlation = float(bt["early_gap"].corr(bt["growth"]))
    labels = ["Q1 most undervalued", "Q2", "Q3", "Q4", "Q5 most overvalued"]
    bt["gap_quintile"] = pd.qcut(bt["early_gap"], 5, labels=labels)
    quintiles = {
        str(k): float(v * 100)
        for k, v in bt.groupby("gap_quintile", observed=True)["growth"].mean().items()
    }
    logger.info("Back-test: n=%s corr=%.3f", len(bt), correlation)
    return {"n_hexagons": int(len(bt)), "correlation": correlation, "quintiles": quintiles}


def save_artifacts(
    gap_df: pd.DataFrame, metrics: dict, model: XGBRegressor, artifacts_dir: str = ARTIFACTS_DIR
) -> None:
    os.makedirs(artifacts_dir, exist_ok=True)
    gap_df.to_parquet(os.path.join(artifacts_dir, "hex_valuation_gap.parquet"), index=False)
    joblib.dump(model, os.path.join(artifacts_dir, "model.joblib"))
    payload = {**metrics, "n_hexagons": int(len(gap_df)), "feature_cols": FEATURE_COLS}
    with open(os.path.join(artifacts_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Artifacts saved to %s/.", artifacts_dir)
