"""Shared app data access: artifact loaders, colour scale, what-if repricing.

The what-if path is the same computation everywhere: in-process against the
committed model.joblib on Streamlit Cloud, or POST /predict when PULSE_API_URL
is set (docker-compose) — both ultimately call model.predict_price.
"""

import json
import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from neighbourhood_pulse.config import (
    BRIEFS_PATH,
    FEATURE_COLS,
    METRICS_PATH,
    MODEL_PATH,
    VALUATION_GAP_PATH,
)
from neighbourhood_pulse.model import predict_price

REPO = Path(__file__).resolve().parent.parent

RED, CREAM, GREEN = (215, 48, 39), (255, 255, 191), (26, 152, 80)


@st.cache_data
def load_gap() -> pd.DataFrame:
    return pd.read_parquet(REPO / VALUATION_GAP_PATH)


@st.cache_data
def load_metrics() -> dict:
    return json.loads((REPO / METRICS_PATH).read_text(encoding="utf-8"))


@st.cache_data
def load_briefs() -> dict:
    path = REPO / BRIEFS_PATH
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_resource
def load_model():
    return joblib.load(REPO / MODEL_PATH)


def gap_colour(gap: float, scale: float) -> list[int]:
    """Diverging red (undervalued) -> cream -> green (overvalued), clamped at ±scale."""
    t = max(-1.0, min(1.0, gap / scale)) if scale else 0.0
    a, b, u = (RED, CREAM, t + 1.0) if t < 0 else (CREAM, GREEN, t)
    return [round(a[i] + u * (b[i] - a[i])) for i in range(3)]


def derive_what_if_features(row: pd.Series, edits: dict) -> dict:
    """Full 9-feature vector from a hexagon row + edited BASE signals.

    Sliders edit counts; the derived features (ratios, velocity) are recomputed
    so the vector stays internally consistent. Centrality is never editable —
    you can't move a hexagon.
    """
    features = {c: float(row[c]) for c in FEATURE_COLS}
    features.update(edits)
    features["change_of_use_ratio"] = (
        features["change_of_use_count"] / features["total_applications"]
    )
    features["planning_velocity"] = features["applications_recent"] / (
        features["total_applications"] / float(row["span_years"])
    )
    features["cafe_to_application_ratio"] = (
        features["total_cafe_count"] / features["total_applications"]
    )
    return features


def reprice(features: dict) -> float:
    """Predicted £ for a feature vector — via the API when configured, else in-process."""
    api_url = os.environ.get("PULSE_API_URL")
    if api_url:
        import requests

        response = requests.post(f"{api_url}/predict", json=features, timeout=10)
        response.raise_for_status()
        return float(response.json()["predicted_price"])
    return float(predict_price(load_model(), pd.DataFrame([features]))[0])
