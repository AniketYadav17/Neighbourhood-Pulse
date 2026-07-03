"""Training pipeline orchestration: features -> target -> model -> artifacts.

Each stage is idempotent (skip if its output exists) unless force=True.
Module-level path bindings (imported names, not config.X attribute access)
are deliberate: tests monkeypatch them per-test.
"""

import logging
import os

import pandas as pd

from neighbourhood_pulse.config import (
    ARTIFACTS_DIR,
    COFFEE_SHOPS_PROCESSED_PATH,
    HEX_FEATURES_PATH,
    HEX_PRICE_TARGET_PATH,
    HEX_TRAINING_PATH,
    LR_RAW_DIR,
    LR_SALES_PATH,
    LR_YEARS,
    METRICS_PATH,
    MIN_SALES_PER_HEX,
    MODEL_PATH,
    PLANNING_PROCESSED_PATH,
    VALUATION_GAP_PATH,
)
from neighbourhood_pulse.features import (
    assign_hex_borough,
    build_coffee_features,
    build_hex_features,
    build_planning_features,
    compute_borough_frames,
    load_planning,
)
from neighbourhood_pulse.model import (
    add_centrality,
    compute_valuation_gap,
    fit_full,
    run_backtest,
    save_artifacts,
    train_and_evaluate,
)
from neighbourhood_pulse.target import build_postcode_lookup, build_price_target, load_sales

logger = logging.getLogger(__name__)


def _fresh(path: str, force: bool) -> bool:
    if force or not os.path.exists(path):
        return True
    logger.info("Reusing existing %s (use --force to rebuild).", path)
    return False


def _ensure_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def run_train(force: bool = False) -> dict:
    """Build every stage from processed data to committed-artifact candidates."""
    ran = False

    # Stage 1: hexagon feature matrix
    if ran or _fresh(HEX_FEATURES_PATH, force):
        ran = True
        planning = load_planning(PLANNING_PROCESSED_PATH)
        frames = compute_borough_frames(planning)
        hex_borough = assign_hex_borough(planning)
        planning_features = build_planning_features(planning, frames, hex_borough)
        coffee = pd.read_parquet(COFFEE_SHOPS_PROCESSED_PATH, columns=["h3_index", "brand"])
        hex_features = build_hex_features(planning_features, build_coffee_features(coffee))
        _ensure_dir(HEX_FEATURES_PATH)
        hex_features.to_parquet(HEX_FEATURES_PATH, index=False)
        logger.info("Feature matrix: %s hexagons.", len(hex_features))

    # Stage 2: pooled London sales
    if ran or _fresh(LR_SALES_PATH, force):
        ran = True
        sales = load_sales(LR_RAW_DIR, LR_YEARS)
        _ensure_dir(LR_SALES_PATH)
        sales.to_parquet(LR_SALES_PATH, index=False)

    # Stage 3: per-hexagon price target
    if ran or _fresh(HEX_PRICE_TARGET_PATH, force):
        ran = True
        sales = pd.read_parquet(LR_SALES_PATH)
        lookup = build_postcode_lookup(PLANNING_PROCESSED_PATH, COFFEE_SHOPS_PROCESSED_PATH)
        target = build_price_target(sales, lookup)
        _ensure_dir(HEX_PRICE_TARGET_PATH)
        target.to_parquet(HEX_PRICE_TARGET_PATH, index=False)

    # Stage 4: training table (features x target at the sales floor)
    if ran or _fresh(HEX_TRAINING_PATH, force):
        ran = True
        features = pd.read_parquet(HEX_FEATURES_PATH)
        target = pd.read_parquet(HEX_PRICE_TARGET_PATH)
        target = target[target["sales_count"] >= MIN_SALES_PER_HEX]
        train = features.merge(target, on="h3_index", how="inner")
        train = add_centrality(train)
        _ensure_dir(HEX_TRAINING_PATH)
        train.to_parquet(HEX_TRAINING_PATH, index=False)
        logger.info("Training table: %s hexagons (>=%s sales).", len(train), MIN_SALES_PER_HEX)

    # Stage 5: model + gap + back-test -> artifacts
    if (
        ran
        or _fresh(VALUATION_GAP_PATH, force)
        or _fresh(METRICS_PATH, force)
        or _fresh(MODEL_PATH, force)
    ):
        ran = True
        train = pd.read_parquet(HEX_TRAINING_PATH)
        metrics = train_and_evaluate(train)
        gap_df = compute_valuation_gap(train)
        sales = pd.read_parquet(LR_SALES_PATH)
        lookup = build_postcode_lookup(PLANNING_PROCESSED_PATH, COFFEE_SHOPS_PROCESSED_PATH)
        metrics["backtest"] = run_backtest(gap_df, sales, lookup)
        model = fit_full(train)
        save_artifacts(gap_df, metrics, model, artifacts_dir=ARTIFACTS_DIR)
        return metrics

    logger.info("All artifacts up to date; nothing to do.")
    import json

    with open(METRICS_PATH, encoding="utf-8") as f:
        return json.load(f)
