"""Shared fixtures: a tiny trained artifact set for API and app-helper tests."""

import json

import pytest

from synth import synthetic_training


@pytest.fixture(scope="session")
def tiny_artifacts(tmp_path_factory):
    """artifacts dir with all four serving artifacts built from synthetic data."""
    from neighbourhood_pulse.model import (
        compute_valuation_gap,
        fit_full,
        save_artifacts,
        train_and_evaluate,
    )

    train = synthetic_training()
    gap = compute_valuation_gap(train)
    metrics = train_and_evaluate(train)
    metrics["backtest"] = {"n_hexagons": 0, "correlation": 0.0, "quintiles": {}}
    art = tmp_path_factory.mktemp("artifacts")
    save_artifacts(gap, metrics, fit_full(train), artifacts_dir=str(art))
    briefed = gap["h3_index"].iloc[0]
    (art / "briefs.json").write_text(
        json.dumps({briefed: {"headline": "h", "brief": "b", "caveat": "c", "model": "test"}}),
        encoding="utf-8",
    )
    return art, gap


@pytest.fixture(scope="session")
def api_client(tiny_artifacts):
    from fastapi.testclient import TestClient

    from neighbourhood_pulse.api import create_app

    art, gap = tiny_artifacts
    return TestClient(create_app(artifacts_dir=str(art))), gap
