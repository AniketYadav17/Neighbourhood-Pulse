"""Parity gate: the pipeline must reproduce the notebook-era artifact.

Runs only where the artifacts exist (a machine that has run `pulse train` on
real data); CI skips it. Tolerances: identical hexagon set, identical median
prices, gap correlation > 0.999 (OOF fold assignment depends on row order,
which the pipeline reproduces; correlation absorbs any residual float drift),
R² within ±0.01 of the report's 0.418 / 0.439.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

GOLDEN = Path("tests/goldens/hex_valuation_gap_notebook.parquet")
NEW = Path("artifacts/hex_valuation_gap.parquet")
METRICS = Path("artifacts/metrics.json")

pytestmark = pytest.mark.skipif(
    not (GOLDEN.exists() and NEW.exists() and METRICS.exists()),
    reason="parity gate needs real-data artifacts (run `pulse train` locally)",
)


def test_same_hexagon_set():
    golden = pd.read_parquet(GOLDEN, columns=["h3_index"])
    new = pd.read_parquet(NEW, columns=["h3_index"])
    assert set(new["h3_index"]) == set(golden["h3_index"])


def test_median_prices_identical_and_gap_correlated():
    golden = pd.read_parquet(GOLDEN).set_index("h3_index").sort_index()
    new = pd.read_parquet(NEW).set_index("h3_index").sort_index()
    pd.testing.assert_series_equal(new["median_price"], golden["median_price"], check_exact=True)
    corr = new["valuation_gap"].corr(golden["valuation_gap"])
    assert corr > 0.999, f"gap correlation {corr:.6f} below parity tolerance"


def test_r2_matches_report():
    metrics = json.loads(METRICS.read_text())
    assert metrics["r2_linear"] == pytest.approx(0.418, abs=0.01)
    assert metrics["r2_xgboost"] == pytest.approx(0.439, abs=0.01)
    assert metrics["backtest"]["correlation"] == pytest.approx(-0.249, abs=0.02)
