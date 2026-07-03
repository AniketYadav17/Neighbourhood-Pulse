"""End-to-end pipeline test on a tiny synthetic world (tmp paths via monkeypatched config)."""

import json

import h3
import numpy as np
import pandas as pd
import pytest

import neighbourhood_pulse.pipeline as pipeline_module
from neighbourhood_pulse.pipeline import run_train

RNG = np.random.default_rng(1)


@pytest.fixture
def synthetic_world(tmp_path, monkeypatch):
    """60 hexagons, 2 boroughs, planning + coffee + LR sales, all under tmp_path."""
    processed = tmp_path / "data" / "processed"
    raw = tmp_path / "data" / "raw"
    artifacts = tmp_path / "artifacts"
    processed.mkdir(parents=True)
    raw.mkdir(parents=True)

    lats = RNG.uniform(51.4, 51.6, 60)
    lons = RNG.uniform(-0.3, 0.1, 60)
    hexes = [h3.latlng_to_cell(la, lo, 8) for la, lo in zip(lats, lons, strict=True)]
    postcodes = [f"E{i:02d}9ZZ" for i in range(60)]

    rows = []
    months = pd.date_range("2021-01-15", periods=48, freq="MS")
    for i, hx in enumerate(hexes):
        borough = "Alpha" if i < 30 else "Beta"
        for m in months:
            for _ in range(2 + i % 3):
                rows.append(
                    {
                        "h3_index": hx,
                        "lpa_name": borough,
                        "description": "Change of use to cafe"
                        if (i + m.month) % 7 == 0
                        else "extension",
                        "valid_date": (m + pd.Timedelta(days=5)).strftime("%d/%m/%Y"),
                        "postcode": postcodes[i],
                    }
                )
    pd.DataFrame(rows).to_parquet(processed / "planning_processed.parquet", index=False)

    coffee = pd.DataFrame(
        {
            "h3_index": RNG.choice(hexes, 40),
            "brand": [None] * 35 + ["Costa"] * 5,
            "addr:postcode": [postcodes[hexes.index(h)] for h in RNG.choice(hexes, 40)],
        }
    )
    coffee.to_parquet(processed / "coffee_shops_processed.parquet", index=False)

    def lr_row(price, date, pc):
        cells = ["x"] * 16
        cells[1], cells[2], cells[3], cells[4], cells[12], cells[15] = (
            str(price),
            date,
            pc,
            "F",
            "NEWHAM",
            "A",
        )
        return ",".join(f'"{c}"' for c in cells)

    for year in (2021, 2022, 2023, 2024, 2025):
        lines = []
        for i, pc in enumerate(postcodes):
            base = 300_000 + 5_000 * i
            for _ in range(8):  # 8/year -> 40 pooled >= MIN_SALES_PER_HEX(30)
                lines.append(
                    lr_row(int(base * (1 + 0.03 * (year - 2021))), f"{year}-06-01 00:00", pc)
                )
        (raw / f"pp-{year}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for name, value in {
        "PLANNING_PROCESSED_PATH": str(processed / "planning_processed.parquet"),
        "COFFEE_SHOPS_PROCESSED_PATH": str(processed / "coffee_shops_processed.parquet"),
        "HEX_FEATURES_PATH": str(processed / "hex_features.parquet"),
        "LR_SALES_PATH": str(processed / "lr_london_sales.parquet"),
        "HEX_PRICE_TARGET_PATH": str(processed / "hex_price_target.parquet"),
        "HEX_TRAINING_PATH": str(processed / "hex_training.parquet"),
        "LR_RAW_DIR": str(raw),
        "ARTIFACTS_DIR": str(artifacts),
        "VALUATION_GAP_PATH": str(artifacts / "hex_valuation_gap.parquet"),
        "METRICS_PATH": str(artifacts / "metrics.json"),
        "MODEL_PATH": str(artifacts / "model.joblib"),
    }.items():
        monkeypatch.setattr(pipeline_module, name, value)
    return tmp_path


def test_run_train_end_to_end(synthetic_world):
    metrics = run_train()
    artifacts = synthetic_world / "artifacts"
    assert (artifacts / "hex_valuation_gap.parquet").exists()
    assert (artifacts / "model.joblib").exists()
    saved = json.loads((artifacts / "metrics.json").read_text())
    assert saved["r2_xgboost"] == pytest.approx(metrics["r2_xgboost"])
    assert "backtest" in saved
    gap = pd.read_parquet(artifacts / "hex_valuation_gap.parquet")
    assert {
        "h3_index",
        "borough",
        "median_price",
        "pred_price",
        "valuation_gap",
        "dist_to_centre_km",
    } <= set(gap.columns)
    assert len(gap) > 0


def test_run_train_is_idempotent_and_force_rebuilds(synthetic_world):
    run_train()
    gap_path = synthetic_world / "artifacts" / "hex_valuation_gap.parquet"
    first_mtime = gap_path.stat().st_mtime_ns
    run_train()  # no force: artifacts exist -> skip recompute
    assert gap_path.stat().st_mtime_ns == first_mtime
    run_train(force=True)  # force: rebuild everything
    assert gap_path.stat().st_mtime_ns > first_mtime
