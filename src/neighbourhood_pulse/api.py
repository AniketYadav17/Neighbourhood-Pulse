"""Read-only serving API over the committed artifacts, plus what-if /predict.

App factory (uvicorn neighbourhood_pulse.api:create_app --factory): artifacts
load once at startup, never per request. No database, no state — the artifact
directory IS the deployment. The public Streamlit app never depends on this
service; it exists for the docker-compose prod-parity path (the app's what-if
panel is /predict's real client there).
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from neighbourhood_pulse import __version__, config
from neighbourhood_pulse.model import feature_bounds, predict_price

SUMMARY_COLS = ["h3_index", "borough", "median_price", "pred_price", "valuation_gap"]


class HexagonSummary(BaseModel):
    h3_index: str
    borough: str
    median_price: float
    pred_price: float
    valuation_gap: float


class BriefOut(BaseModel):
    headline: str
    brief: str
    caveat: str


class HexagonDetail(HexagonSummary):
    sales_count: int
    signals: dict[str, float]
    brief: BriefOut | None = None


class PredictRequest(BaseModel):
    """One row of model features. Field set is pinned to FEATURE_COLS by a test."""

    model_config = ConfigDict(extra="forbid")

    total_applications: float
    change_of_use_count: float
    applications_recent: float
    change_of_use_ratio: float
    planning_velocity: float
    total_cafe_count: float
    independent_cafe_count: float
    cafe_to_application_ratio: float
    dist_to_centre_km: float


class PredictResponse(BaseModel):
    predicted_price: float


def create_app(artifacts_dir: str | None = None) -> FastAPI:
    base = Path(artifacts_dir if artifacts_dir is not None else config.ARTIFACTS_DIR)
    gap = pd.read_parquet(base / Path(config.VALUATION_GAP_PATH).name)
    model = joblib.load(base / Path(config.MODEL_PATH).name)
    briefs_path = base / Path(config.BRIEFS_PATH).name
    briefs = json.loads(briefs_path.read_text(encoding="utf-8")) if briefs_path.exists() else {}
    bounds = feature_bounds(gap)
    gap = gap.set_index("h3_index", drop=False)

    app = FastAPI(
        title="Neighbourhood Pulse API",
        version=__version__,
        description="Valuation-gap data for London H3 hexagons + what-if repricing.",
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "n_hexagons": int(len(gap)), "version": __version__}

    @app.get("/hexagons", response_model=list[HexagonSummary])
    def hexagons(
        borough: str | None = None,
        min_gap: float | None = None,
        max_gap: float | None = None,
    ):
        df = gap
        if borough is not None:
            df = df[df["borough"] == borough]
        if min_gap is not None:
            df = df[df["valuation_gap"] >= min_gap]
        if max_gap is not None:
            df = df[df["valuation_gap"] <= max_gap]
        return df[SUMMARY_COLS].to_dict("records")

    @app.get("/hexagons/{h3_index}", response_model=HexagonDetail)
    def hexagon(h3_index: str):
        if h3_index not in gap.index:
            raise HTTPException(status_code=404, detail=f"unknown hexagon {h3_index}")
        row = gap.loc[h3_index]
        return {
            **{c: row[c] for c in SUMMARY_COLS},
            "sales_count": int(row["sales_count"]),
            "signals": {c: float(row[c]) for c in config.FEATURE_COLS},
            "brief": briefs.get(h3_index),
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest):
        payload = request.model_dump()
        for col, (lo, hi) in bounds.items():
            if not lo <= payload[col] <= hi:
                raise HTTPException(
                    status_code=422,
                    detail=f"{col}={payload[col]:.4g} outside observed range [{lo:.4g}, {hi:.4g}]",
                )
        price = float(predict_price(model, pd.DataFrame([payload]))[0])
        return {"predicted_price": price}

    return app
