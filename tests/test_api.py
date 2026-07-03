"""API contract tests over synthetic artifacts (TestClient, no server, no network)."""

from neighbourhood_pulse.config import FEATURE_COLS


def features_of(gap, i=0):
    return {c: float(gap[c].iloc[i]) for c in FEATURE_COLS}


def test_health(api_client):
    client, gap = api_client
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["n_hexagons"] == len(gap)


def test_hexagons_borough_and_gap_filters(api_client):
    client, gap = api_client
    assert len(client.get("/hexagons").json()) == len(gap)
    assert client.get("/hexagons", params={"borough": "Nowhere"}).json() == []
    undervalued = client.get("/hexagons", params={"max_gap": 0.0}).json()
    assert len(undervalued) == int((gap["valuation_gap"] <= 0).sum())
    assert set(undervalued[0]) == {
        "h3_index",
        "borough",
        "median_price",
        "pred_price",
        "valuation_gap",
    }


def test_hexagon_detail_brief_and_404(api_client):
    client, gap = api_client
    briefed = gap["h3_index"].iloc[0]
    body = client.get(f"/hexagons/{briefed}").json()
    assert body["brief"]["headline"] == "h"
    assert set(body["signals"]) == set(FEATURE_COLS)
    unbriefed = client.get(f"/hexagons/{gap['h3_index'].iloc[1]}").json()
    assert unbriefed["brief"] is None
    assert client.get("/hexagons/8809a5a5a5fffff").status_code == 404


def test_predict_roundtrip(api_client):
    client, gap = api_client
    body = client.post("/predict", json=features_of(gap)).json()
    assert body["predicted_price"] > 0


def test_predict_rejects_out_of_bounds(api_client):
    client, gap = api_client
    payload = features_of(gap) | {"dist_to_centre_km": 1e9}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "dist_to_centre_km" in response.text


def test_predict_rejects_unknown_and_missing_fields(api_client):
    client, gap = api_client
    assert client.post("/predict", json=features_of(gap) | {"bogus": 1}).status_code == 422
    short = features_of(gap)
    short.pop("planning_velocity")
    assert client.post("/predict", json=short).status_code == 422


def test_predict_request_matches_feature_contract():
    from neighbourhood_pulse.api import PredictRequest

    assert set(PredictRequest.model_fields) == set(FEATURE_COLS)


def test_feature_bounds_envelope(api_client):
    from neighbourhood_pulse.model import feature_bounds

    _, gap = api_client
    bounds = feature_bounds(gap)
    assert set(bounds) == set(FEATURE_COLS)
    lo, hi = bounds["dist_to_centre_km"]
    assert lo <= gap["dist_to_centre_km"].min() and hi >= gap["dist_to_centre_km"].max()
