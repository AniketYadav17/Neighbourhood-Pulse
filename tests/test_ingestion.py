"""Characterization tests for ingestion. No real network: session.post is stubbed."""
import geopandas as gpd
import pandas as pd
import pytest
import requests
from shapely.geometry import Point

import neighbourhood_pulse.ingestion as ing
from neighbourhood_pulse.ingestion import DataIngestion, IngestionError


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, headers=None):
        self.status_code = status_code
        self._json = json_data if json_data is not None else {}
        self.headers = headers or {}

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(str(self.status_code))


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Ingestion sleeps between batches/retries; tests record instead of waiting."""
    sleeps = []
    monkeypatch.setattr(ing.time, "sleep", sleeps.append)
    return sleeps


def make_ingestion(tmp_path, boroughs=("Testborough",)):
    return DataIngestion(
        raw_dir=str(tmp_path / "planning"),
        combined_path=str(tmp_path / "planning_applications.parquet"),
        coffee_path=str(tmp_path / "coffee_shops.parquet"),
        boroughs=list(boroughs),
    )


def queue_responses(monkeypatch, ingestion, responses):
    remaining = list(responses)
    monkeypatch.setattr(ingestion.session, "post", lambda *a, **k: remaining.pop(0))


def seed_coffee_file(tmp_path):
    """Pre-create the coffee parquet so run() never reaches OSMnx."""
    gdf = gpd.GeoDataFrame(
        {"name": ["Indie"], "brand": [None]},
        geometry=[Point(-0.1, 51.5)],
        crs="EPSG:4326",
    )
    gdf.to_parquet(tmp_path / "coffee_shops.parquet")


class TestPostWithRetry:
    def test_429_honours_retry_after_then_succeeds(self, tmp_path, monkeypatch, no_sleep):
        di = make_ingestion(tmp_path)
        queue_responses(monkeypatch, di, [
            FakeResponse(429, headers={"Retry-After": "7"}),
            FakeResponse(200, json_data={"ok": True}),
        ])
        assert di._post_with_retry("http://x", {}).json() == {"ok": True}
        assert no_sleep == [7]  # server hint used, not RETRY_DELAY

    def test_503_uses_fallback_delay(self, tmp_path, monkeypatch, no_sleep):
        di = make_ingestion(tmp_path)
        queue_responses(monkeypatch, di, [FakeResponse(503), FakeResponse(200)])
        di._post_with_retry("http://x", {})
        assert no_sleep == [ing.RETRY_DELAY]

    def test_http_date_retry_after_falls_back(self, tmp_path, monkeypatch, no_sleep):
        di = make_ingestion(tmp_path)
        queue_responses(monkeypatch, di, [
            FakeResponse(429, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}),
            FakeResponse(200),
        ])
        di._post_with_retry("http://x", {})
        assert no_sleep == [ing.RETRY_DELAY]

    def test_exhausted_retries_raise_ingestion_error(self, tmp_path, monkeypatch, no_sleep):
        di = make_ingestion(tmp_path)
        monkeypatch.setattr(di.session, "post", lambda *a, **k: FakeResponse(429))
        with pytest.raises(IngestionError, match="Max retries"):
            di._post_with_retry("http://x", {})

    def test_4xx_raises_immediately_without_retry(self, tmp_path, monkeypatch, no_sleep):
        di = make_ingestion(tmp_path)
        monkeypatch.setattr(di.session, "post", lambda *a, **k: FakeResponse(404))
        with pytest.raises(requests.exceptions.HTTPError):
            di._post_with_retry("http://x", {})
        assert no_sleep == []


class TestFetchPlanningData:
    def test_scroll_pagination_accumulates_all_batches(self, tmp_path, monkeypatch, no_sleep):
        di = make_ingestion(tmp_path)
        queue_responses(monkeypatch, di, [
            FakeResponse(200, {"_scroll_id": "s1",
                               "hits": {"hits": [{"_source": {"id": 1}}, {"_source": {"id": 2}}]}}),
            FakeResponse(200, {"_scroll_id": "s2", "hits": {"hits": [{"_source": {"id": 3}}]}}),
            FakeResponse(200, {"_scroll_id": "s3", "hits": {"hits": []}}),
        ])
        records = di.fetch_planning_data("Testborough")
        assert [r["_source"]["id"] for r in records] == [1, 2, 3]

    def test_network_error_chains_to_ingestion_error(self, tmp_path, monkeypatch, no_sleep):
        di = make_ingestion(tmp_path)

        def boom(*a, **k):
            raise requests.exceptions.ConnectionError("refused")

        monkeypatch.setattr(di.session, "post", boom)
        with pytest.raises(IngestionError) as excinfo:
            di.fetch_planning_data("Testborough")
        assert isinstance(excinfo.value.__cause__, requests.exceptions.ConnectionError)


@pytest.mark.parametrize(
    ("borough", "slug"),
    [
        ("Barking & Dagenham", "Barking_Dagenham"),
        ("Kensington & Chelsea", "Kensington_Chelsea"),
        ("City of London", "City_of_London"),
        ("Richmond", "Richmond"),
    ],
)
def test_borough_slug(borough, slug):
    assert DataIngestion._borough_slug(borough) == slug


PLANNING_RECORD = {
    "_source": {
        "id": "A1", "lpa_name": "Full",
        "centroid_easting": "530000", "centroid_northing": "180000",
    }
}


class TestRun:
    def test_empty_borough_is_not_cached_as_done(self, tmp_path, monkeypatch, no_sleep):
        seed_coffee_file(tmp_path)
        di = make_ingestion(tmp_path, boroughs=["Full", "Empty"])
        monkeypatch.setattr(
            di, "fetch_planning_data",
            lambda b: [PLANNING_RECORD] if b == "Full" else [],
        )
        planning, coffee = di.run()
        assert (tmp_path / "planning" / "Full.parquet").exists()
        assert not (tmp_path / "planning" / "Empty.parquet").exists()
        assert len(planning) == 1
        assert len(coffee) == 1

    def test_rerun_resumes_without_network(self, tmp_path, monkeypatch, no_sleep):
        seed_coffee_file(tmp_path)
        di = make_ingestion(tmp_path, boroughs=["Full"])
        monkeypatch.setattr(di, "fetch_planning_data", lambda b: [PLANNING_RECORD])
        first, _ = di.run()

        def fail(_):
            raise AssertionError("network touched on resume")

        monkeypatch.setattr(di, "fetch_planning_data", fail)
        second, _ = di.run()
        pd.testing.assert_frame_equal(first, second)

    def test_all_empty_raises(self, tmp_path, monkeypatch, no_sleep):
        seed_coffee_file(tmp_path)
        di = make_ingestion(tmp_path, boroughs=["Empty"])
        monkeypatch.setattr(di, "fetch_planning_data", lambda b: [])
        with pytest.raises(IngestionError, match="No per-borough"):
            di.run()
