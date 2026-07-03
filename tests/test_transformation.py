"""Characterization tests for coordinate cleaning and H3 indexing."""

import geopandas as gpd
import h3
import pandas as pd
import pytest
from shapely.geometry import Point

from neighbourhood_pulse.config import H3_RESOLUTION
from neighbourhood_pulse.transformation import DataTransformation

# Trafalgar Square: BNG (easting, northing) and the expected WGS84 (lat, lon).
TRAFALGAR_BNG = (530_047, 180_422)
TRAFALGAR_WGS = (51.508, -0.128)


@pytest.fixture(scope="module")
def dt():
    return DataTransformation()


def planning_frame():
    return pd.DataFrame(
        {
            "centroid_easting": pd.array(
                [TRAFALGAR_BNG[0], -4722, 800_000, pd.NA], dtype="Int64"
            ),
            "centroid_northing": pd.array(
                [TRAFALGAR_BNG[1], 6_725_758, 100_000, pd.NA], dtype="Int64"
            ),
            "description": ["valid", "web-mercator junk", "easting out of range", "missing coords"],
        }
    )


def coffee_frame():
    return gpd.GeoDataFrame(
        {
            "name": ["Indie Cafe", "Costa"],
            "brand": [None, "Costa"],
            "cuisine": [None, None],
            "opening_hours": [None, None],
            "start_date": [None, None],
            "addr:postcode": ["WC2N 5DN", None],
            "operator": [None, None],
        },
        geometry=[Point(-0.128, 51.508), Point(0.005, 51.540)],
        crs="EPSG:4326",
    )


def test_invalid_bng_rows_are_dropped(dt):
    out = dt.transform_planning_data(planning_frame())
    assert list(out["description"]) == ["valid"]


def test_na_coordinates_are_dropped_not_crashing(dt):
    out = dt.transform_planning_data(planning_frame())
    assert list(out["description"]) == ["valid"]


def test_bng_to_wgs84(dt):
    row = dt.transform_planning_data(planning_frame()).iloc[0]
    assert row["latitude"] == pytest.approx(TRAFALGAR_WGS[0], abs=1e-2)
    assert row["longitude"] == pytest.approx(TRAFALGAR_WGS[1], abs=1e-2)


def test_h3_index_is_res8_and_matches_direct_computation(dt):
    row = dt.transform_planning_data(planning_frame()).iloc[0]
    assert h3.get_resolution(row["h3_index"]) == H3_RESOLUTION
    assert row["h3_index"] == h3.latlng_to_cell(row["latitude"], row["longitude"], H3_RESOLUTION)


def test_planning_output_is_wgs84_geodataframe(dt):
    out = dt.transform_planning_data(planning_frame())
    assert isinstance(out, gpd.GeoDataFrame)
    assert out.crs.to_epsg() == 4326


def test_coffee_transform_keeps_columns_and_assigns_h3(dt):
    out = dt.transform_coffee_data(coffee_frame())
    assert {"h3_index", "latitude", "longitude", "name", "brand"} <= set(out.columns)
    assert out["h3_index"].map(h3.get_resolution).eq(H3_RESOLUTION).all()
    # A point's centroid is itself: coordinates survive the CRS round-trip.
    assert out.iloc[0]["latitude"] == pytest.approx(51.508, abs=1e-3)
    assert out.iloc[0]["longitude"] == pytest.approx(-0.128, abs=1e-3)


def test_run_writes_outputs_then_resumes_from_disk(tmp_path, monkeypatch):
    planning_raw = tmp_path / "planning_raw.parquet"
    coffee_raw = tmp_path / "coffee_raw.parquet"
    planning_frame().to_parquet(planning_raw, index=False)
    coffee_frame().to_parquet(coffee_raw)

    dt = DataTransformation(
        planning_raw=str(planning_raw),
        coffee_raw=str(coffee_raw),
        planning_out=str(tmp_path / "out" / "planning.parquet"),
        coffee_out=str(tmp_path / "out" / "coffee.parquet"),
    )
    planning_1, coffee_1 = dt.run()
    assert len(planning_1) == 1
    assert len(coffee_1) == 2

    def fail(_df):
        raise AssertionError("recomputed despite existing outputs")

    monkeypatch.setattr(dt, "transform_planning_data", fail)
    planning_2, _ = dt.run()
    assert list(planning_2["h3_index"]) == list(planning_1["h3_index"])
