"""Coordinate cleaning and spatial indexing.

Converts planning-record British National Grid coordinates (EPSG:27700) to
WGS84, filters records whose coordinates fall outside the valid BNG range
(~20% of raw records store a different CRS — see notebook section 2), and
assigns each record its H3 res-8 hexagon.
"""
import logging
import os

import geopandas as gpd
import h3
import pandas as pd
from pyproj import Transformer

from neighbourhood_pulse.config import (
    COFFEE_SHOPS_PROCESSED_PATH,
    COFFEE_SHOPS_RAW_PATH,
    H3_RESOLUTION,
    PLANNING_PROCESSED_PATH,
    PLANNING_RAW_PATH,
)

logger = logging.getLogger(__name__)


class DataTransformation:
    """Cleans raw coordinates and adds the H3 spatial index. Paths injectable for tests."""

    def __init__(
        self,
        planning_raw: str = PLANNING_RAW_PATH,
        coffee_raw: str = COFFEE_SHOPS_RAW_PATH,
        planning_out: str = PLANNING_PROCESSED_PATH,
        coffee_out: str = COFFEE_SHOPS_PROCESSED_PATH,
    ):
        self.planning_raw = planning_raw
        self.coffee_raw = coffee_raw
        self.planning_out = planning_out
        self.coffee_out = coffee_out
        self.transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)

    def transform_planning_data(self, df: pd.DataFrame) -> gpd.GeoDataFrame:
        # Filter out records with invalid BNG coordinates (mixed-CRS records).
        valid_records = df[
            (df["centroid_easting"] >= 0)
            & (df["centroid_easting"] <= 700_000)
            & (df["centroid_northing"] >= 0)
            & (df["centroid_northing"] <= 1_300_000)
        ].copy()

        lon, lat = self.transformer.transform(
            valid_records["centroid_easting"].values,
            valid_records["centroid_northing"].values,
        )
        valid_records["latitude"] = lat
        valid_records["longitude"] = lon
        valid_records["geometry"] = gpd.points_from_xy(lon, lat)
        valid_records = gpd.GeoDataFrame(valid_records, crs="EPSG:4326")

        valid_records["h3_index"] = [
            h3.latlng_to_cell(row_lat, row_lon, H3_RESOLUTION)
            for row_lat, row_lon in zip(
                valid_records["latitude"], valid_records["longitude"], strict=True
            )
        ]
        return valid_records

    def transform_coffee_data(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        updated_data = gdf[
            [
                "geometry", "name", "brand", "cuisine",
                "opening_hours", "start_date", "addr:postcode", "operator",
            ]
        ].copy()

        # Project to BNG for a metrically-correct centroid, then back to WGS84.
        projected = updated_data.geometry.to_crs("EPSG:27700")
        updated_data["latitude"] = projected.centroid.to_crs("EPSG:4326").y
        updated_data["longitude"] = projected.centroid.to_crs("EPSG:4326").x

        updated_data["h3_index"] = [
            h3.latlng_to_cell(row_lat, row_lon, H3_RESOLUTION)
            for row_lat, row_lon in zip(
                updated_data["latitude"], updated_data["longitude"], strict=True
            )
        ]
        return updated_data

    def run(self) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        logger.info("Starting data transformation process...")

        if os.path.exists(self.planning_out) and os.path.exists(self.coffee_out):
            logger.info("Processed files already exist. Loading from disk...")
            return gpd.read_parquet(self.planning_out), gpd.read_parquet(self.coffee_out)

        planning_df = pd.read_parquet(self.planning_raw)
        coffee_df = gpd.read_parquet(self.coffee_raw)

        transformed_planning_df = self.transform_planning_data(planning_df)
        transformed_coffee_df = self.transform_coffee_data(coffee_df)

        os.makedirs(os.path.dirname(self.planning_out) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(self.coffee_out) or ".", exist_ok=True)
        transformed_planning_df.to_parquet(self.planning_out, index=False)
        transformed_coffee_df.to_parquet(self.coffee_out, index=False)

        logger.info("Data transformation process completed successfully.")
        return transformed_planning_df, transformed_coffee_df
