from pyproj import Transformer
import pandas as pd
import geopandas as gpd
import h3
import os

from src.logger import logger
from src.exceptions import CustomException
from src.config import (
    PLANNING_RAW_PATH,
    COFFEE_SHOPS_RAW_PATH,
    H3_RESOLUTION, 
    PLANNING_PROCESSED_PATH, 
    COFFEE_SHOPS_PROCESSED_PATH
)

class DataTransformation:

    def __init__(self):
        logger.info("Initialising Data Transformation class...")
        self.transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)

    def transform_planning_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Filter out records with invalid BNG coordinates
            valid_records = df[
                (df['centroid_easting'] >= 0) & 
                (df['centroid_easting'] <= 700000) &
                (df['centroid_northing'] >= 0) & 
                (df['centroid_northing'] <= 1300000)
            ].copy()
            
            lon, lat = self.transformer.transform(
                valid_records['centroid_easting'].values, 
                valid_records['centroid_northing'].values
            )

            valid_records['latitude'] = lat
            valid_records['longitude'] = lon

            # Create geometry column for GeoDataFrame
            valid_records['geometry'] = gpd.points_from_xy(lon, lat)
            valid_records = gpd.GeoDataFrame(valid_records, crs="EPSG:4326")

            # Calculate H3 index for each record
            valid_records['h3_index'] = [
            h3.latlng_to_cell(row_lat, row_lon, H3_RESOLUTION)
            for row_lat, row_lon in zip(valid_records['latitude'], valid_records['longitude'])
                                        ]

            return valid_records

        except Exception as e:
            logger.error(f"Error occurred while transforming planning data: {e}")
            raise CustomException(e)

    def transform_coffee_data(self, gdf: gpd.GeoDataFrame):
        try:
            updated_data = gdf[[
                "geometry", "name", "brand", "cuisine", 
                "opening_hours", "start_date", "addr:postcode", "operator"]].copy()
            
            projected = updated_data.geometry.to_crs("EPSG:27700")
            updated_data['latitude'] = projected.centroid.to_crs("EPSG:4326").y
            updated_data['longitude'] = projected.centroid.to_crs("EPSG:4326").x

            updated_data['h3_index'] = [
            h3.latlng_to_cell(row_lat, row_lon, H3_RESOLUTION)
            for row_lat, row_lon in zip(updated_data['latitude'], updated_data['longitude'])
                                        ]
            return updated_data
            

        except Exception as e:
            logger.error(f"Error occurred while transforming coffee data: {e}")
            raise CustomException(e)

    def run(self):
        try:
            logger.info("Starting data transformation process...")

            if os.path.exists(PLANNING_PROCESSED_PATH) and os.path.exists(COFFEE_SHOPS_PROCESSED_PATH):
                logger.info("Processed files already exist. Loading from disk...")
                return gpd.read_parquet(PLANNING_PROCESSED_PATH), gpd.read_parquet(COFFEE_SHOPS_PROCESSED_PATH)

            planning_df = pd.read_parquet(PLANNING_RAW_PATH)
            coffee_df = gpd.read_parquet(COFFEE_SHOPS_RAW_PATH)

            transformed_planning_df = self.transform_planning_data(planning_df)
            transformed_coffee_df = self.transform_coffee_data(coffee_df)

            os.makedirs(os.path.dirname(PLANNING_PROCESSED_PATH), exist_ok=True)
            os.makedirs(os.path.dirname(COFFEE_SHOPS_PROCESSED_PATH), exist_ok=True)

            transformed_planning_df.to_parquet(PLANNING_PROCESSED_PATH, index=False)
            transformed_coffee_df.to_parquet(COFFEE_SHOPS_PROCESSED_PATH, index=False)

            logger.info("Data transformation process completed successfully.")
            return transformed_planning_df, transformed_coffee_df
        except Exception as e:
            logger.error(f"Error occurred during data transformation: {e}")
            raise CustomException(e)