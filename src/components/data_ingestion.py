import os
import time
import requests
import osmnx
import pandas as pd
import geopandas as gpd

from src.logger import logger
from src.exceptions import CustomException
from src.config import (
    PLANNING_API_URL, SCROLL_API_URL, START_DATE, END_DATE,
    FIELDS_TO_RETURN, PAGE_SIZE, SCROLL_DURATION,
    OSMNX_PLACE_NAME, PLANNING_RAW_PATH, COFFEE_SHOPS_RAW_PATH,
    TARGET_BOROUGHS, REQUEST_TIMEOUT, MAX_RETRIES, RATE_LIMIT_DELAY, RETRY_DELAY
)

class DataIngestion:

    def __init__(self):
        logger.info("Initialising DataIngestion class...")
        self.session = requests.Session()

    def _build_planning_query(self, borough: str) -> dict:
        """Helper method to construct the Elasticsearch query payload."""
        return {
            "query": {
                "bool": {
                    "must": [
                        {"exists": {"field": "valid_date"}},
                        {
                            "range": {
                                "valid_date": {
                                    "gte": START_DATE.strftime("%d/%m/%Y"),
                                    "lte": END_DATE.strftime("%d/%m/%Y")
                                }
                            }
                        },
                        {"term": {"lpa_name.raw": borough}}
                    ],
                    "must_not": [
                        {"term": {"status.raw": "Withdrawn"}}
                    ]
                }
            },
            "_source": FIELDS_TO_RETURN,
            "size": PAGE_SIZE
        }
    
    def _post_with_retry(self, url: str, payload: dict) -> requests.Response:
        retries = 0
        while retries < MAX_RETRIES:
            response = self.session.post(url, json=payload, timeout=REQUEST_TIMEOUT)
            if response.status_code == 503:
                logger.warning(f"503 received. Retrying ({retries + 1}/{MAX_RETRIES})...")
                time.sleep(RETRY_DELAY)
                retries += 1
            else:
                response.raise_for_status()
                return response
        logger.error("Max retries reached.")
        raise CustomException("Max retries exceeded")

    def fetch_planning_data(self, borough: str) -> list:
        logger.info(f"Fetching planning data for borough: {borough}...")
        query = self._build_planning_query(borough)
        
        try:
            response = self._post_with_retry(PLANNING_API_URL, query)
            data = response.json()
            
            scroll_id = data.get("_scroll_id")
            # Safe dict extraction using .get()
            batch = data.get("hits", {}).get("hits", []) 
            
            # Explicitly cast to a new list to prevent reference bugs
            all_records = list(batch) 
            
            while batch:
                logger.info(f"Fetching next batch for {borough}. Total so far: {len(all_records)}")
                time.sleep(RATE_LIMIT_DELAY)
                
                scroll_payload = {"scroll": SCROLL_DURATION, "scroll_id": scroll_id}
                scroll_response = self._post_with_retry(SCROLL_API_URL, scroll_payload)
                
                data_scroll = scroll_response.json()
                scroll_id = data_scroll.get("_scroll_id")
                batch = data_scroll.get("hits", {}).get("hits", [])
                all_records.extend(batch)
                
            logger.info(f"Successfully fetched {len(all_records)} records for {borough}")
            return all_records

        # Specific exception handling rather than a blanket catch-all
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error while fetching planning data: {e}")
            raise CustomException(e) 

    def fetch_coffee_shop_data(self) -> pd.DataFrame:
        logger.info(f"Fetching coffee shop data for {OSMNX_PLACE_NAME}...")
        try:
            return osmnx.features_from_place(OSMNX_PLACE_NAME, tags={"amenity": "cafe"})
        except Exception as e:
            logger.error(f"Error fetching OSMnx data: {e}")
            raise CustomException(e)
            
    def _process_and_save_planning_data(self, all_records: list) -> pd.DataFrame:
        """Isolates DataFrame transformation and saving logic."""
        if not all_records:
            logger.warning("No planning records found. Creating empty DataFrame.")
            planning_df = pd.DataFrame(columns=["centroid_northing", "centroid_easting"])
        else:
            planning_df = pd.DataFrame([record.get("_source", {}) for record in all_records])
            
        planning_df["centroid_northing"] = pd.to_numeric(planning_df["centroid_northing"], errors='coerce')
        planning_df["centroid_easting"] = pd.to_numeric(planning_df["centroid_easting"], errors='coerce')
        planning_df = planning_df.convert_dtypes()
        
        planning_df.to_parquet(PLANNING_RAW_PATH, index=False)
        logger.info(f"Planning data saved to {PLANNING_RAW_PATH}")
        return planning_df

    def _process_and_save_coffee_data(self, coffee_df: pd.DataFrame) -> pd.DataFrame:
        """Isolates DataFrame transformation and saving logic."""
        # coffee_df = coffee_df.convert_dtypes()
        coffee_df = gpd.GeoDataFrame(coffee_df)
        coffee_df.to_parquet(COFFEE_SHOPS_RAW_PATH)
        logger.info(f"Coffee shop data saved to {COFFEE_SHOPS_RAW_PATH}")
        return coffee_df

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("Starting data ingestion process...")
        
        # 1. Independent Idempotency Check for Planning Data
        if os.path.exists(PLANNING_RAW_PATH):
            logger.info(f"Found existing planning data. Loading from {PLANNING_RAW_PATH}.")
            planning_df = pd.read_parquet(PLANNING_RAW_PATH)
        else:
            all_records = []
            for borough in TARGET_BOROUGHS:
                borough_records = self.fetch_planning_data(borough)
                all_records.extend(borough_records)
            planning_df = self._process_and_save_planning_data(all_records)

        # 2. Independent Idempotency Check for Coffee Shop Data
        if os.path.exists(COFFEE_SHOPS_RAW_PATH):
            logger.info(f"Found existing coffee shop data. Loading from {COFFEE_SHOPS_RAW_PATH}.")
            coffee_shops_df = gpd.read_parquet(COFFEE_SHOPS_RAW_PATH)
        else:
            raw_coffee_df = self.fetch_coffee_shop_data()
            coffee_shops_df = self._process_and_save_coffee_data(raw_coffee_df)

        logger.info("Data ingestion process completed successfully.")
        return planning_df, coffee_shops_df