import os
import re
import glob
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
    OSMNX_PLACE_NAME, PLANNING_RAW_DIR, PLANNING_RAW_PATH, COFFEE_SHOPS_RAW_PATH,
    TARGET_BOROUGHS, REQUEST_TIMEOUT, MAX_RETRIES, RATE_LIMIT_DELAY, RETRY_DELAY,
    BOROUGH_DELAY, USER_AGENT
)

class DataIngestion:

    def __init__(self):
        logger.info("Initialising DataIngestion class...")
        self.session = requests.Session()
        # Identify the client politely so the API operator can recognise (and
        # contact, rather than silently block) this traffic if needed.
        self.session.headers.update({"User-Agent": USER_AGENT})

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
            # 429 = rate-limited, 503 = service unavailable. Back off and retry
            # rather than crashing; honour the server's Retry-After hint if given.
            if response.status_code in (429, 503):
                retry_after = response.headers.get("Retry-After")
                try:
                    wait = int(retry_after) if retry_after is not None else RETRY_DELAY
                except ValueError:
                    # Retry-After may be an HTTP-date rather than seconds; fall back.
                    wait = RETRY_DELAY
                logger.warning(
                    f"{response.status_code} received. Backing off {wait}s and "
                    f"retrying ({retries + 1}/{MAX_RETRIES})..."
                )
                time.sleep(wait)
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
            
    @staticmethod
    def _borough_slug(borough: str) -> str:
        """Convert a borough name into a stable, filesystem-safe slug.

        'Barking & Dagenham' -> 'Barking_Dagenham'
        'Kensington & Chelsea' -> 'Kensington_Chelsea'
        'City of London' -> 'City_of_London'
        """
        return re.sub(r"[^\w]+", "_", borough).strip("_")

    def _process_and_save_planning_data(self, all_records: list, output_path: str) -> pd.DataFrame:
        """Isolates DataFrame transformation and saving logic for one borough.

        Callers must pass a non-empty record list; empty boroughs are skipped
        upstream in run() so no malformed (column-mismatched) file is written.
        """
        planning_df = pd.DataFrame([record.get("_source", {}) for record in all_records])

        planning_df["centroid_northing"] = pd.to_numeric(planning_df["centroid_northing"], errors='coerce')
        planning_df["centroid_easting"] = pd.to_numeric(planning_df["centroid_easting"], errors='coerce')
        planning_df = planning_df.convert_dtypes()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        planning_df.to_parquet(output_path, index=False)
        logger.info(f"Planning data saved to {output_path}")
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

        # 1. Planning Data: per-borough fetch with resume capability.
        #
        # The combined file is the "done" marker. Note the trade-off: if you
        # ADD boroughs to TARGET_BOROUGHS after this file already exists, this
        # check short-circuits and the new boroughs are never fetched. To pick
        # them up, delete PLANNING_RAW_PATH — the per-borough files already on
        # disk are skipped, so only the genuinely new boroughs are fetched
        # before the dataset is recombined.
        if os.path.exists(PLANNING_RAW_PATH):
            # The combined dataset already exists — ingestion is fully done.
            logger.info(f"Found combined planning data. Loading from {PLANNING_RAW_PATH}.")
            planning_df = pd.read_parquet(PLANNING_RAW_PATH)
        else:
            os.makedirs(PLANNING_RAW_DIR, exist_ok=True)

            # Fetch and save each borough independently, skipping any already on disk.
            # A crash mid-run only loses the in-flight borough; a re-run resumes here.
            for borough in TARGET_BOROUGHS:
                borough_path = os.path.join(PLANNING_RAW_DIR, f"{self._borough_slug(borough)}.parquet")
                if os.path.exists(borough_path):
                    logger.info(f"Skipping {borough}: already saved at {borough_path}.")
                    continue
                borough_records = self.fetch_planning_data(borough)
                # A zero-record borough is almost always an anomaly (e.g. an
                # lpa_name term mismatch), not a real result — don't cache it as
                # "done" with an empty file (which would also break the concat
                # schema). Skip without saving so it stays visible and re-runs.
                if not borough_records:
                    logger.warning(f"No records returned for {borough}; not saving (will retry next run).")
                    continue
                self._process_and_save_planning_data(borough_records, borough_path)
                # Courtesy pause before moving to the next borough.
                time.sleep(BOROUGH_DELAY)

            # Combine every per-borough file into the final dataset.
            borough_files = glob.glob(os.path.join(PLANNING_RAW_DIR, "*.parquet"))
            if not borough_files:
                raise CustomException("No per-borough planning files found to combine.")

            logger.info(f"Combining {len(borough_files)} borough files into {PLANNING_RAW_PATH}.")
            planning_df = pd.concat(
                (pd.read_parquet(f) for f in borough_files), ignore_index=True
            )
            os.makedirs(os.path.dirname(PLANNING_RAW_PATH), exist_ok=True)
            planning_df.to_parquet(PLANNING_RAW_PATH, index=False)
            logger.info(f"Combined planning data saved to {PLANNING_RAW_PATH} ({len(planning_df)} records).")

        # 2. Independent Idempotency Check for Coffee Shop Data
        if os.path.exists(COFFEE_SHOPS_RAW_PATH):
            logger.info(f"Found existing coffee shop data. Loading from {COFFEE_SHOPS_RAW_PATH}.")
            coffee_shops_df = gpd.read_parquet(COFFEE_SHOPS_RAW_PATH)
        else:
            raw_coffee_df = self.fetch_coffee_shop_data()
            coffee_shops_df = self._process_and_save_coffee_data(raw_coffee_df)

        logger.info("Data ingestion process completed successfully.")
        return planning_df, coffee_shops_df