"""Planning-application and café data acquisition.

Network-facing stage. Planning applications come from the Planning London
Datahub's Elasticsearch guest API via scroll pagination; cafés come from
OpenStreetMap via OSMnx. Resumable by design: each borough is saved to its
own parquet as it completes, so a crash loses only the in-flight borough and
a re-run skips everything already on disk.
"""

import glob
import logging
import os
import re
import time

import geopandas as gpd
import osmnx
import pandas as pd
import requests

from neighbourhood_pulse.config import (
    BOROUGH_DELAY,
    COFFEE_SHOPS_RAW_PATH,
    END_DATE,
    FIELDS_TO_RETURN,
    MAX_RETRIES,
    OSMNX_PLACE_NAME,
    PAGE_SIZE,
    PLANNING_API_URL,
    PLANNING_RAW_DIR,
    PLANNING_RAW_PATH,
    RATE_LIMIT_DELAY,
    REQUEST_TIMEOUT,
    RETRY_DELAY,
    SCROLL_API_URL,
    SCROLL_DURATION,
    START_DATE,
    TARGET_BOROUGHS,
    USER_AGENT,
)

logger = logging.getLogger(__name__)


class IngestionError(RuntimeError):
    """Ingestion cannot complete (e.g. retries exhausted, nothing to combine)."""


class DataIngestion:
    """Fetches and saves the two raw datasets.

    Paths and the borough list are injectable so tests can point the class at
    temp directories and a stubbed API; defaults come from config.
    """

    def __init__(
        self,
        raw_dir: str = PLANNING_RAW_DIR,
        combined_path: str = PLANNING_RAW_PATH,
        coffee_path: str = COFFEE_SHOPS_RAW_PATH,
        boroughs: list[str] | None = None,
    ):
        self.raw_dir = raw_dir
        self.combined_path = combined_path
        self.coffee_path = coffee_path
        self.boroughs = TARGET_BOROUGHS if boroughs is None else boroughs
        self.session = requests.Session()
        # Identify the client politely so the API operator can recognise (and
        # contact, rather than silently block) this traffic if needed.
        self.session.headers.update({"User-Agent": USER_AGENT})

    def _build_planning_query(self, borough: str) -> dict:
        """Construct the Elasticsearch query payload for one borough."""
        return {
            "query": {
                "bool": {
                    "must": [
                        {"exists": {"field": "valid_date"}},
                        {
                            "range": {
                                "valid_date": {
                                    "gte": START_DATE.strftime("%d/%m/%Y"),
                                    "lte": END_DATE.strftime("%d/%m/%Y"),
                                }
                            }
                        },
                        {"term": {"lpa_name.raw": borough}},
                    ],
                    "must_not": [{"term": {"status.raw": "Withdrawn"}}],
                }
            },
            "_source": FIELDS_TO_RETURN,
            "size": PAGE_SIZE,
        }

    def _post_with_retry(self, url: str, payload: dict) -> requests.Response:
        retries = 0
        while retries < MAX_RETRIES:
            response = self.session.post(url, json=payload, timeout=REQUEST_TIMEOUT)
            # 429 = rate-limited, 503 = unavailable. Back off and retry rather
            # than crashing; honour the server's Retry-After hint if given.
            if response.status_code in (429, 503):
                retry_after = response.headers.get("Retry-After")
                try:
                    wait = int(retry_after) if retry_after is not None else RETRY_DELAY
                except ValueError:
                    # Retry-After may be an HTTP-date rather than seconds.
                    wait = RETRY_DELAY
                logger.warning(
                    "%s received. Backing off %ss and retrying (%s/%s)...",
                    response.status_code,
                    wait,
                    retries + 1,
                    MAX_RETRIES,
                )
                time.sleep(wait)
                retries += 1
            else:
                response.raise_for_status()
                return response
        raise IngestionError(f"Max retries ({MAX_RETRIES}) exceeded for {url}")

    def fetch_planning_data(self, borough: str) -> list:
        logger.info("Fetching planning data for borough: %s...", borough)
        query = self._build_planning_query(borough)
        try:
            response = self._post_with_retry(PLANNING_API_URL, query)
            data = response.json()

            scroll_id = data.get("_scroll_id")
            batch = data.get("hits", {}).get("hits", [])
            all_records = list(batch)

            while batch:
                logger.info(
                    "Fetching next batch for %s. Total so far: %s", borough, len(all_records)
                )
                time.sleep(RATE_LIMIT_DELAY)
                scroll_payload = {"scroll": SCROLL_DURATION, "scroll_id": scroll_id}
                scroll_response = self._post_with_retry(SCROLL_API_URL, scroll_payload)
                data_scroll = scroll_response.json()
                scroll_id = data_scroll.get("_scroll_id")
                batch = data_scroll.get("hits", {}).get("hits", [])
                all_records.extend(batch)

            logger.info("Fetched %s records for %s", len(all_records), borough)
            return all_records
        except requests.exceptions.RequestException as e:
            # Chain rather than wrap-and-hide: the original traceback survives.
            raise IngestionError(f"Network error while fetching {borough}") from e

    def fetch_coffee_shop_data(self) -> gpd.GeoDataFrame:
        # No try/except: we can't handle OSMnx failures here, and a wrapper
        # would only obscure the original exception.
        logger.info("Fetching coffee shop data for %s...", OSMNX_PLACE_NAME)
        return osmnx.features_from_place(OSMNX_PLACE_NAME, tags={"amenity": "cafe"})

    @staticmethod
    def _borough_slug(borough: str) -> str:
        """Filesystem-safe, stable slug: 'Barking & Dagenham' -> 'Barking_Dagenham'."""
        return re.sub(r"[^\w]+", "_", borough).strip("_")

    def _process_and_save_planning_data(self, all_records: list, output_path: str) -> pd.DataFrame:
        """Flatten and save one borough. Callers must pass a non-empty list;
        empty boroughs are skipped upstream so no malformed file is written."""
        planning_df = pd.DataFrame([record.get("_source", {}) for record in all_records])
        planning_df["centroid_northing"] = pd.to_numeric(
            planning_df["centroid_northing"], errors="coerce"
        )
        planning_df["centroid_easting"] = pd.to_numeric(
            planning_df["centroid_easting"], errors="coerce"
        )
        planning_df = planning_df.convert_dtypes()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        planning_df.to_parquet(output_path, index=False)
        logger.info("Planning data saved to %s", output_path)
        return planning_df

    def _process_and_save_coffee_data(self, coffee_df: pd.DataFrame) -> gpd.GeoDataFrame:
        coffee_df = gpd.GeoDataFrame(coffee_df)
        os.makedirs(os.path.dirname(self.coffee_path) or ".", exist_ok=True)
        coffee_df.to_parquet(self.coffee_path)
        logger.info("Coffee shop data saved to %s", self.coffee_path)
        return coffee_df

    def run(self) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
        logger.info("Starting data ingestion process...")

        # 1. Planning: per-borough fetch with resume. The combined file is the
        # "done" marker — if you ADD boroughs later, delete combined_path; the
        # per-borough files are skipped, so only new boroughs are fetched.
        if os.path.exists(self.combined_path):
            logger.info("Found combined planning data at %s.", self.combined_path)
            planning_df = pd.read_parquet(self.combined_path)
        else:
            os.makedirs(self.raw_dir, exist_ok=True)
            for borough in self.boroughs:
                borough_path = os.path.join(self.raw_dir, f"{self._borough_slug(borough)}.parquet")
                if os.path.exists(borough_path):
                    logger.info("Skipping %s: already saved.", borough)
                    continue
                borough_records = self.fetch_planning_data(borough)
                # A zero-record borough is almost always an anomaly (lpa_name
                # term mismatch), not a real result — don't cache it as "done".
                if not borough_records:
                    logger.warning("No records for %s; not saving (retries next run).", borough)
                    continue
                self._process_and_save_planning_data(borough_records, borough_path)
                time.sleep(BOROUGH_DELAY)  # courtesy pause between boroughs

            borough_files = glob.glob(os.path.join(self.raw_dir, "*.parquet"))
            if not borough_files:
                raise IngestionError("No per-borough planning files found to combine.")
            logger.info("Combining %s borough files.", len(borough_files))
            planning_df = pd.concat((pd.read_parquet(f) for f in borough_files), ignore_index=True)
            os.makedirs(os.path.dirname(self.combined_path) or ".", exist_ok=True)
            planning_df.to_parquet(self.combined_path, index=False)
            logger.info("Combined planning data saved (%s records).", len(planning_df))

        # 2. Coffee shops: independent idempotency check.
        if os.path.exists(self.coffee_path):
            logger.info("Found existing coffee shop data at %s.", self.coffee_path)
            coffee_shops_df = gpd.read_parquet(self.coffee_path)
        else:
            raw_coffee_df = self.fetch_coffee_shop_data()
            coffee_shops_df = self._process_and_save_coffee_data(raw_coffee_df)

        logger.info("Data ingestion process completed successfully.")
        return planning_df, coffee_shops_df
