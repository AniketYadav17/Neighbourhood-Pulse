import os
import requests
import osmnx
import pandas as pd
from src.logger import logger
from src.exceptions import CustomException
from src.config import *

class DataIngestion:
    def __init__(self):
        logger.info("Initializing DataIngestion class...")
        self.s = requests.Session()

    def fetch_planning_data(self, borough):
        try:
            logger.info(f"Fetching planning data for borough: {borough}...")
            query = {"query":
                    {"bool":
                    {"must":
                    [
                        {"exists":
                            {"field":"valid_date"}},
                            {"range":
                            {"valid_date":
                            {"gte":START_DATE.strftime("%d/%m/%Y"),"lte":END_DATE.strftime("%d/%m/%Y")}}},
                            {"term":{"lpa_name.raw": borough}}
                            ],
                            "must_not":
                            [
                                {"term":{"status.raw":"Withdrawn"}}
                                ]}},
                                "_source":FIELDS_TO_RETURN,
                                "size":PAGE_SIZE
                                }
            post = self.s.post(PLANNING_API_URL, json=query)
            post.raise_for_status()
            data = post.json()
            scroll_id = data["_scroll_id"]
            batch = data["hits"]["hits"]

            all_records = batch
            while batch:
                logger.info(f"Fetching next batch of data for borough: {borough}...")
                post_scroll = self.s.post(SCROLL_API_URL, json={"scroll": SCROLL_DURATION, "scroll_id": scroll_id})
                post_scroll.raise_for_status()
                data_scroll = post_scroll.json()
                scroll_id = data_scroll["_scroll_id"]
                batch = data_scroll["hits"]["hits"]
                all_records.extend(batch)
            logger.info(f"Fetched {len(all_records)} records for borough: {borough}")
            return all_records

        except Exception as e:
            logger.error(f"Error while creating the query: {e}")
            raise CustomException(e)
            

    def fetch_coffee_shop_data(self):
        logger.info("Fetching coffee shop data...")
        return osmnx.features_from_place(OSMNX_PLACE_NAME, tags={"amenity": "cafe"})

    def run(self):
        logger.info("Starting data ingestion process...")
        all_records = []
        for borough in TARGET_BOROUGHS:
            borough_records = self.fetch_planning_data(borough)
            all_records.extend(borough_records)
        df1 = pd.DataFrame([record["_source"] for record in all_records])
        df2 = self.fetch_coffee_shop_data()

        df1.to_parquet(PLANNING_RAW_PATH, index=False)
        logger.info(f"Planning data saved to {PLANNING_RAW_PATH}")
        df2.to_parquet(COFFEE_SHOPS_RAW_PATH, index=False)
        logger.info(f"Coffee shop data saved to {COFFEE_SHOPS_RAW_PATH}")

        logger.info("Data ingestion process completed.")
        return df1, df2
    