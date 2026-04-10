import datetime
from dateutil.relativedelta import relativedelta

PLANNING_API_URL = "https://planninglondondatahub.london.gov.uk/api-guest/applications/_search?scroll=1m"
SCROLL_API_URL = "https://planninglondondatahub.london.gov.uk/api-guest/_search/scroll"
SCROLL_DURATION = "1m"
PAGE_SIZE = 10000
TARGET_BOROUGHS = [
    "Newham",
    "Redbridge",
    "Waltham Forest"
]
END_DATE = datetime.date.today()
START_DATE = END_DATE-relativedelta(years=5)
FIELDS_TO_RETURN = [
    "id",
    "lpa_name",
    "lpa_app_no",
    "borough",
    "status",
    "valid_date",
    "decision",
    "decision_date",
    "description",
    "application_type",
    "site_name",
    "site_number",
    "street_name",
    "locality",
    "application_details.residential_details",
    "decision_target_date",
    "url_planning_app",
    "appeal_decision",
    "appeal_decision_date",
    "polygon",
    "uprn",
    "postcode",
    "centroid_easting",
    "centroid_northing",
    "ward"
]
OSMNX_PLACE_NAME = "Greater London, England, United Kingdom"
PLANNING_RAW_PATH = "data/raw/planning_applications.parquet"
COFFEE_SHOPS_RAW_PATH = "data/raw/coffee_shops.parquet"

REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 5
RATE_LIMIT_DELAY = 10 # seconds
RETRY_DELAY = 30      # seconds

H3_RESOLUTION = 8
PLANNING_PROCESSED_PATH = "data/processed/planning_processed.parquet"
COFFEE_SHOPS_PROCESSED_PATH = "data/processed/coffee_shops_processed.parquet"