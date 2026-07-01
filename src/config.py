import datetime
from dateutil.relativedelta import relativedelta

PLANNING_API_URL = "https://planninglondondatahub.london.gov.uk/api-guest/applications/_search?scroll=1m"
SCROLL_API_URL = "https://planninglondondatahub.london.gov.uk/api-guest/_search/scroll"
SCROLL_DURATION = "1m"
PAGE_SIZE = 10000
# All 33 London boroughs. Names must match the API's `lpa_name.raw` values
# exactly (verified via the aggregation query in notebooks/01_eda.ipynb) —
# note the ampersands and the absence of "upon Thames"/"City of" on some.
# The LLDC and OPDC development corporations are intentionally excluded.
TARGET_BOROUGHS = [
    "Barking & Dagenham",
    "Barnet",
    "Bexley",
    "Brent",
    "Bromley",
    "Camden",
    "City of London",
    "Croydon",
    "Ealing",
    "Enfield",
    "Greenwich",
    "Hackney",
    "Hammersmith & Fulham",
    "Haringey",
    "Harrow",
    "Havering",
    "Hillingdon",
    "Hounslow",
    "Islington",
    "Kensington & Chelsea",
    "Kingston",
    "Lambeth",
    "Lewisham",
    "Merton",
    "Newham",
    "Redbridge",
    "Richmond",
    "Southwark",
    "Sutton",
    "Tower Hamlets",
    "Waltham Forest",
    "Wandsworth",
    "Westminster"
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
    "ward",
    "last_updated",
    "last_synced"
]
OSMNX_PLACE_NAME = "Greater London, England, United Kingdom"
# Per-borough parquet files are written here for resumable ingestion;
# they are concatenated into PLANNING_RAW_PATH once every borough is fetched.
PLANNING_RAW_DIR = "data/raw/planning"
PLANNING_RAW_PATH = "data/raw/planning_applications.parquet"
COFFEE_SHOPS_RAW_PATH = "data/raw/coffee_shops.parquet"

REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 5
RATE_LIMIT_DELAY = 10 # seconds, between scroll batches within a borough
RETRY_DELAY = 30      # seconds, fallback backoff on 429/503 (when no Retry-After)
BOROUGH_DELAY = 5     # seconds, courtesy pause between boroughs
USER_AGENT = "Neighbourhood-Pulse/0.1 (research project)"

H3_RESOLUTION = 8
PLANNING_PROCESSED_PATH = "data/processed/planning_processed.parquet"
COFFEE_SHOPS_PROCESSED_PATH = "data/processed/coffee_shops_processed.parquet"

CHAIN_BRANDS = ['Costa', 'Starbucks', 'Caffè Nero', 'Joe & The Juice',
                'Wild Bean Cafe', 'Morrisons', 'Asda', 'T4', 'Mooboo']

# --- Feature engineering: reporting-lag trim ---
# Planning records appear with an ingestion lag, so the most recent month(s) are
# under-reported. Trim a CONTIGUOUS run of trailing months whose count falls
# below LAG_TRIM_FRACTION of the median monthly count. Lag is monotonic from the
# trailing edge, so only a contiguous tail is trimmed — an interior dip violates
# that monotonicity and is therefore real signal, not lag, and is kept.
# MAX_LAG_MONTHS bounds the blast radius: a longer run is implausible as lag
# (likely a genuine decline) and is flagged rather than silently trimmed.
LAG_TRIM_FRACTION = 0.75
MAX_LAG_MONTHS = 4