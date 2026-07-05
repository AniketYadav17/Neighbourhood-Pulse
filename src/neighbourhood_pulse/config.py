"""All constants and configuration.

Values with non-obvious provenance carry comments; borough names must match
the API's `lpa_name.raw` values exactly (verified via the aggregation query
recorded in notebooks/01_eda.ipynb section 7).
"""

import datetime

from dateutil.relativedelta import relativedelta

from neighbourhood_pulse import __version__

PLANNING_API_URL = (
    "https://planninglondondatahub.london.gov.uk/api-guest/applications/_search?scroll=1m"
)
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
    "Westminster",
]
END_DATE = datetime.date.today()
START_DATE = END_DATE - relativedelta(years=5)
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
    "last_synced",
]
OSMNX_PLACE_NAME = "Greater London, England, United Kingdom"
# Per-borough parquet files are written here for resumable ingestion;
# they are concatenated into PLANNING_RAW_PATH once every borough is fetched.
PLANNING_RAW_DIR = "data/raw/planning"
PLANNING_RAW_PATH = "data/raw/planning_applications.parquet"
COFFEE_SHOPS_RAW_PATH = "data/raw/coffee_shops.parquet"

REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 5
RATE_LIMIT_DELAY = 10  # seconds, between scroll batches within a borough
RETRY_DELAY = 30  # seconds, fallback backoff on 429/503 (when no Retry-After)
BOROUGH_DELAY = 5  # seconds, courtesy pause between boroughs
USER_AGENT = f"Neighbourhood-Pulse/{__version__} (research project)"

H3_RESOLUTION = 8
PLANNING_PROCESSED_PATH = "data/processed/planning_processed.parquet"
COFFEE_SHOPS_PROCESSED_PATH = "data/processed/coffee_shops_processed.parquet"

CHAIN_BRANDS = [
    "Costa",
    "Starbucks",
    "Caffè Nero",
    "Joe & The Juice",
    "Wild Bean Cafe",
    "Morrisons",
    "Asda",
    "T4",
    "Mooboo",
]

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

# Silent-mass-drop guard: load_planning raises if more than this fraction of
# rows would be dropped (unparseable date / null h3). Real data loses <<1%.
PLANNING_MAX_DROP_FRACTION = 0.01

# --- Phase B: modelling pipeline ---

# Intermediates (gitignored, data/processed) and final artifacts (artifacts/).
HEX_FEATURES_PATH = "data/processed/hex_features.parquet"
LR_SALES_PATH = "data/processed/lr_london_sales.parquet"
HEX_PRICE_TARGET_PATH = "data/processed/hex_price_target.parquet"
HEX_TRAINING_PATH = "data/processed/hex_training.parquet"
ARTIFACTS_DIR = "artifacts"
VALUATION_GAP_PATH = "artifacts/hex_valuation_gap.parquet"
METRICS_PATH = "artifacts/metrics.json"
MODEL_PATH = "artifacts/model.joblib"

# Land Registry Price Paid CSVs (national, ~870 MB total, data/raw).
LR_RAW_DIR = "data/raw"
LR_YEARS = (2021, 2022, 2023, 2024, 2025)
# LR spells a few boroughs in full; map config names (upper, & -> AND) to LR district names.
NAME_FIX = {
    "KINGSTON": "KINGSTON UPON THAMES",
    "RICHMOND": "RICHMOND UPON THAMES",
    "WESTMINSTER": "CITY OF WESTMINSTER",
}
RESIDENTIAL_PTYPES = ("D", "S", "T", "F")  # detached/semi/terraced/flat; drop "O" (other)
MIN_SALES_PER_HEX = 30  # pooled-sales floor for a stable per-hexagon median

RECENT_WINDOW_MONTHS = 12  # "recent" window for applications_recent / velocity

# Back-test windows: early gap (2021-22) vs subsequent growth (2024-25).
BACKTEST_EARLY_YEARS = (2021, 2022)
BACKTEST_LATE_YEARS = (2024, 2025)
BACKTEST_MIN_SALES = 15  # per window, per hexagon

# Centrality control: haversine distance to Charing Cross.
CX_LAT, CX_LON = 51.507, -0.1278

# Model feature set (order matters for the saved model's predict).
FEATURE_COLS = [
    "total_applications",
    "change_of_use_count",
    "applications_recent",
    "change_of_use_ratio",
    "planning_velocity",
    "total_cafe_count",
    "independent_cafe_count",
    "cafe_to_application_ratio",
    "dist_to_centre_km",
]

# --- Phase C: precomputed neighbourhood briefs (Gemini API, build time only) ---

BRIEFS_PATH = "artifacts/briefs.json"
BRIEFS_N_HEXAGONS = 50  # the most-undervalued hexagons — what app users click first
# Model choice (documented cost/quality trade-off): the briefs are short, tightly
# grounded generations with a strict JSON schema — flash-class work, and the free
# tier covers a full run at $0. gemini-2.5-flash ($0.30/$2.50 per MTok) is the
# cheaper stable flash if the paid tier is ever hit; gemini-3.5-flash (current
# stable flash, used here) is the quality upgrade at ~5x the price ($1.50/$9.00),
# and gemini-3.1-pro-class models are the further step up for harder reasoning.
# A full 50-brief run costs pennies on either paid tier; free tier is $0 either way.
# (The Claude/Anthropic version of this module lives in git history at commit
# 9393202 if ever needed again.)
BRIEFS_MODEL = "gemini-3.5-flash"
# gemini-3.5-flash is a "thinking" model: max_output_tokens is a shared cap over
# thinking tokens AND the visible completion, not the completion alone. The
# gemini-3 family also replaces the older thinking_budget (token count, 0 =
# off) with thinking_level ("minimal"/"low"/"medium"/"high"); "minimal" is the
# lowest available level for this model family — thinking cannot be fully
# disabled, only minimized. 400 tokens was consumed entirely by thought before
# any JSON was emitted (every response hit finish_reason=MAX_TOKENS); 4000
# gives generous headroom for minimal-level thinking plus the ~200-token brief.
BRIEFS_MAX_TOKENS = 4000
# briefs are formulaic; thinking adds cost/latency with no quality gain here
BRIEFS_THINKING_LEVEL = "minimal"
BRIEFS_TEMPERATURE = 0.2
BRIEFS_MAX_COST_USD = 1.00  # hard stop; an actual full run is ~$0.06 on the paid tier
BRIEFS_PRICE_PER_MTOK = {"input": 1.50, "output": 9.00}  # gemini-3.5-flash, 2026-07
# Free tier allows 5 requests/min (GenerateRequestsPerMinutePerProjectPerModel-
# FreeTier) for gemini-3.5-flash; 13s spacing (~4.6 req/min) stays under that
# with margin. The server's 429 retryDelay is ~50s — far longer than the SDK's
# default backoff ceiling — so pacing requests up front avoids relying on
# retry-after-the-fact entirely. Set to 0 on a paid tier (no RPM cap there).
BRIEFS_MIN_REQUEST_INTERVAL_S = 13.0
