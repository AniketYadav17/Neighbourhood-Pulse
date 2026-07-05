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
# Model choice, take 2 (see git history at commit 83f5103 and the "Fix: daily
# quota + model switch" entry in .superpowers/sdd/task-10-gemini-report.md for
# the full incident): the first real run against gemini-3.5-flash 429'd on
# every single request with quotaId
# GenerateRequestsPerDayPerProjectPerModel-FreeTier, quotaValue 20 — the
# free-tier daily budget for that model is just 20 requests/day, so a 50-brief
# run can *never* complete on it in one day, no matter how patient the retry
# logic is. ai.google.dev/gemini-api/docs/rate-limits no longer publishes a
# static per-model free-tier table (limits are dynamic/per-project, viewable
# only in AI Studio), so this was cross-checked against multiple independent
# reports (Google AI developer forum + several 2026 rate-limit write-ups):
# flash-lite-tier models consistently get ~15 RPM / ~1000 RPD on the free
# tier — ~50x the daily headroom a 50-brief run + retries needs, vs. the
# standard (non-lite) flash tier where real-world daily quotas have been
# reported anywhere from ~250 down to the ~20 this project actually hit.
# That real-world flakiness rules out gemini-2.5-flash as a candidate despite
# its official pricing-page billing, since it's the same non-lite class as
# the model that just failed.
# Between the two flash-lite options: gemini-3.1-flash-lite is the newer,
# higher-quality pick ("frontier-class performance rivaling larger models" per
# Google's own docs) but is gemini-3.x-family — its thinking mechanism is
# thinking_level, whose floor is "minimal", not zero; thinking cannot be fully
# disabled, so every call spends an unpredictable number of tokens on hidden
# thought before the JSON completion (exactly the mechanism that caused a
# separate max_output_tokens incident on gemini-3.5-flash). gemini-2.5-flash-lite
# is gemini-2.5-family: its thinking_config uses thinking_budget (a token
# count), and 0 genuinely and fully disables thinking — confirmed both by the
# installed SDK's own field docstring ("0 is DISABLED") in
# google/genai/types.py's ThinkingConfig, and by ai.google.dev/gemini-api/docs/thinking
# listing gemini-2.5-flash-lite as the one model in its table whose *default*
# thinking state is "Off". Given the whole point of this fix is eliminating
# hidden-cost/hidden-quota surprises, gemini-2.5-flash-lite's fully-off,
# fully-predictable thinking wins over gemini-3.1-flash-lite's marginal
# quality edge for this short, formulaic, schema-constrained brief-generation
# task — chosen as BRIEFS_MODEL, with gemini-3.1-flash-lite documented here as
# the runner-up quality trade-off (same free-tier quota class, ~2.5x the
# paid-tier price: $0.25/$1.50 vs. $0.10/$0.40 per MTok).
BRIEFS_MODEL = "gemini-2.5-flash-lite"
# With thinking fully disabled (thinking_budget=0, see BRIEFS_THINKING_CONFIG
# below), max_output_tokens covers only the actual JSON completion — no hidden
# thought tokens can eat into it the way they did on gemini-3.5-flash (which
# needed 4000 for that reason). The brief itself is a ~12-word headline plus a
# 2-3 sentence body plus a 1-sentence caveat: comfortably under 200 tokens in
# practice. 800 leaves 4x headroom for verbose model output without inviting
# runaway generations.
BRIEFS_MAX_TOKENS = 800
# gemini-2.5-flash-lite is gemini-2.5-family: thinking_config takes
# thinking_budget (a token count), where 0 fully disables thinking (confirmed
# via the installed SDK's ThinkingConfig field docstring: "0 is DISABLED. -1
# is AUTOMATIC.") — unlike the gemini-3.x family's thinking_level, which has
# no true "off" (its floor is "minimal"). Briefs are formulaic; thinking adds
# cost, latency, and (per the incident this fix addresses) token-budget risk
# with no quality gain here. Kept as a single dict (rather than a bare level
# string) so the whole thinking-config shape travels with the model choice —
# switching models later only means swapping this constant.
BRIEFS_THINKING_CONFIG = {"thinking_budget": 0}
BRIEFS_TEMPERATURE = 0.2
BRIEFS_MAX_COST_USD = 1.00  # hard stop; an actual full run is well under $0.01 on the paid tier
BRIEFS_PRICE_PER_MTOK = {"input": 0.10, "output": 0.40}  # gemini-2.5-flash-lite, 2026-07
# Observed free-tier RPM for gemini-2.5-flash-lite is 10 (quotaValue from a
# real 429 on 2026-07-05; ai.google.dev/gemini-api/docs/rate-limits no longer
# publishes a static per-model table). 60/10 = 6.0s minimum spacing; +10%
# margin = 6.6s. Headroom matters doubly here because the SDK's own 503
# retries also count against the RPM budget. Set to 0 on a paid tier.
BRIEFS_MIN_REQUEST_INTERVAL_S = 6.6
