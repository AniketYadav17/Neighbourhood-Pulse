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
# Model choice, take 3 (see git history at commits 83f5103, 2acc6d4 and the
# "Fix: daily quota + model switch" / "Fix: model churn resilience" entries in
# .superpowers/sdd/task-10-gemini-report.md for the full incident history). A
# third real run failed differently again: every call to the take-2 model,
# gemini-2.5-flash-lite, 404'd with `NOT_FOUND: "This model
# models/gemini-2.5-flash-lite is no longer available to new users. Please
# update your code to use a newer model."` — the whole 2.5 family is being
# closed off, and this project has now been broken three times in eight days
# by a model that was "stable" right up until it silently stopped accepting
# new-user traffic.
# Verified live via WebFetch against ai.google.dev/gemini-api/docs/models
# (2026-07-13): gemini-3.1-flash-lite is listed Stable/GA ("Frontier-class
# performance rivaling larger models at a fraction of the cost"), reached GA
# per the docs changelog on 2026-05-07, and is the flash-lite-class model
# actually open to new users right now — gemini-2.5-flash-lite is still
# listed on that same page (the docs evidently lag the live API; the real
# 404 is the ground truth for what's actually cut off, not the docs page).
# Chosen as BRIEFS_MODEL.
# Given three breakages in eight days are now a demonstrated pattern rather
# than a one-off, this model name is deliberately no longer the *only* lever:
# `pulse briefs --model NAME` overrides it end to end (cli.py -> run_briefs ->
# generate_briefs -> the per-brief "model" field) with no code edit needed,
# and a 404 on whichever model is configured is treated as a fatal
# configuration error — caught by a cheap preflight before the run starts
# and, as a backstop, inside the generation loop itself — so a fourth
# breakage degrades to "rerun with --model X" instead of another emergency
# patch (see run_briefs, _model_available, and generate_briefs in briefs.py).
BRIEFS_MODEL = "gemini-3.1-flash-lite"
# gemini-3.1-flash-lite is gemini-3.x-family: per
# ai.google.dev/gemini-api/docs/thinking (verified live via WebFetch,
# 2026-07-13) every gemini-3.x model listed there uses thinking_level, not
# thinking_budget, and none of them list anything below "minimal" — the
# installed SDK's ThinkingLevel enum (google/genai/types.py) itself has no
# "OFF"/"NONE" value at all (THINKING_LEVEL_UNSPECIFIED, MINIMAL, LOW,
# MEDIUM, HIGH). gemini-3.1-flash-lite's own docs page
# (ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite) confirms
# thinking_level as its parameter but doesn't enumerate accepted values, so
# "minimal" as the floor is inferred from that family-wide pattern, not
# stated for this exact model — the safest reading either way, since it's
# the lowest level that exists anywhere in the family. Unlike
# gemini-2.5-flash-lite's thinking_budget=0, thinking cannot be fully
# disabled here, so every call spends an unpredictable number of hidden
# thought tokens before the JSON completion — the same mechanism that forced
# gemini-3.5-flash's BRIEFS_MAX_TOKENS up to 4000 during the take-1 incident.
# BRIEFS_MAX_TOKENS is raised back to 4000 below for the same reason.
BRIEFS_THINKING_CONFIG = {"thinking_level": "minimal"}
# Raised from 800 back to 4000 (see BRIEFS_THINKING_CONFIG above): with
# thinking floored at "minimal" rather than disabled, hidden thought tokens
# now share max_output_tokens with the visible JSON completion, so the
# headroom that protected gemini-3.5-flash from truncation is needed again
# here — 800 was only ever safe while thinking was fully off.
BRIEFS_MAX_TOKENS = 4000
BRIEFS_TEMPERATURE = 0.2
BRIEFS_MAX_COST_USD = 1.00  # hard stop; an actual full run is well under $0.01 on the paid tier
# gemini-3.1-flash-lite paid-tier pricing (ai.google.dev/gemini-api/docs/pricing,
# verified live via WebFetch, 2026-07-13): $0.25 input / $1.50 output per MTok
# for text (audio input is priced separately at $0.50 and doesn't apply here)
# — free tier is $0 either way, and this project has never gotten close to
# paid-tier spend regardless of which flash-lite model was configured.
BRIEFS_PRICE_PER_MTOK = {"input": 0.25, "output": 1.50}
# Kept at the take-2 value rather than re-derived: it was observed-quota
# pacing specific to gemini-2.5-flash-lite (10 RPM, +10% margin), and
# ai.google.dev/gemini-api/docs/rate-limits still doesn't publish a static
# per-model table, so gemini-3.1-flash-lite's real free-tier RPM is unknown
# until it's actually observed. Rather than guess a new number with no more
# evidence behind it than this one, the real backstop against a wrong pace
# is no longer just retry-and-skip but the clean-stop handlers accumulated
# across all three incidents (daily-quota stop, model-404 abort): a bad
# guess here now degrades to "slower than necessary," never "silently
# wrong." Set to 0 on a paid tier.
BRIEFS_MIN_REQUEST_INTERVAL_S = 6.6
