"""Hexagon feature engineering.

Transcribed from the validated research notebook (sections 7-10, per-borough
trim + single-frame rule — the canonical final version). The reporting-lag
trim: planning records appear with borough-specific ingestion lag, so each
borough's trailing months whose count falls below LAG_TRIM_FRACTION x its
median monthly count are trimmed (contiguous tail only — an interior dip is
real signal). MAX_LAG_MONTHS caps the trim; a longer run is implausible as
lag and is flagged as likely real decline instead.
"""

import logging

import pandas as pd

from neighbourhood_pulse.config import (
    LAG_TRIM_FRACTION,
    MAX_LAG_MONTHS,
)

logger = logging.getLogger(__name__)

PLANNING_COLUMNS = ["h3_index", "lpa_name", "description", "valid_date"]


def load_planning(path: str) -> pd.DataFrame:
    """Load processed planning records; parse day-first dates; drop unusable rows."""
    df = pd.read_parquet(path, columns=PLANNING_COLUMNS)
    df["valid_date"] = pd.to_datetime(df["valid_date"], format="%d/%m/%Y", errors="coerce")
    n_bad = int(df["valid_date"].isna().sum() + df["h3_index"].isna().sum())
    if n_bad:
        logger.info("Dropping %s rows with unparseable dates or null h3_index.", n_bad)
    return df.dropna(subset=["valid_date", "h3_index"]).reset_index(drop=True)


def compute_borough_frames(planning: pd.DataFrame) -> pd.DataFrame:
    """Per-borough anchor/span from each borough's OWN monthly series.

    Returns a frame indexed by borough: anchor (last fully-reported month end),
    span_years (first record -> anchor), trim_months, capped.
    """
    frames = {}
    for borough, sub in planning.groupby("lpa_name"):
        monthly = sub.set_index("valid_date").resample("ME").size()
        threshold = LAG_TRIM_FRACTION * monthly.median()
        below = 0
        for count in reversed(monthly.values):
            if count < threshold:
                below += 1
            else:
                break
        capped = below > MAX_LAG_MONTHS
        if capped:
            logger.warning(
                "%s: %s trailing months below threshold (> MAX_LAG_MONTHS=%s) — "
                "implausible as pure lag, likely real decline; capping trim.",
                borough,
                below,
                MAX_LAG_MONTHS,
            )
        # Guard len-1: a degenerate single-month series must keep its only month.
        # (Never fires on real data; notebook had no such borough.)
        trim_n = min(below, MAX_LAG_MONTHS, len(monthly) - 1)
        anchor = monthly.index[-(trim_n + 1)]
        span_years = (anchor - sub["valid_date"].min()).days / 365.25
        frames[borough] = {
            "anchor": anchor,
            "span_years": span_years,
            "trim_months": trim_n,
            "capped": capped,
        }
    return pd.DataFrame.from_dict(frames, orient="index").rename_axis("borough")


def assign_hex_borough(planning: pd.DataFrame) -> pd.Series:
    """Each hexagon's dominant (modal) borough — the single-frame rule.

    One borough's frame governs everything for a hexagon (anchor, span,
    recent-window, trim). The alternative (dominant-span + record-level trim)
    would divide one borough's records by another's span — re-introducing the
    fake-acceleration bias the trim removes.
    """
    return planning.groupby("h3_index")["lpa_name"].agg(lambda s: s.value_counts().idxmax())
