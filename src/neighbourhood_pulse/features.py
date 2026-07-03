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
    CHAIN_BRANDS,
    LAG_TRIM_FRACTION,
    MAX_LAG_MONTHS,
    PLANNING_MAX_DROP_FRACTION,
    RECENT_WINDOW_MONTHS,
)

logger = logging.getLogger(__name__)

PLANNING_COLUMNS = ["h3_index", "lpa_name", "description", "valid_date"]


def load_planning(path: str, max_drop_fraction: float = PLANNING_MAX_DROP_FRACTION) -> pd.DataFrame:
    """Load processed planning records; parse day-first dates; drop unusable rows.

    Raises ValueError if more than max_drop_fraction of rows would drop — a
    mass drop means the input is malformed, not merely noisy.
    """
    df = pd.read_parquet(path, columns=PLANNING_COLUMNS)
    df["valid_date"] = pd.to_datetime(df["valid_date"], format="%d/%m/%Y", errors="coerce")
    bad = df["valid_date"].isna() | df["h3_index"].isna()
    n_bad = int(bad.sum())
    if n_bad:
        fraction = n_bad / len(df)
        if fraction > max_drop_fraction:
            raise ValueError(
                f"{n_bad} of {len(df)} planning rows ({fraction:.1%}) would be dropped "
                f"(unparseable date or null h3_index) — above the {max_drop_fraction:.0%} "
                "guard; the input file looks malformed."
            )
        logger.info("Dropping %s rows with unparseable dates or null h3_index.", n_bad)
    return df[~bad].reset_index(drop=True)


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
        # Defence-in-depth: provably unreachable — the maximum month can never be
        # below 0.75 x median, so the walk-back always stops before consuming the
        # whole series. Kept so a pathological series can't index out of range.
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


def build_planning_features(
    planning: pd.DataFrame, frames: pd.DataFrame, hex_borough: pd.Series
) -> pd.DataFrame:
    """Volume + temporal features per hexagon, trimmed by each hexagon's governing frame."""
    df = planning.copy()
    df["gov_borough"] = df["h3_index"].map(hex_borough)
    df["gov_anchor"] = df["gov_borough"].map(frames["anchor"])
    df["gov_cutoff"] = df["gov_anchor"] - pd.DateOffset(months=RECENT_WINDOW_MONTHS)

    n_before = len(df)
    df = df[df["valid_date"] <= df["gov_anchor"]].copy()
    logger.info(
        "Per-borough trim dropped %s of %s records (%.1f%%).",
        n_before - len(df),
        n_before,
        100 * (n_before - len(df)) / max(n_before, 1),
    )

    # na=False keeps the boolean clean for summing; with no NaNs in the mask,
    # size == count on it — `size` is used as the NaN-proof true row count.
    df["is_change_of_use"] = df["description"].str.contains("change of use", case=False, na=False)
    df["is_recent"] = df["valid_date"] > df["gov_cutoff"]

    hexf = (
        df.groupby("h3_index")
        .agg(
            total_applications=("is_change_of_use", "size"),
            change_of_use_count=("is_change_of_use", "sum"),
            applications_recent=("is_recent", "sum"),
        )
        .reset_index()
    )
    hexf["change_of_use_ratio"] = hexf["change_of_use_count"] / hexf["total_applications"]
    hexf["borough"] = hexf["h3_index"].map(hex_borough)
    hexf["span_years"] = hexf["borough"].map(frames["span_years"])
    # Recent 12-month rate vs the hexagon's own historical annual average (>1 = accelerating),
    # both measured within the same governing borough's frame.
    hexf["planning_velocity"] = hexf["applications_recent"] / (
        hexf["total_applications"] / hexf["span_years"]
    )
    return hexf


def build_coffee_features(coffee: pd.DataFrame) -> pd.DataFrame:
    """Café counts per hexagon; a null brand is independent (OSM rarely tags independents)."""
    df = coffee.dropna(subset=["h3_index"]).copy()
    df["is_independent"] = ~df["brand"].isin(CHAIN_BRANDS)
    return (
        df.groupby("h3_index")
        .agg(
            total_cafe_count=("is_independent", "size"),
            independent_cafe_count=("is_independent", "sum"),
        )
        .reset_index()
    )


def build_hex_features(
    planning_features: pd.DataFrame, coffee_features: pd.DataFrame
) -> pd.DataFrame:
    """Left-merge cafés onto the planning matrix — planning hexagons are the unit of analysis."""
    hexf = planning_features.merge(coffee_features, on="h3_index", how="left")
    for col in ["total_cafe_count", "independent_cafe_count"]:
        hexf[col] = hexf[col].fillna(0).astype(int)
    # Divide-safe: every planning hexagon has >= 1 application by construction.
    hexf["cafe_to_application_ratio"] = hexf["total_cafe_count"] / hexf["total_applications"]
    return hexf
