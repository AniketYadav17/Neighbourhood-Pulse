"""Land Registry Price Paid data -> per-hexagon pooled price target.

Transcribed from notebook sections 11-12. Sales are pooled across 2021-2025
(median per hexagon) because per-year medians are too thin at res-8; the
MIN_SALES_PER_HEX floor is applied downstream when building the training table.
"""

import logging

import pandas as pd

from neighbourhood_pulse.config import (
    NAME_FIX,
    RESIDENTIAL_PTYPES,
    TARGET_BOROUGHS,
)

logger = logging.getLogger(__name__)

# Positional columns of the headerless national Price Paid CSVs.
LR_COLUMNS = {1: "price", 2: "date", 3: "postcode", 4: "ptype", 12: "district", 15: "category"}


def normalise_postcodes(s: pd.Series) -> pd.Series:
    """Uppercase, strip ALL whitespace: 'e15 1aa' -> 'E151AA' (join key form)."""
    return s.astype("string").str.upper().str.replace(r"\s+", "", regex=True)


def london_districts() -> set[str]:
    """Config borough names -> LR district spellings (upper, & -> AND, NAME_FIX)."""
    upper = (b.upper().replace("&", "AND") for b in TARGET_BOROUGHS)
    return {NAME_FIX.get(b, b) for b in upper}


def load_sales(raw_dir: str, years) -> pd.DataFrame:
    """Load + filter the national CSVs to London residential category-A sales, pooled."""
    districts = london_districts()
    frames = []
    for year in years:
        path = f"{raw_dir}/pp-{year}.csv"
        df = pd.read_csv(
            path, header=None, usecols=list(LR_COLUMNS), names=list(LR_COLUMNS.values())
        )
        df = df[
            df["district"].isin(districts)
            & df["ptype"].isin(RESIDENTIAL_PTYPES)
            & (df["category"] == "A")
        ]
        logger.info("pp-%s: %s London residential cat-A sales.", year, len(df))
        frames.append(df)
    sales = pd.concat(frames, ignore_index=True)
    sales["year"] = pd.to_datetime(sales["date"]).dt.year
    return sales


def build_postcode_lookup(planning_path: str, coffee_path: str) -> pd.Series:
    """postcode -> modal h3_index, built from the two geocoded datasets in hand."""
    pl = pd.read_parquet(planning_path, columns=["postcode", "h3_index"]).dropna()
    pl["pc"] = normalise_postcodes(pl["postcode"])
    cof = pd.read_parquet(coffee_path, columns=["addr:postcode", "h3_index"]).dropna()
    cof["pc"] = normalise_postcodes(cof["addr:postcode"])
    both = pd.concat([pl[["pc", "h3_index"]], cof[["pc", "h3_index"]]], ignore_index=True)
    lookup = both.groupby("pc")["h3_index"].agg(lambda s: s.value_counts().idxmax())
    logger.info("Postcode -> hexagon lookup: %s postcodes.", len(lookup))
    return lookup


def build_price_target(sales: pd.DataFrame, lookup: pd.Series) -> pd.DataFrame:
    """Median pooled price + sales count per hexagon (unmapped sales dropped)."""
    df = sales.dropna(subset=["postcode"]).copy()
    df["pc"] = normalise_postcodes(df["postcode"])
    df["h3_index"] = df["pc"].map(lookup)
    mapped = df.dropna(subset=["h3_index"])
    logger.info(
        "Mapped %s of %s sales to a hexagon (%.1f%%).",
        len(mapped),
        len(df),
        100 * len(mapped) / max(len(df), 1),
    )
    return (
        mapped.groupby("h3_index")["price"]
        .agg(median_price="median", sales_count="size")
        .reset_index()
    )
