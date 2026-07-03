"""Shared synthetic training-data builder for model/API/app tests."""

import h3
import numpy as np
import pandas as pd

from neighbourhood_pulse.model import add_centrality


def synthetic_training(n=80):
    """Price is a noisy function of the signals so the model has something to learn.

    A FRESH seeded generator per call makes repeated calls return identical
    frames (the determinism test depends on this). Random lat/lon pairs can
    land in the same res-8 cell, so duplicates are dropped LAST — after every
    RNG draw of length n — keeping all draws aligned; expect len ~= n, not == n.
    """
    rng = np.random.default_rng(0)
    lat = rng.uniform(51.35, 51.65, n)
    lon = rng.uniform(-0.4, 0.2, n)
    hexes = [h3.latlng_to_cell(la, lo, 8) for la, lo in zip(lat, lon, strict=True)]
    df = pd.DataFrame(
        {
            "h3_index": hexes,
            "borough": ["Synth"] * n,
            "total_applications": rng.integers(30, 400, n),
            "applications_recent": rng.integers(5, 80, n),
            "total_cafe_count": rng.integers(0, 12, n),
            "independent_cafe_count": rng.integers(0, 10, n),
            "span_years": 4.5,
            "sales_count": rng.integers(30, 150, n),
        }
    )
    df["change_of_use_count"] = (df["total_applications"] * rng.uniform(0, 0.1, n)).astype(int)
    df["change_of_use_ratio"] = df["change_of_use_count"] / df["total_applications"]
    df["planning_velocity"] = df["applications_recent"] / (
        df["total_applications"] / df["span_years"]
    )
    df["cafe_to_application_ratio"] = df["total_cafe_count"] / df["total_applications"]
    df = add_centrality(df)
    df["median_price"] = (
        250_000
        + 12_000 * df["total_cafe_count"]
        - 8_000 * df["dist_to_centre_km"]
        + rng.normal(0, 20_000, n)
    ).clip(lower=80_000)
    return df.drop_duplicates(subset="h3_index").reset_index(drop=True)
