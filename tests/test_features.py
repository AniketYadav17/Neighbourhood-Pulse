"""Tests for the per-borough reporting-lag frames — the subtlest logic in the project."""

import pandas as pd
import pytest

from neighbourhood_pulse.config import LAG_TRIM_FRACTION, MAX_LAG_MONTHS
from neighbourhood_pulse.features import (
    assign_hex_borough,
    build_coffee_features,
    build_hex_features,
    build_planning_features,
    compute_borough_frames,
    load_planning,
)


def month_series(borough, counts, start="2021-01-01", hex_id="hexA"):
    """One record per application, `counts[i]` records in month i.

    `start` must be an exact month start: pd.date_range with freq="MS" rolls
    a mid-month start forward to the NEXT month, which would shift every
    anchor assertion by one month.
    """
    months = pd.date_range(start, periods=len(counts), freq="MS")
    rows = []
    for month, n in zip(months, counts, strict=True):
        for _ in range(n):
            rows.append(
                {
                    "lpa_name": borough,
                    "valid_date": month + pd.Timedelta(days=10),
                    "h3_index": hex_id,
                    "description": "extension",
                }
            )
    return pd.DataFrame(rows)


class TestComputeBoroughFrames:
    def test_contiguous_trailing_trim(self):
        # 10 healthy months (100) then 2 lag months (30, 10): trim exactly 2.
        df = month_series("B1", [100] * 10 + [30, 10])
        frames = compute_borough_frames(df)
        assert frames.loc["B1", "trim_months"] == 2
        assert not frames.loc["B1", "capped"]
        # Anchor = last month BEFORE the trimmed tail (month index 9).
        assert frames.loc["B1", "anchor"] == pd.Timestamp("2021-10-31")

    def test_interior_dip_is_kept(self):
        # Dip in the middle is real signal, not lag: nothing trimmed.
        df = month_series("B1", [100, 100, 10, 100, 100, 100])
        frames = compute_borough_frames(df)
        assert frames.loc["B1", "trim_months"] == 0

    def test_cap_fires_and_flags(self):
        # 6 trailing low months > MAX_LAG_MONTHS=4: trim capped at 4, flagged.
        df = month_series("B1", [100] * 8 + [10] * 6)
        frames = compute_borough_frames(df)
        assert frames.loc["B1", "trim_months"] == MAX_LAG_MONTHS
        assert frames.loc["B1", "capped"]

    def test_threshold_uses_median_and_fraction(self):
        # Median monthly = 100 -> threshold 75: a trailing 80 is NOT trimmed.
        df = month_series("B1", [100] * 9 + [80])
        frames = compute_borough_frames(df)
        assert frames.loc["B1", "trim_months"] == 0
        assert LAG_TRIM_FRACTION == 0.75  # constant pinned; test math depends on it

    def test_per_borough_independence(self):
        laggy = month_series("Laggy", [100] * 10 + [5, 5], hex_id="hexL")
        clean = month_series("Clean", [50] * 12, hex_id="hexC")
        frames = compute_borough_frames(pd.concat([laggy, clean], ignore_index=True))
        assert frames.loc["Laggy", "trim_months"] == 2
        assert frames.loc["Clean", "trim_months"] == 0
        # span_years measured from each borough's own first record to ITS anchor
        assert frames.loc["Clean", "span_years"] > frames.loc["Laggy", "span_years"]

    def test_degenerate_all_low_borough_does_not_crash(self):
        # Every month below its own... median==itself: threshold < counts -> no trim;
        # a truly pathological single-month borough must not index out of range.
        df = month_series("Tiny", [3])
        frames = compute_borough_frames(df)
        assert frames.loc["Tiny", "trim_months"] == 0


def test_assign_hex_borough_is_modal():
    df = pd.DataFrame(
        {
            "h3_index": ["h1"] * 3 + ["h2"] * 2,
            "lpa_name": ["A", "A", "B", "B", "B"],
            "valid_date": pd.Timestamp("2022-01-01"),
            "description": "x",
        }
    )
    hb = assign_hex_borough(df)
    assert hb["h1"] == "A"
    assert hb["h2"] == "B"


def test_load_planning_parses_dayfirst_and_drops_bad_rows(tmp_path):
    raw = pd.DataFrame(
        {
            "h3_index": ["h1", "h2", None],
            "lpa_name": ["A", "A", "A"],
            "description": ["x", "y", "z"],
            "valid_date": ["05/03/2023", "not a date", "01/01/2023"],
            "extra_column": [1, 2, 3],
        }
    )
    p = tmp_path / "planning.parquet"
    raw.to_parquet(p, index=False)
    out = load_planning(str(p), max_drop_fraction=1.0)
    # day-first: 05/03/2023 is 5 March, not 3 May
    assert out["valid_date"].iloc[0] == pd.Timestamp("2023-03-05")
    # unparseable date and null h3 rows dropped
    assert len(out) == 1
    assert list(out.columns) == ["h3_index", "lpa_name", "description", "valid_date"]


def test_load_planning_raises_on_mass_drop(tmp_path):
    raw = pd.DataFrame(
        {
            "h3_index": ["h1", "h2", None],
            "lpa_name": ["A", "A", "A"],
            "description": ["x", "y", "z"],
            "valid_date": ["05/03/2023", "not a date", "01/01/2023"],
        }
    )
    p = tmp_path / "planning.parquet"
    raw.to_parquet(p, index=False)
    with pytest.raises(ValueError, match="would be dropped"):
        load_planning(str(p))  # default 1% threshold; 2/3 bad rows


def two_borough_planning():
    """hexA wholly in A (healthy); hexB wholly in B (2 lag months); hexAB straddles,
    majority A. B's records in hexAB must be trimmed by A's frame (single-frame rule)."""
    a = month_series("A", [10] * 12, hex_id="hexA")
    b = month_series("B", [10] * 10 + [1, 1], hex_id="hexB")
    ab_a = month_series("A", [3] * 12, hex_id="hexAB")
    ab_b = month_series("B", [1] * 12, hex_id="hexAB")
    df = pd.concat([a, b, ab_a, ab_b], ignore_index=True)
    df.loc[df.index[:5], "description"] = "Change of Use from shop to cafe"
    return df


class TestBuildPlanningFeatures:
    def setup_method(self):
        self.df = two_borough_planning()
        self.frames = compute_borough_frames(self.df)
        self.hexb = assign_hex_borough(self.df)

    def test_single_frame_rule_governs_straddling_hexagon(self):
        assert self.hexb["hexAB"] == "A"  # modal borough
        feats = build_planning_features(self.df, self.frames, self.hexb)
        row = feats.set_index("h3_index").loc["hexAB"]
        # A's frame: no trim -> all 12 months of BOTH sub-borough records kept
        assert row["total_applications"] == 12 * 3 + 12 * 1
        assert row["borough"] == "A"
        assert row["span_years"] == pytest.approx(self.frames.loc["A", "span_years"])

    def test_laggy_borough_records_trimmed_by_own_frame(self):
        feats = build_planning_features(self.df, self.frames, self.hexb).set_index("h3_index")
        # B trimmed 2 months: hexB keeps 10 x 10 = 100 records
        assert feats.loc["hexB", "total_applications"] == 100

    def test_change_of_use_is_case_insensitive_and_ratio(self):
        feats = build_planning_features(self.df, self.frames, self.hexb).set_index("h3_index")
        assert feats.loc["hexA", "change_of_use_count"] == 5
        assert feats.loc["hexA", "change_of_use_ratio"] == pytest.approx(5 / 120)

    def test_velocity_definition(self):
        feats = build_planning_features(self.df, self.frames, self.hexb).set_index("h3_index")
        row = feats.loc["hexA"]
        expected = row["applications_recent"] / (row["total_applications"] / row["span_years"])
        assert row["planning_velocity"] == pytest.approx(expected)
        # steady borough, ~1-year window vs ~1-year span: velocity ~ 1, sane bounds
        assert 0.5 < row["planning_velocity"] < 1.5

    def test_nan_description_counts_as_not_change_of_use(self):
        df = self.df.copy()
        df.loc[df.index[-1], "description"] = None
        feats = build_planning_features(df, self.frames, self.hexb)
        assert feats["change_of_use_count"].sum() == 5  # NaN row didn't crash or count


def coffee_frame_features():
    return pd.DataFrame(
        {
            "h3_index": ["hexA", "hexA", "hexA", "hexZ", None],
            "brand": [None, "Costa", "Local Roasters", None, None],
        }
    )


class TestCoffeeAndMerge:
    def test_null_and_unknown_brands_are_independent(self):
        cf = build_coffee_features(coffee_frame_features())
        row = cf.set_index("h3_index").loc["hexA"]
        assert row["total_cafe_count"] == 3
        assert row["independent_cafe_count"] == 2  # Costa is a chain

    def test_null_h3_dropped(self):
        cf = build_coffee_features(coffee_frame_features())
        assert set(cf["h3_index"]) == {"hexA", "hexZ"}

    def test_merge_left_fills_zero_and_ratio(self):
        df = two_borough_planning()
        frames = compute_borough_frames(df)
        hexb = assign_hex_borough(df)
        pf = build_planning_features(df, frames, hexb)
        hf = build_hex_features(pf, build_coffee_features(coffee_frame_features()))
        idx = hf.set_index("h3_index")
        # planning hexagons are the unit: café-only hexZ must NOT appear
        assert "hexZ" not in idx.index
        assert idx.loc["hexB", "total_cafe_count"] == 0  # left-merge fillna(0)
        assert idx.loc["hexA", "cafe_to_application_ratio"] == pytest.approx(3 / 120)
        assert idx["total_cafe_count"].dtype.kind == "i"  # int after fillna
