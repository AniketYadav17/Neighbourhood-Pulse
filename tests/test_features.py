"""Tests for the per-borough reporting-lag frames — the subtlest logic in the project."""

import pandas as pd

from neighbourhood_pulse.config import LAG_TRIM_FRACTION, MAX_LAG_MONTHS
from neighbourhood_pulse.features import (
    assign_hex_borough,
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
    out = load_planning(str(p))
    # day-first: 05/03/2023 is 5 March, not 3 May
    assert out["valid_date"].iloc[0] == pd.Timestamp("2023-03-05")
    # unparseable date and null h3 rows dropped
    assert len(out) == 1
    assert list(out.columns) == ["h3_index", "lpa_name", "description", "valid_date"]
