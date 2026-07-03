"""Tests for the Land Registry target build (synthetic 16-column CSVs, no real data)."""

import pandas as pd
import pytest

from neighbourhood_pulse.target import (
    build_postcode_lookup,
    build_price_target,
    load_sales,
    london_districts,
    normalise_postcodes,
)


def test_normalise_postcodes():
    s = pd.Series(["e15 1aa", " SW1A 2AA ", "e151aa", None])
    out = normalise_postcodes(s)
    assert out.iloc[0] == "E151AA"
    assert out.iloc[1] == "SW1A2AA"
    assert out.iloc[2] == "E151AA"
    assert pd.isna(out.iloc[3])


def test_london_districts_applies_name_fix():
    d = london_districts()
    assert "KINGSTON UPON THAMES" in d
    assert "KINGSTON" not in d
    assert "CITY OF WESTMINSTER" in d
    assert "BARKING AND DAGENHAM" in d  # & -> AND
    assert len(d) == 33


def lr_csv_row(price, date, postcode, ptype, district, category):
    """A full 16-column Price Paid row; only positions 1,2,3,4,12,15 matter."""
    cells = ["{GUID}"] * 16
    cells[1] = str(price)
    cells[2] = date
    cells[3] = postcode
    cells[4] = ptype
    cells[12] = district
    cells[15] = category
    return ",".join(f'"{c}"' for c in cells)


@pytest.fixture
def lr_raw_dir(tmp_path):
    rows_2021 = [
        lr_csv_row(500000, "2021-05-01 00:00", "E15 1AA", "F", "NEWHAM", "A"),
        lr_csv_row(300000, "2021-06-01 00:00", "E15 1AA", "T", "NEWHAM", "A"),
        lr_csv_row(999999, "2021-06-01 00:00", "E15 1AA", "O", "NEWHAM", "A"),  # ptype O: drop
        lr_csv_row(888888, "2021-07-01 00:00", "E15 1AA", "F", "NEWHAM", "B"),  # cat B: drop
        lr_csv_row(777777, "2021-07-01 00:00", "M1 1AA", "F", "MANCHESTER", "A"),  # not London
        lr_csv_row(400000, "2021-08-01 00:00", "KT1 1AA", "D", "KINGSTON UPON THAMES", "A"),
    ]
    rows_2022 = [
        lr_csv_row(600000, "2022-03-01 00:00", "E15 1AA", "S", "NEWHAM", "A"),
    ]
    (tmp_path / "pp-2021.csv").write_text("\n".join(rows_2021) + "\n", encoding="utf-8")
    (tmp_path / "pp-2022.csv").write_text("\n".join(rows_2022) + "\n", encoding="utf-8")
    return tmp_path


def test_load_sales_filters_and_pools(lr_raw_dir):
    sales = load_sales(str(lr_raw_dir), years=(2021, 2022))
    assert len(sales) == 4  # O/B/Manchester dropped; 3 x 2021 + 1 x 2022 kept
    assert set(sales["year"]) == {2021, 2022}
    assert sales["price"].dtype.kind in "if"
    assert set(sales["district"]) == {"NEWHAM", "KINGSTON UPON THAMES"}


def test_build_postcode_lookup_is_modal(tmp_path):
    # Unambiguous majority: E151AA has 3 votes for hex1, 2 for hex2 (never a tie —
    # value_counts tie-breaking is version-dependent and must not be asserted on).
    planning = pd.DataFrame(
        {
            "postcode": ["E15 1AA", "E15 1AA", "E15 1AA", "E15 1AA", "KT1 1AA"],
            "h3_index": ["hex1", "hex1", "hex1", "hex2", "hex9"],
        }
    )
    coffee = pd.DataFrame(
        {
            "addr:postcode": ["e151aa"],
            "h3_index": ["hex2"],
        }
    )
    pp = tmp_path / "planning.parquet"
    cp = tmp_path / "coffee.parquet"
    planning.to_parquet(pp, index=False)
    coffee.to_parquet(cp, index=False)
    lookup = build_postcode_lookup(str(pp), str(cp))
    assert lookup["E151AA"] == "hex1"  # 3 vs 2, case/space-normalised across both sources
    assert lookup["KT11AA"] == "hex9"


def test_build_price_target_median_and_count(lr_raw_dir):
    sales = load_sales(str(lr_raw_dir), years=(2021, 2022))
    lookup = pd.Series({"E151AA": "hexE", "KT11AA": "hexK"})
    target = build_price_target(sales, lookup).set_index("h3_index")
    assert target.loc["hexE", "sales_count"] == 3
    assert target.loc["hexE", "median_price"] == 500000  # median of 500k/300k/600k
    assert target.loc["hexK", "sales_count"] == 1
