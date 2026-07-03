# tests/test_package.py
"""Package installs and its modules import cleanly."""


def test_package_imports():
    import neighbourhood_pulse
    from neighbourhood_pulse import cli, config, ingestion, transformation  # noqa: F401

    assert neighbourhood_pulse.__version__


def test_config_has_all_33_boroughs():
    from neighbourhood_pulse.config import TARGET_BOROUGHS

    assert len(TARGET_BOROUGHS) == 33
    assert len(set(TARGET_BOROUGHS)) == 33
    # Exact API spellings that silently return zero records if wrong.
    assert "Barking & Dagenham" in TARGET_BOROUGHS  # ampersand, not "and"
    assert "Kingston" in TARGET_BOROUGHS  # no "upon Thames"
    assert "Westminster" in TARGET_BOROUGHS  # no "City of"
    assert "LLDC" not in TARGET_BOROUGHS  # development corps excluded


def test_feature_cols_and_paths_consistent():
    from neighbourhood_pulse import config

    assert len(config.FEATURE_COLS) == 9
    assert config.FEATURE_COLS[-1] == "dist_to_centre_km"
    assert config.VALUATION_GAP_PATH.startswith(config.ARTIFACTS_DIR)
    assert set(config.NAME_FIX) == {"KINGSTON", "RICHMOND", "WESTMINSTER"}
