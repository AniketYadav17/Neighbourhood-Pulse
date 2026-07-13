"""App helper tests (pure functions + repricing routing). AppTest smoke at the end."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import shared  # noqa: E402


def test_gap_colour_endpoints_and_clamp():
    assert shared.gap_colour(-0.3, 0.3) == [215, 48, 39]  # fully undervalued -> red
    assert shared.gap_colour(0.0, 0.3) == [255, 255, 191]  # neutral -> cream
    assert shared.gap_colour(0.3, 0.3) == [26, 152, 80]  # fully overvalued -> green
    assert shared.gap_colour(-9.9, 0.3) == [215, 48, 39]  # clamped


def what_if_row():
    return pd.Series(
        {
            "total_applications": 120.0,
            "change_of_use_count": 12.0,
            "applications_recent": 30.0,
            "change_of_use_ratio": 0.1,
            "planning_velocity": 30.0 / (120.0 / 4.5),
            "total_cafe_count": 4.0,
            "independent_cafe_count": 3.0,
            "cafe_to_application_ratio": 4.0 / 120.0,
            "dist_to_centre_km": 9.5,
            "span_years": 4.5,
        }
    )


def test_derive_what_if_recomputes_ratios():
    features = shared.derive_what_if_features(what_if_row(), {"total_cafe_count": 8.0})
    assert features["total_cafe_count"] == 8.0
    assert features["cafe_to_application_ratio"] == pytest.approx(8.0 / 120.0)
    assert features["dist_to_centre_km"] == 9.5  # centrality is not editable


def test_derive_what_if_velocity_uses_span():
    features = shared.derive_what_if_features(what_if_row(), {"applications_recent": 60.0})
    assert features["planning_velocity"] == pytest.approx(60.0 / (120.0 / 4.5))


def test_reprice_in_process(monkeypatch):
    monkeypatch.delenv("PULSE_API_URL", raising=False)
    price = shared.reprice(shared.derive_what_if_features(what_if_row(), {}))
    assert price > 0  # committed model.joblib answers


def test_reprice_uses_api_when_env_set(monkeypatch):
    calls = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"predicted_price": 123456.0}

    def fake_post(url, json, timeout):
        calls["url"] = url
        return FakeResponse()

    monkeypatch.setenv("PULSE_API_URL", "http://api:8000")
    monkeypatch.setattr("requests.post", fake_post)
    assert shared.reprice(shared.derive_what_if_features(what_if_row(), {})) == 123456.0
    assert calls["url"] == "http://api:8000/predict"


def test_app_smoke_explore():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app/app.py", default_timeout=60)
    at.run()
    assert not at.exception


def test_app_smoke_model_and_methodology_pages():
    from streamlit.testing.v1 import AppTest

    for page in ("views/model.py", "views/methodology.py"):
        at = AppTest.from_file("app/app.py", default_timeout=60)
        at.switch_page(page)
        at.run()
        assert not at.exception, page


def test_fmt_gbp_compact():
    assert shared.fmt_gbp(675_000) == "£675k"
    assert shared.fmt_gbp(1_239_709) == "£1.24M"
    assert shared.fmt_gbp(999_499) == "£999k"
    assert shared.fmt_gbp(999_500) == "£1.00M"  # never renders as £1000k
    assert shared.fmt_gbp(850) == "£850"
    assert shared.fmt_gbp(1_000) == "£1k"
