"""Briefs generation against a faked Anthropic client — no network, ever."""

import json
from types import SimpleNamespace

import pandas as pd

from neighbourhood_pulse.briefs import (
    build_user_prompt,
    estimate_cost_usd,
    generate_briefs,
    select_brief_hexagons,
)

GOOD = {
    "headline": "Signals outpace prices",
    "brief": "Planning activity is high.",
    "caveat": "Small sample.",
}


def gap_frame():
    rows = []
    for i, gap in enumerate([-0.5, -0.2, 0.3]):
        rows.append(
            {
                "h3_index": f"hex{i}",
                "borough": "Newham",
                "median_price": 400_000.0,
                "pred_price": 500_000.0,
                "valuation_gap": gap,
                "total_applications": 120.0,
                "applications_recent": 30.0,
                "change_of_use_count": 12.0,
                "change_of_use_ratio": 0.1,
                "planning_velocity": 1.2,
                "total_cafe_count": 4.0,
                "independent_cafe_count": 3.0,
                "cafe_to_application_ratio": 0.033,
                "dist_to_centre_km": 9.5,
                "span_years": 4.5,
                "sales_count": 40,
            }
        )
    return pd.DataFrame(rows)


def fake_response(payload):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=400, output_tokens=120),
        stop_reason="end_turn",
    )


class FakeClient:
    def __init__(self, payloads):
        self.calls = []
        outer = self

        class Messages:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                return fake_response(payloads[len(outer.calls) - 1])

        self.messages = Messages()


def test_select_orders_most_undervalued_first():
    hexes = select_brief_hexagons(gap_frame(), n=2)
    assert list(hexes["h3_index"]) == ["hex0", "hex1"]  # most negative gap first


def test_prompt_is_grounded_in_the_row():
    prompt = build_user_prompt(gap_frame().iloc[0])
    assert "Newham" in prompt
    assert "£400,000" in prompt
    assert "-50.0%" in prompt


def test_generate_skips_cached_and_saves_each(tmp_path):
    client = FakeClient([GOOD, GOOD])
    saves = []
    briefs = generate_briefs(
        gap_frame(),
        client,
        {"hex0": {**GOOD, "model": "cached"}},
        save=lambda b: saves.append(len(b)),
    )
    assert len(client.calls) == 2  # hex0 cached, hex1 + hex2 generated
    assert set(briefs) == {"hex0", "hex1", "hex2"}
    assert briefs["hex0"]["model"] == "cached"  # cache untouched
    assert saves == [2, 3]  # saved after every accepted brief


def test_invalid_json_and_extra_keys_are_skipped_not_fatal():
    client = FakeClient(["not json", {**GOOD, "surprise": "x"}, GOOD])
    briefs = generate_briefs(gap_frame(), client, {})
    assert set(briefs) == {"hex2"}  # only the schema-valid response survives


def test_cost_cap_stops_generation():
    client = FakeClient([GOOD, GOOD, GOOD])
    briefs = generate_briefs(gap_frame(), client, {}, max_cost_usd=1e-9)
    assert len(client.calls) == 1  # cap checked before each call after the first
    assert len(briefs) == 1


def test_request_shape_pins_grounding_controls():
    client = FakeClient([GOOD, GOOD, GOOD])
    generate_briefs(gap_frame(), client, {})
    kwargs = client.calls[0]
    assert kwargs["model"] == "claude-haiku-4-5"
    assert kwargs["temperature"] == 0.2
    assert "ONLY the signals" in kwargs["system"]
    assert kwargs["output_config"]["format"]["type"] == "json_schema"


def test_estimate_cost_usd():
    assert estimate_cost_usd(1_000_000, 0) == 1.00
    assert estimate_cost_usd(0, 1_000_000) == 5.00
