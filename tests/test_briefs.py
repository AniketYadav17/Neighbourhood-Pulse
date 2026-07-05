"""Briefs generation against a faked google-genai client — no network, ever."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from neighbourhood_pulse import briefs as briefs_module
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


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Every test in this module runs through generate_briefs's pacing/retry
    sleeps. Recording them (instead of letting them elapse) keeps the suite
    instant while still letting individual tests assert on what was slept."""
    calls = []
    monkeypatch.setattr(briefs_module.time, "sleep", lambda s: calls.append(s))
    return calls


class FakeQuotaError(Exception):
    """Duck-types google.genai.errors.ClientError's 429 shape (a `.code` attribute)."""

    code = 429


class FakeOtherError(Exception):
    """A non-quota API error — should be skipped, not retried."""

    code = 500


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


def fake_response(payload, finish_reason="STOP"):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(
        text=text,
        candidates=[SimpleNamespace(finish_reason=finish_reason)],
        usage_metadata=SimpleNamespace(prompt_token_count=400, candidates_token_count=120),
    )


class FakeModels:
    def __init__(self, outer, payloads):
        self.outer = outer
        self.payloads = payloads

    def generate_content(self, **kwargs):
        self.outer.calls.append(kwargs)
        return fake_response(self.payloads[len(self.outer.calls) - 1])


class FakeClient:
    def __init__(self, payloads):
        self.calls = []
        self.models = FakeModels(self, payloads)


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
    assert kwargs["model"] == "gemini-2.5-flash-lite"
    assert kwargs["config"]["temperature"] == 0.2
    assert "ONLY the signals" in kwargs["config"]["system_instruction"]
    assert kwargs["config"]["response_mime_type"] == "application/json"
    schema = kwargs["config"]["response_schema"]
    # Gemini's Schema proto rejects "additionalProperties" outright (400
    # INVALID_ARGUMENT); it must never be sent, even though it's part of the
    # canonical BRIEF_SCHEMA used for local validation.
    assert "additionalProperties" not in schema
    assert schema["type"] == "object"
    assert set(schema["required"]) == {"headline", "brief", "caveat"}


def test_request_shape_disables_thinking_and_budgets_the_short_completion():
    # gemini-2.5-flash-lite is thinking_budget-family: 0 fully disables
    # thinking (unlike gemini-3.x's thinking_level, whose floor is "minimal"
    # and can never reach true zero) — so no hidden thought tokens can eat
    # into max_output_tokens the way they did on gemini-3.5-flash (which
    # needed 4000 for that reason). 800 is ample headroom for the short
    # headline/brief/caveat JSON completion alone.
    client = FakeClient([GOOD])
    generate_briefs(gap_frame(), client, {}, request_interval_s=0)
    config = client.calls[0]["config"]
    assert config["max_output_tokens"] == 800
    assert config["thinking_config"] == {"thinking_budget": 0}


def test_estimate_cost_usd():
    assert estimate_cost_usd(1_000_000, 0) == 0.10
    assert estimate_cost_usd(0, 1_000_000) == 0.40


def test_quota_error_retries_then_succeeds(no_sleep):
    """A 429 on the first attempt for a hexagon is retried, not fatal."""

    class QuotaThenSuccessModels:
        def __init__(self, outer):
            self.outer = outer
            self.attempts = 0

        def generate_content(self, **kwargs):
            self.outer.calls.append(kwargs)
            self.attempts += 1
            if self.attempts == 1:
                raise FakeQuotaError("rate limited")
            return fake_response(GOOD)

    class QuotaThenSuccessClient:
        def __init__(self):
            self.calls = []
            self.models = QuotaThenSuccessModels(self)

    client = QuotaThenSuccessClient()
    briefs = generate_briefs(gap_frame().iloc[[0]], client, {}, request_interval_s=0)
    assert set(briefs) == {"hex0"}  # the hexagon was recovered, not skipped
    assert len(client.calls) == 2  # one failed attempt, one successful retry
    assert 60.0 in no_sleep  # the quota-retry sleep happened (mocked, not real)


def test_persistent_quota_error_skips_hexagon_and_continues(no_sleep):
    """A hexagon whose quota errors never clear is skipped after
    QUOTA_MAX_RETRIES, but the run continues to the next hexagon."""

    class AlwaysQuotaErrorModels:
        def __init__(self, outer, fail_calls):
            self.outer = outer
            self.fail_calls = fail_calls

        def generate_content(self, **kwargs):
            self.outer.calls.append(kwargs)
            if len(self.outer.calls) <= self.fail_calls:
                raise FakeQuotaError("rate limited")
            return fake_response(GOOD)

    class AlwaysQuotaErrorClient:
        def __init__(self, fail_calls):
            self.calls = []
            self.models = AlwaysQuotaErrorModels(self, fail_calls)

    # hex0's every attempt (1 initial + 3 retries = 4 calls) fails; hex1, hex2
    # then succeed normally, proving the loop moved on.
    client = AlwaysQuotaErrorClient(fail_calls=4)
    briefs = generate_briefs(gap_frame(), client, {}, request_interval_s=0)
    assert set(briefs) == {"hex1", "hex2"}  # hex0 skipped, run continued
    assert len(client.calls) == 6  # 4 for hex0 + 1 each for hex1, hex2
    assert no_sleep.count(60.0) == 3  # three quota-retry sleeps, then gave up


def test_daily_quota_error_stops_the_whole_run_cleanly(no_sleep):
    """A 429 whose payload's quotaId contains 'PerDay' means the free-tier
    daily budget is exhausted for the whole project — sleeping and retrying
    is pointless (it won't clear until midnight Pacific). The loop must stop
    immediately, with no retry sleep, keeping whatever briefs were already
    generated (this is the real failure the fix addresses: quotaId
    'GenerateRequestsPerDayPerProjectPerModel-FreeTier', quotaValue 20)."""

    class FakeDailyQuotaError(Exception):
        code = 429
        # Duck-types google.genai.errors.APIError's `.details` (parsed
        # response JSON), shaped like a real QuotaFailure payload.
        details = {
            "error": {
                "code": 429,
                "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [
                            {
                                "quotaId": ("GenerateRequestsPerDayPerProjectPerModel-FreeTier"),
                                "quotaValue": "20",
                            }
                        ],
                    }
                ],
            }
        }

    class DailyQuotaModels:
        def __init__(self, outer):
            self.outer = outer

        def generate_content(self, **kwargs):
            self.outer.calls.append(kwargs)
            if len(self.outer.calls) == 1:
                return fake_response(GOOD)  # hex0 succeeds
            raise FakeDailyQuotaError("RESOURCE_EXHAUSTED")  # hex1 hits the daily cap

    class DailyQuotaClient:
        def __init__(self):
            self.calls = []
            self.models = DailyQuotaModels(self)

    client = DailyQuotaClient()
    briefs = generate_briefs(gap_frame(), client, {}, request_interval_s=0)
    assert set(briefs) == {"hex0"}  # the earlier success is retained
    assert len(client.calls) == 2  # hex0 succeeded; hex1's daily 429 stopped the run
    assert 60.0 not in no_sleep  # no retry sleep for a daily-quota error


def test_daily_quota_id_with_unrecognized_shape_falls_back_to_retry(no_sleep):
    """If a 429's payload doesn't carry a recognizable quotaId (e.g. no
    `.details` at all, like FakeQuotaError), the existing per-minute
    retry-then-skip behavior must still apply — 'can't tell' is not 'not a
    daily quota'."""

    class AlwaysQuotaErrorModels:
        def __init__(self, outer):
            self.outer = outer

        def generate_content(self, **kwargs):
            self.outer.calls.append(kwargs)
            raise FakeQuotaError("rate limited")

    class AlwaysQuotaErrorClient:
        def __init__(self):
            self.calls = []
            self.models = AlwaysQuotaErrorModels(self)

    client = AlwaysQuotaErrorClient()
    briefs = generate_briefs(gap_frame().iloc[[0]], client, {}, request_interval_s=0)
    assert briefs == {}  # hex0 skipped after exhausting retries, not a clean stop
    assert len(client.calls) == 4  # 1 initial + 3 retries (QUOTA_MAX_RETRIES)
    assert no_sleep.count(60.0) == 3  # ordinary per-minute retry sleeps happened


def test_non_quota_api_error_skips_hexagon_immediately(no_sleep):
    """A non-429 API error is not retried — it skips straight away."""

    class OneShotErrorModels:
        def __init__(self, outer):
            self.outer = outer

        def generate_content(self, **kwargs):
            self.outer.calls.append(kwargs)
            if len(self.outer.calls) == 1:
                raise FakeOtherError("server error")
            return fake_response(GOOD)

    class OneShotErrorClient:
        def __init__(self):
            self.calls = []
            self.models = OneShotErrorModels(self)

    client = OneShotErrorClient()
    briefs = generate_briefs(gap_frame(), client, {}, request_interval_s=0)
    assert set(briefs) == {"hex1", "hex2"}  # hex0 skipped, no retry attempted
    assert len(client.calls) == 3  # one failed attempt for hex0 (not retried) + 2
    assert 60.0 not in no_sleep  # no quota-retry sleep for a non-quota error


def test_thoughts_tokens_counted_in_spend():
    """Paid-tier billing counts thinking as output; the cost cap must too."""
    response_with_thoughts = SimpleNamespace(
        text=json.dumps(GOOD),
        candidates=[SimpleNamespace(finish_reason="STOP")],
        usage_metadata=SimpleNamespace(
            prompt_token_count=400, candidates_token_count=120, thoughts_token_count=1000
        ),
    )

    class ThoughtsModels:
        def __init__(self, outer):
            self.outer = outer

        def generate_content(self, **kwargs):
            self.outer.calls.append(kwargs)
            return response_with_thoughts

    class ThoughtsClient:
        def __init__(self):
            self.calls = []
            self.models = ThoughtsModels(self)

    client = ThoughtsClient()
    briefs = generate_briefs(
        gap_frame().iloc[[0]], client, {}, request_interval_s=0, max_cost_usd=1e9
    )
    assert set(briefs) == {"hex0"}
    expected = estimate_cost_usd(400, 120 + 1000)
    # No direct spend getter, so re-derive: with a cap just under the expected
    # spend, a second hexagon must not be attempted.
    client2 = ThoughtsClient()
    briefs2 = generate_briefs(
        gap_frame(), client2, {}, request_interval_s=0, max_cost_usd=expected - 1e-9
    )
    assert len(client2.calls) == 1  # thoughts tokens pushed spend over the cap already
    assert set(briefs2) == {"hex0"}


def test_pacing_sleeps_between_consecutive_calls(no_sleep):
    """Consecutive calls are spaced at least request_interval_s apart."""
    client = FakeClient([GOOD, GOOD, GOOD])
    generate_briefs(gap_frame(), client, {}, request_interval_s=13.0)
    assert len(client.calls) == 3
    # First call has nothing to wait on; the next two must each wait roughly
    # the full interval since the fake calls return instantly.
    waits = [s for s in no_sleep if s != 60.0]
    assert len(waits) == 2
    assert all(9.0 < w <= 13.0 for w in waits)
