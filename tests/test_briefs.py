"""Briefs generation against a faked google-genai client — no network, ever."""

import json
import logging
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

    def get(self, **kwargs):
        """No-op stand-in for the SDK's model-info preflight call
        (_model_available). None of the generate_briefs-focused tests below
        exercise the preflight — they only need this to exist and not raise."""
        return None


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
    assert kwargs["model"] == "gemini-3.1-flash-lite"
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
    # gemini-3.1-flash-lite is thinking_level-family: "minimal" is the lowest
    # level it accepts, not a true off-switch (unlike gemini-2.5-flash-lite's
    # thinking_budget=0, which fully disabled thinking) — so hidden thought
    # tokens can still eat into max_output_tokens the way they did on
    # gemini-3.5-flash, which is why this needs the same 4000-token headroom
    # rather than the 800 that was only safe while thinking was fully off.
    client = FakeClient([GOOD])
    generate_briefs(gap_frame(), client, {}, request_interval_s=0)
    config = client.calls[0]["config"]
    assert config["max_output_tokens"] == 4000
    assert config["thinking_config"] == {"thinking_level": "minimal"}


def test_estimate_cost_usd():
    assert estimate_cost_usd(1_000_000, 0) == 0.25
    assert estimate_cost_usd(0, 1_000_000) == 1.50


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


def test_model_not_found_error_aborts_the_whole_run(no_sleep):
    """A 404 on generate_content means the configured model itself doesn't
    exist (deprecated / not available to new users / typo'd) — every
    remaining hexagon would fail identically, so this is a fatal
    configuration error, not a per-hexagon fluke. The loop must abort
    immediately, keeping whatever briefs were already generated — the same
    clean-stop shape as the daily-quota case (this is the real failure the
    fix addresses: gemini-2.5-flash-lite 404'd with `NOT_FOUND: "This model
    models/gemini-2.5-flash-lite is no longer available to new users."`)."""

    class FakeNotFoundError(Exception):
        code = 404
        message = (
            "This model models/bogus-model is no longer available to new "
            "users. Please update your code to use a newer model."
        )

    class NotFoundModels:
        def __init__(self, outer):
            self.outer = outer

        def generate_content(self, **kwargs):
            self.outer.calls.append(kwargs)
            if len(self.outer.calls) == 1:
                return fake_response(GOOD)  # hex0 succeeds
            raise FakeNotFoundError("NOT_FOUND")  # hex1's model 404s

    class NotFoundClient:
        def __init__(self):
            self.calls = []
            self.models = NotFoundModels(self)

    client = NotFoundClient()
    briefs = generate_briefs(gap_frame(), client, {}, request_interval_s=0)
    assert set(briefs) == {"hex0"}  # the earlier success is retained
    assert len(client.calls) == 2  # hex0 succeeded; hex1's 404 stopped the run
    assert 60.0 not in no_sleep  # a model 404 is not a quota error; no retry sleep


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


def test_model_override_reaches_request_kwargs_and_is_recorded():
    """`generate_briefs`'s `model` param (threaded from `pulse briefs
    --model`) must reach the actual generate_content call and be recorded on
    the accepted brief — this is what decouples the repo from Google
    deprecating BRIEFS_MODEL out from under it without a code edit."""
    client = FakeClient([GOOD])
    briefs = generate_briefs(
        gap_frame().iloc[[0]], client, {}, request_interval_s=0, model="gemini-9.9-flash-lite"
    )
    assert client.calls[0]["model"] == "gemini-9.9-flash-lite"
    assert briefs["hex0"]["model"] == "gemini-9.9-flash-lite"


def test_preflight_model_not_found_lists_available_and_returns_without_looping(caplog):
    """_model_available's 404 path must log a clear diagnostic (including a
    best-effort list of currently-available flash-family models, filtered
    down from a full model listing) and signal the caller not to loop at
    all — a 404 here means every subsequent generate_content call would fail
    identically, so there's no point spending even one of them."""

    class FakeNotFoundError(Exception):
        code = 404
        message = "This model models/bogus-model is no longer available to new users."

    class PreflightModels:
        def get(self, **kwargs):
            raise FakeNotFoundError("NOT_FOUND")

        def list(self):
            return [
                SimpleNamespace(name="models/gemini-9.9-flash-lite"),
                SimpleNamespace(name="models/gemini-9.9-flash"),
                SimpleNamespace(name="models/gemini-9.9-pro"),  # no "flash" — filtered out
            ]

    class PreflightClient:
        def __init__(self):
            self.models = PreflightModels()

    with caplog.at_level(logging.ERROR):
        available = briefs_module._model_available(PreflightClient(), "bogus-model")

    assert available is False  # signal to the caller: don't loop
    messages = " ".join(r.message for r in caplog.records)
    assert "bogus-model" in messages
    assert "gemini-9.9-flash-lite" in messages
    assert "gemini-9.9-flash" in messages
    assert "gemini-9.9-pro" not in messages  # filtered: no "flash" in the name


def test_preflight_model_not_found_survives_a_broken_listing_call(caplog):
    """`client.models.list()` is itself guarded (best effort) — if listing
    also fails (e.g. the same outage that took out the model, or a transient
    network error), the preflight must still return cleanly, not raise."""

    class FakeNotFoundError(Exception):
        code = 404
        message = "not found"

    class PreflightModels:
        def get(self, **kwargs):
            raise FakeNotFoundError("NOT_FOUND")

        def list(self):
            raise RuntimeError("listing is also down")

    class PreflightClient:
        def __init__(self):
            self.models = PreflightModels()

    with caplog.at_level(logging.ERROR):
        available = briefs_module._model_available(PreflightClient(), "bogus-model")
    assert available is False  # still a clean "don't loop" signal, not a raised exception


def test_preflight_passes_through_a_model_that_exists():
    """A successful `models.get` (the common case) must let the caller
    proceed — the preflight only ever says "stop" on a confirmed 404."""

    class OkModels:
        def get(self, **kwargs):
            return SimpleNamespace(name=f"models/{kwargs['model']}")

    class OkClient:
        def __init__(self):
            self.models = OkModels()

    assert briefs_module._model_available(OkClient(), "gemini-3.1-flash-lite") is True


def test_preflight_treats_non_404_errors_as_inconclusive():
    """A non-404 error from `models.get` (network blip, auth hiccup, ...) is
    not confirmation the model is unavailable — the preflight must not block
    the run on it; generate_briefs's own per-call handling remains the net."""

    class FlakyModels:
        def get(self, **kwargs):
            raise FakeOtherError("temporary server error")

    class FlakyClient:
        def __init__(self):
            self.models = FlakyModels()

    assert briefs_module._model_available(FlakyClient(), "gemini-3.1-flash-lite") is True
