"""Precomputed LLM neighbourhood briefs (Gemini API at build time only).

The deployed app makes zero API calls: `pulse briefs` runs locally against the
committed gap artifact and commits the result as artifacts/briefs.json. Every
response is forced to a strict JSON schema server-side (response_schema) and
re-validated locally with pydantic before acceptance. Per-hexagon caching makes
reruns fill only the gaps (resumable, like ingestion); a hard cost cap bounds
spend. The SDK's HttpRetryOptions handles transient 408/5xx backoff; free-tier
429s (per-minute RPM limits) are additionally paced up front
(BRIEFS_MIN_REQUEST_INTERVAL_S) and, if one still slips through, retried
per-hexagon with a longer sleep. A 429 whose *daily* quota is exhausted
instead stops the whole run cleanly (no point retrying within a sleep loop
against a quota that only resets at midnight Pacific) — see generate_briefs.

The configured model itself is a moving target — Google has deprecated it
three times in eight days — so BRIEFS_MODEL is only ever a default: `pulse
briefs --model NAME` overrides it end to end (run_briefs -> generate_briefs ->
the per-brief "model" field) with no code edit needed. A 404 on the model is
treated as a fatal configuration error rather than a per-hexagon fluke, since
every subsequent call would fail identically: a cheap preflight checks the
model before the run starts (see _model_available) and, as a backstop, the
generation loop itself aborts immediately on a 404 (see generate_briefs) —
either way the run stops with one clear message instead of 404ing through
every remaining hexagon.

Needs GEMINI_API_KEY (or GOOGLE_API_KEY) set in the environment — never
hardcoded.
"""

import json
import logging
import os
import time

import pandas as pd
from pydantic import BaseModel, ConfigDict, ValidationError

from neighbourhood_pulse.config import (
    BRIEFS_MAX_COST_USD,
    BRIEFS_MAX_TOKENS,
    BRIEFS_MIN_REQUEST_INTERVAL_S,
    BRIEFS_MODEL,
    BRIEFS_N_HEXAGONS,
    BRIEFS_PATH,
    BRIEFS_PRICE_PER_MTOK,
    BRIEFS_TEMPERATURE,
    BRIEFS_THINKING_CONFIG,
    VALUATION_GAP_PATH,
)

logger = logging.getLogger(__name__)

# Per-hexagon quota-retry policy for 429s: sleep and retry a handful of times,
# then skip the hexagon like any other failure. A single throttled or broken
# hexagon must never crash the whole (resumable) run.
QUOTA_MAX_RETRIES = 3
QUOTA_RETRY_SLEEP_S = 60.0

SYSTEM_PROMPT = """You write short, honest property-market notes on London neighbourhoods
for a general reader.

Rules:
- Use ONLY the facts supplied in the user message. Never invent local facts,
  place names, amenities, or transport links that are not in the data.
- Plain English a non-specialist understands. No jargon, no filler.
- The predicted figure is "the model's estimate". Never call it the market rate,
  a discount, or what the area is "trading at". The actual sale price
  IS the market; the estimate is a statistical guess from activity signals.
- Describe a negative gap as "sells for X% less than the model estimates".
  A large gap can mean the model is wrong about this area (unusual housing
  mix, factors it cannot see), not confirmed opportunity. Avoid certainty
  words such as "significantly undervalued".
- Analytical tone, not promotional. This is a research signal, not advice.
- Return JSON with exactly three keys: "headline" (at most 12 words), "brief"
  (2-3 sentences interpreting the facts), "caveat" (1 sentence on why the
  signal could be wrong for this specific area)."""

BRIEF_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "brief": {"type": "string"},
        "caveat": {"type": "string"},
    },
    "required": ["headline", "brief", "caveat"],
    "additionalProperties": False,
}

# Gemini's response_schema is an OpenAPI-subset Schema proto that does not
# accept "additionalProperties" — the REST layer rejects it outright with
# `400 INVALID_ARGUMENT: Unknown name "additional_properties"`. Send Gemini a
# sanitized copy; BRIEF_SCHEMA above stays the canonical, fully-strict schema.
# Extra-key rejection is still enforced locally by the pydantic `Brief` model
# (`extra="forbid"`) — belt and braces was always the design, this just moves
# where the "no extra keys" rule is checked.
GEMINI_RESPONSE_SCHEMA = {k: v for k, v in BRIEF_SCHEMA.items() if k != "additionalProperties"}


class Brief(BaseModel):
    """Local re-validation of the model's JSON (belt and braces over response_schema)."""

    model_config = ConfigDict(extra="forbid")

    headline: str
    brief: str
    caveat: str


def select_brief_hexagons(gap: pd.DataFrame, n: int = BRIEFS_N_HEXAGONS) -> pd.DataFrame:
    """The n most undervalued hexagons — the ones users will click first."""
    return gap.nsmallest(n, "valuation_gap").reset_index(drop=True)


def build_user_prompt(row: pd.Series) -> str:
    """Every fact the model may use — nothing else exists as far as it knows."""
    return (
        f"Borough: {row['borough']}\n"
        f"Actual median sale price (2021-2025 pooled): £{row['median_price']:,.0f}\n"
        f"Model's estimated price: £{row['pred_price']:,.0f}\n"
        f"Valuation gap: {row['valuation_gap'] * 100:.1f}% (negative = sells below the model's estimate)\n"
        f"Planning applications (5-year window): {row['total_applications']:.0f}\n"
        f"Applications in the last 12 months: {row['applications_recent']:.0f}\n"
        f"Change-of-use applications: {row['change_of_use_count']:.0f} "
        f"({row['change_of_use_ratio'] * 100:.1f}% of all applications)\n"
        f"Planning velocity (recent rate vs own history, 1.0 = steady): "
        f"{row['planning_velocity']:.2f}\n"
        f"Cafes: {row['total_cafe_count']:.0f} total, "
        f"{row['independent_cafe_count']:.0f} independent\n"
        f"Distance to central London: {row['dist_to_centre_km']:.1f} km"
    )


def estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens * BRIEFS_PRICE_PER_MTOK["input"]
        + output_tokens * BRIEFS_PRICE_PER_MTOK["output"]
    ) / 1e6


def _is_quota_error(exc: Exception) -> bool:
    """Duck-types the SDK's 429 (google.genai.errors.ClientError) without
    importing google.genai here, so this module stays import-light and the
    fake-client tests never need the real SDK installed."""
    return getattr(exc, "code", None) == 429


def _is_not_found_error(exc: Exception) -> bool:
    """Duck-types the SDK's 404 (google.genai.errors.ClientError), the same
    way _is_quota_error duck-types 429s. A 404 on a model-scoped call means
    the configured model doesn't exist for this project — deprecated, not
    (yet) available, or simply mistyped — so every subsequent call with the
    same model would fail identically. That makes it a fatal configuration
    error, unlike an ordinary per-hexagon API hiccup."""
    return getattr(exc, "code", None) == 404


def _find_quota_id(payload: object) -> str | None:
    """Recursively hunts a nested error payload for a QuotaFailure
    violation's `quotaId`, without assuming exact nesting. Google's 429 body
    is (roughly) `{"error": {"details": [{"violations": [{"quotaId": ...}]}]}}`,
    but this walks any dict/list shape defensively rather than indexing into
    it, since the exact structure isn't a documented contract."""
    if isinstance(payload, dict):
        quota_id = payload.get("quotaId")
        if isinstance(quota_id, str):
            return quota_id
        for value in payload.values():
            found = _find_quota_id(value)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _find_quota_id(item)
            if found is not None:
                return found
    return None


def _daily_quota_id(exc: Exception) -> str | None:
    """Best-effort extraction of a 429's QuotaFailure quotaId, duck-typed
    against google.genai.errors.APIError's `.details` attribute (the parsed
    JSON response body — confirmed via the installed SDK's errors.py, where
    `APIError.__init__` sets `self.details = response_json`). Returns None if
    the attribute is missing or the shape doesn't match; callers must treat
    that as "can't tell", not "not a daily quota", and fall back to the
    existing per-minute retry behavior."""
    details = getattr(exc, "details", None)
    if details is None:
        return None
    return _find_quota_id(details)


def generate_briefs(
    hexes: pd.DataFrame,
    client,
    briefs: dict,
    max_cost_usd: float = BRIEFS_MAX_COST_USD,
    save=None,
    request_interval_s: float = BRIEFS_MIN_REQUEST_INTERVAL_S,
    model: str = BRIEFS_MODEL,
) -> dict:
    """Fill missing briefs; mutates and returns `briefs`.

    Resumable: cached hexagons are skipped. `save`, if given, is called with the
    dict after every accepted brief so partial progress survives any crash.
    Schema-invalid or refused responses are skipped, never fatal — a rerun
    retries only the gaps. `model` (default BRIEFS_MODEL) is recorded on every
    accepted brief so a run generated with `pulse briefs --model X` is
    traceable per-hexagon, not just via the config default.

    Requests are paced at least `request_interval_s` apart (free-tier RPM
    limits) measuring from the previous call's start, so a slow call doesn't
    stack an unnecessary full-length sleep on top. A 429 quota error retries
    the same hexagon (sleeping `QUOTA_RETRY_SLEEP_S`) up to `QUOTA_MAX_RETRIES`
    times before that hexagon is skipped; any other API error skips the
    hexagon immediately. Either way the loop continues — one bad or throttled
    hexagon never kills the run.

    Exception: a 429 whose QuotaFailure quotaId contains "PerDay" means the
    free-tier *daily* budget is exhausted — retrying is pointless, since it
    won't clear until the quota resets at midnight Pacific, not within this
    process's lifetime. That case stops the whole run immediately (no sleep,
    no retry), not just the current hexagon; everything generated so far is
    already saved per-brief, so a rerun tomorrow continues from where this
    one stopped. If the quotaId can't be determined from the error's payload,
    this falls back to the ordinary per-minute retry-then-skip behavior.

    A second, unconditional exception: a 404 on `model` means the model
    itself is unavailable (deprecated, not yet available to new users, or
    mistyped) — every remaining hexagon would fail identically, so this also
    stops the whole run immediately (see _is_not_found_error), the same
    clean-stop shape as the daily-quota case. `run_briefs`'s preflight
    (_model_available) normally catches this before the loop even starts;
    this is the backstop for whenever that preflight is bypassed or the model
    becomes unavailable mid-run.
    """
    spent = 0.0
    last_call_start = None
    for _, row in hexes.iterrows():
        h3_index = row["h3_index"]
        if h3_index in briefs:
            continue
        if spent >= max_cost_usd:
            logger.warning(
                "Cost cap $%.2f reached ($%.4f spent); stopping — rerun to continue.",
                max_cost_usd,
                spent,
            )
            break

        response = None
        quota_retries = 0
        daily_quota_hit = False
        model_not_found = False
        while True:
            if last_call_start is not None and request_interval_s > 0:
                wait = request_interval_s - (time.monotonic() - last_call_start)
                if wait > 0:
                    time.sleep(wait)
            last_call_start = time.monotonic()
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=build_user_prompt(row),
                    config={
                        "system_instruction": SYSTEM_PROMPT,
                        "temperature": BRIEFS_TEMPERATURE,
                        "max_output_tokens": BRIEFS_MAX_TOKENS,
                        "thinking_config": BRIEFS_THINKING_CONFIG,
                        "response_mime_type": "application/json",
                        "response_schema": GEMINI_RESPONSE_SCHEMA,
                    },
                )
            except Exception as e:
                response = None
                if _is_not_found_error(e):
                    logger.error(
                        "%s: model %r not found (%s). Every remaining hexagon would "
                        "fail identically, so this is a configuration error, not a "
                        "per-hexagon fluke — stopping now instead of churning through "
                        "the rest. This usually means the model has been deprecated "
                        "or is no longer available to new users. Pass a current model "
                        "with `pulse briefs --model <name>` (see "
                        "https://ai.google.dev/gemini-api/docs/models for current "
                        "names). %d/%d briefs already generated are saved.",
                        h3_index,
                        model,
                        getattr(e, "message", None) or e,
                        len(briefs),
                        len(hexes),
                    )
                    model_not_found = True
                    break
                if _is_quota_error(e):
                    quota_id = _daily_quota_id(e)
                    if quota_id is not None and "PerDay" in quota_id:
                        logger.warning(
                            "%s: daily free-tier quota exhausted (quotaId=%s) — "
                            "%d/%d briefs cached; rerun tomorrow (quota resets "
                            "midnight Pacific) or switch to a paid tier.",
                            h3_index,
                            quota_id,
                            len(briefs),
                            len(hexes),
                        )
                        daily_quota_hit = True
                        break
                    if quota_retries < QUOTA_MAX_RETRIES:
                        quota_retries += 1
                        logger.warning(
                            "%s: quota error (attempt %d/%d) — sleeping %.0fs and retrying.",
                            h3_index,
                            quota_retries,
                            QUOTA_MAX_RETRIES,
                            QUOTA_RETRY_SLEEP_S,
                        )
                        time.sleep(QUOTA_RETRY_SLEEP_S)
                        continue
                    logger.warning(
                        "%s: quota still exceeded after %d retries — skipped.",
                        h3_index,
                        QUOTA_MAX_RETRIES,
                    )
                else:
                    logger.warning("%s: API error (%s) — skipped.", h3_index, e)
                break
            else:
                break

        if model_not_found:
            break

        if daily_quota_hit:
            break

        if response is None:
            continue

        usage = response.usage_metadata
        if usage is not None:
            # Paid-tier billing counts thinking tokens as output; fold them in
            # here so the cost cap reflects what the invoice will actually say.
            thoughts_tokens = getattr(usage, "thoughts_token_count", None) or 0
            spent += estimate_cost_usd(
                usage.prompt_token_count or 0,
                (usage.candidates_token_count or 0) + thoughts_tokens,
            )
        if not response.candidates:
            logger.warning("%s: no candidates in response (likely blocked) — skipped.", h3_index)
            continue
        finish_reason = response.candidates[0].finish_reason
        if finish_reason is not None and finish_reason != "STOP":
            logger.warning("%s: finish_reason=%s — skipped.", h3_index, finish_reason)
            continue
        text = response.text
        if not text:
            logger.warning("%s: empty response text — skipped.", h3_index)
            continue
        try:
            brief = Brief.model_validate(json.loads(text))
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning("%s: schema-invalid response skipped (%s).", h3_index, e)
            continue
        briefs[h3_index] = {**brief.model_dump(), "model": model}
        if save is not None:
            save(briefs)
        logger.info(
            "Brief %s/%s (%s) — $%.4f spent.", len(briefs), len(hexes), row["borough"], spent
        )
    return briefs


def load_briefs_file(path: str = BRIEFS_PATH) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_briefs_file(briefs: dict, path: str = BRIEFS_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(briefs, f, indent=2, ensure_ascii=False)


def _model_available(client, model: str) -> bool:
    """Cheap preflight: confirms `model` exists before generate_briefs spends
    even one call of the per-hexagon budget on it. Returns False (after
    logging one clear error, plus a best-effort list of currently-available
    flash-family models) if `model` 404s — every generate_content call would
    fail identically, so the caller should not loop at all. Any other
    exception (network blip, auth hiccup, ...) is inconclusive, not a
    confirmed problem with `model` itself, so it's treated as "go ahead" —
    generate_briefs's own per-call handling (and the SDK's built-in retries)
    is the unchanged backstop for those, same as before this preflight
    existed."""
    try:
        client.models.get(model=model)
    except Exception as e:
        if not _is_not_found_error(e):
            return True
        logger.error(
            "%s: model not found (%s). This usually means the model has been "
            "deprecated or is no longer available to new users. Pass a "
            "current model with `pulse briefs --model <name>` (see "
            "https://ai.google.dev/gemini-api/docs/models for current names).",
            model,
            getattr(e, "message", None) or e,
        )
        try:
            available = sorted(
                {
                    m.name.removeprefix("models/")
                    for m in client.models.list()
                    if m.name and "flash" in m.name
                }
            )
        except Exception as list_exc:
            logger.warning("Could not list available models (best effort): %s", list_exc)
        else:
            logger.error(
                "Flash-family models currently available: %s",
                ", ".join(available) if available else "(none found)",
            )
        return False
    return True


def run_briefs(force: bool = False, model: str | None = None) -> dict:
    """CLI entry: gap artifact -> briefs.json (needs GEMINI_API_KEY or GOOGLE_API_KEY).

    `model` (default None) overrides BRIEFS_MODEL for this run only — the
    override behind `pulse briefs --model NAME`, so a model deprecation can be
    worked around without editing config.py. A cheap preflight
    (_model_available) checks the model before the loop starts; on a 404 it
    logs a clear diagnostic and this returns immediately without looping,
    leaving whatever was already cached untouched.
    """
    from google import genai  # deferred: serving and app code paths never import the SDK
    from google.genai import types

    model = model or BRIEFS_MODEL
    gap = pd.read_parquet(VALUATION_GAP_PATH)
    hexes = select_brief_hexagons(gap)
    briefs = {} if force else load_briefs_file()
    # SDK backs off on 408/429/5xx; 5 attempts / 1s initial / 60s max / 2x backoff
    # are the library defaults, made explicit here since we rely on them.
    retry_options = types.HttpRetryOptions(
        attempts=5, initial_delay=1.0, max_delay=60.0, exp_base=2.0
    )
    client = genai.Client(http_options=types.HttpOptions(retry_options=retry_options))
    if not _model_available(client, model):
        return briefs
    generate_briefs(hexes, client, briefs, save=write_briefs_file, model=model)
    write_briefs_file(briefs)
    logger.info("%s: %s briefs.", BRIEFS_PATH, len(briefs))
    return briefs
