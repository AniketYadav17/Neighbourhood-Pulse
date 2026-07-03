"""Precomputed LLM neighbourhood briefs (Claude API at build time only).

The deployed app makes zero API calls: `pulse briefs` runs locally against the
committed gap artifact and commits the result as artifacts/briefs.json. Every
response is forced to a strict JSON schema server-side (output_config) and
re-validated locally with pydantic before acceptance. Per-hexagon caching makes
reruns fill only the gaps (resumable, like ingestion); a hard cost cap bounds
spend; the SDK's max_retries handles 429/5xx backoff.
"""

import json
import logging
import os

import pandas as pd
from pydantic import BaseModel, ConfigDict, ValidationError

from neighbourhood_pulse.config import (
    BRIEFS_MAX_COST_USD,
    BRIEFS_MAX_TOKENS,
    BRIEFS_MODEL,
    BRIEFS_N_HEXAGONS,
    BRIEFS_PATH,
    BRIEFS_PRICE_PER_MTOK,
    BRIEFS_TEMPERATURE,
    VALUATION_GAP_PATH,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You write short, grounded property-market briefs for London neighbourhoods.

Rules:
- Use ONLY the signals supplied in the user message. Never invent local facts,
  place names, amenities, or transport links that are not in the data.
- Analytical tone, not promotional. This is a research signal, not advice.
- Return JSON with exactly three keys: "headline" (at most 12 words), "brief"
  (2-3 sentences interpreting the signals), "caveat" (1 sentence on why the
  signal could be wrong for this specific hexagon)."""

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


class Brief(BaseModel):
    """Local re-validation of the model's JSON (belt and braces over output_config)."""

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
        f"Signal-implied (predicted) price: £{row['pred_price']:,.0f}\n"
        f"Valuation gap: {row['valuation_gap'] * 100:.1f}% (negative = priced below its signals)\n"
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


def generate_briefs(
    hexes: pd.DataFrame,
    client,
    briefs: dict,
    max_cost_usd: float = BRIEFS_MAX_COST_USD,
    save=None,
) -> dict:
    """Fill missing briefs; mutates and returns `briefs`.

    Resumable: cached hexagons are skipped. `save`, if given, is called with the
    dict after every accepted brief so partial progress survives any crash.
    Schema-invalid or refused responses are skipped, never fatal — a rerun
    retries only the gaps.
    """
    spent = 0.0
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
        response = client.messages.create(
            model=BRIEFS_MODEL,
            max_tokens=BRIEFS_MAX_TOKENS,
            temperature=BRIEFS_TEMPERATURE,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_prompt(row)}],
            output_config={"format": {"type": "json_schema", "schema": BRIEF_SCHEMA}},
        )
        spent += estimate_cost_usd(response.usage.input_tokens, response.usage.output_tokens)
        if response.stop_reason != "end_turn":
            logger.warning("%s: stop_reason=%s — skipped.", h3_index, response.stop_reason)
            continue
        text = next(block.text for block in response.content if block.type == "text")
        try:
            brief = Brief.model_validate(json.loads(text))
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning("%s: schema-invalid response skipped (%s).", h3_index, e)
            continue
        briefs[h3_index] = {**brief.model_dump(), "model": BRIEFS_MODEL}
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


def run_briefs(force: bool = False) -> dict:
    """CLI entry: gap artifact -> briefs.json (needs ANTHROPIC_API_KEY)."""
    import anthropic  # deferred: serving and app code paths never import the SDK

    gap = pd.read_parquet(VALUATION_GAP_PATH)
    hexes = select_brief_hexagons(gap)
    briefs = {} if force else load_briefs_file()
    client = anthropic.Anthropic(max_retries=5)  # SDK backs off on 429/5xx
    generate_briefs(hexes, client, briefs, save=write_briefs_file)
    write_briefs_file(briefs)
    logger.info("%s: %s briefs.", BRIEFS_PATH, len(briefs))
    return briefs
