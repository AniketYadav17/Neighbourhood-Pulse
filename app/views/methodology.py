"""Methodology — plain-English pipeline walkthrough, data sources, research record."""

import streamlit as st
from shared import load_gap

st.title("🧭 Methodology")

gap = load_gap()

st.markdown(
    """
### How it works, in four steps

1. **Collect.** Every planning application from all 33 London boroughs (about
   350,000) and every café on OpenStreetMap (about 6,600).
2. **Clean and locate.** Each record is placed into a hexagon of roughly 0.7
   square kilometres (H3 resolution 8), with a correction for boroughs that
   report planning data late.
3. **Model.** XGBoost learns to estimate each hexagon's typical sale price
   (HM Land Registry sales, 2021 to 2025) from the activity signals plus
   distance from central London. Every hexagon is scored by a model that
   never saw that hexagon's own price, so the estimate is honest.
4. **Explain.** The map shows where actual prices sit above or below the
   model's estimate. Short AI briefs, generated once at build time and then
   cached, summarise the signals for the 50 most undervalued areas.

### The valuation gap

The gap is `actual price / model estimate − 1`. Negative means the area sells
for less than its activity signals suggest. The model explains under half of
price variation, so treat a large gap as a place to look closer, not a verdict.

### For reproducers

```
pulse ingest     Planning London Datahub (33 boroughs, scroll API, resumable)
                 + OpenStreetMap cafés (OSMnx)
pulse transform  BNG -> WGS84, H3 resolution-8 hexagon indexing
pulse train      per-borough lag-trimmed features -> Land Registry price target
                 -> XGBoost on log(median price) -> out-of-fold valuation gap
pulse briefs     Gemini API neighbourhood briefs (build time; the app makes
                 zero API calls)
```

Every stage skips itself if its output already exists, and the app reads only
the four committed files in `artifacts/`. These are the same files CI
load-tests on every push.

### Data sources

- [Planning London Datahub](https://planninglondondatahub.london.gov.uk):
  every planning application, with status and change-of-use detail
- [OpenStreetMap via OSMnx](https://github.com/gboeing/osmnx): café locations,
  split into independents and chains
- [HM Land Registry Price Paid](https://landregistry.data.gov.uk/): actual
  sale prices 2021–2025, the ground truth the model is scored against
"""
)

st.caption(
    f"Current artifact: {len(gap):,} modelled hexagons. Full research record: "
    "`notebooks/01_eda.ipynb` (frozen); engineering decisions: `docs/DECISIONS.md`; "
    "system overview: `docs/ARCHITECTURE.md`."
)
