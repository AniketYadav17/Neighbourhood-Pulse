"""Methodology — pipeline, data sources, and where the research lives."""

import streamlit as st
from shared import load_gap

st.title("🧭 Methodology")

gap = load_gap()

st.markdown(
    """
### Pipeline

```
pulse ingest     Planning London Datahub (33 boroughs, scroll API, resumable)
                 + OpenStreetMap cafés (OSMnx)
pulse transform  BNG -> WGS84, H3 resolution-8 hexagon indexing
pulse train      per-borough lag-trimmed features -> Land Registry price target
                 -> XGBoost on log(median price) -> out-of-fold valuation gap
pulse briefs     Claude API neighbourhood briefs (build time; the app makes
                 zero API calls)
```

Every stage is idempotent (skip-if-artifact-exists) and the app reads only the
four committed artifacts in `artifacts/` — this page is served by the same
files CI load-tests on every push.

### The valuation gap

XGBoost predicts `log(median price)` per hexagon from planning + café signals
plus a single centrality control. Predictions are **out-of-fold** (5-fold
`cross_val_predict`): no hexagon is scored by a model that saw it. The gap is
`actual / predicted − 1` — negative means priced below what the signals imply.

### Data sources
- [Planning London Datahub](https://planninglondondatahub.london.gov.uk) — planning applications, all 33 boroughs
- [OpenStreetMap via OSMnx](https://github.com/gboeing/osmnx) — café locations and brands
- [HM Land Registry Price Paid](https://landregistry.data.gov.uk/) — residential sales 2021–2025 (ground truth)
"""
)

st.caption(
    f"Current artifact: {len(gap):,} modelled hexagons. Full research record: "
    "`notebooks/01_eda.ipynb` (frozen); engineering decisions: `docs/DECISIONS.md`; "
    "system overview: `docs/ARCHITECTURE.md`."
)
