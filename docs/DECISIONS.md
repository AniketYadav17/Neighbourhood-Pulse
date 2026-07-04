# Design decisions

Short records of the decisions that shaped this system, roughly in pipeline order.

## 1. H3 resolution 8 as the unit of analysis
~0.74 km² hexagons balance signal density against locality: coarse enough that
most hexagons accumulate enough planning applications and sales to be stable,
fine enough to distinguish neighbourhoods. Res-9 starved the price target
(median <30 sales per cell); borough-level would erase the phenomenon.

## 2. Valuation gap on price levels, not temporal price prediction
Predicting *future* prices needs years of feature history the sources don't
provide. Instead the model learns what a hexagon "should" cost given its
signals today; the gap (actual vs predicted) flags mispricing. The back-test
then validates the temporal claim separately: early gaps precede later growth.

## 3. Per-borough reporting-lag trim with a contiguous-tail rule
Planning records arrive with borough-specific ingestion lag, so recent months
are under-reported at different rates per borough. Each borough's trailing
months below 75% of its own median monthly count are trimmed — contiguous tail
only (lag is monotonic from the edge; an interior dip is real signal), capped
at 4 months (a longer run is decline, not lag — flagged, not trimmed).

## 4. Single-frame borough rule for straddling hexagons
A hexagon crossing a borough boundary gets ONE governing frame (its modal
borough's anchor/span/trim). Mixing frames would divide one borough's records
by another's span — re-introducing the fake-acceleration bias the trim removes.

## 5. Median pooled price, ≥30 sales per hexagon
Sales are pooled over 2021–2025 because per-year medians are too thin at res-8.
Median over mean: single £20m transactions shouldn't move a neighbourhood.
The 30-sale floor keeps the target stable at the cost of coverage (~66%).

## 6. Out-of-fold predictions for the gap; full fit only for serving
The gap must be honest: with in-sample predictions an overfit model calls
everything fairly priced. 5-fold `cross_val_predict` means no hexagon is scored
by a model that saw it. The committed `model.joblib` is a separate full-data
fit used ONLY for what-if repricing — never for the gap.

## 7. The science is frozen; a parity gate proves it
The refactor from notebook to package changed packaging, not results: the
pipeline's gap artifact must match the notebook-era golden (identical hexagon
set and prices, gap correlation >0.999). Any code change that moves outputs
beyond tolerance is a bug by definition.

## 8. Precomputed LLM briefs, committed as an artifact
Briefs are generated at build time (`pulse briefs`) with a closed-world prompt
(only supplied signals may be cited), server-side JSON-schema enforcement, and
local re-validation. The deployed app makes zero API calls: no key to leak, no
latency, no cost surprises, fully reproducible reviews. Per-hexagon caching and
a hard cost cap make regeneration safe.

## 9. pydeck H3HexagonLayer over folium
v1 hand-built 2,567 folium polygons into an HTML string (slow, no interaction).
pydeck renders H3 natively on the GPU and gives click-to-select through
`st.pydeck_chart(on_select=...)` — which is what makes the detail panel and
what-if workflow possible. The folium map remains in the notebook as the
research artifact.

## 10. What-if inputs are clamped to observed feature ranges
Tree ensembles extrapolate silently and meaninglessly. Slider bounds and the
API's `/predict` validation both derive from the gap artifact's per-feature
min/max: repricing outside the model's observed envelope is refused, not
mispredicted.

## 11. The API ships via docker-compose, not a second cloud deployment
A bare online-inference endpoint would be synthetic for a batch model. The
what-if panel gives `/predict` a real client under compose, while the public
Streamlit app stays standalone — one deployment, no uptime coupling, the
portfolio demo can't be taken down by a free-tier API host.

## 12. Reproducibility: data-anchored dates, seeded models, one caveat
Training features anchor to each borough's own `max(valid_date)` — a function
of the data, not the run date — and every model uses `random_state=42`, so
`pulse train` is reproducible given `data/raw/`. Known caveat: postcode→hexagon
and hexagon→borough modal lookups break ties by first-seen order
(`value_counts().idxmax()`), so a pathological exact tie is input-order
dependent. Observed real-data impact: none (the parity gate is bit-exact).

## 13. No MLflow, no scheduled retraining
One model, one dataset vintage, one developer: experiment tracking would be
ceremony. Provenance lives in `metrics.json` (`build`: versions, git SHA,
train date); retraining is `pulse train --force`. The scaling story (PostGIS,
scheduled ingestion, drift monitoring) is documented future work, not
half-built infra.
