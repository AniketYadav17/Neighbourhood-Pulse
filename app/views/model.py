"""Model — metrics.json rendered honestly: R², importances, back-test, caveats."""

import pandas as pd
import streamlit as st
from shared import load_metrics

st.title("📈 Model")

metrics = load_metrics()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Held-out R² (XGBoost)", f"{metrics['r2_xgboost']:.3f}")
c2.metric("Held-out R² (Linear)", f"{metrics['r2_linear']:.3f}")
c3.metric("Modelled hexagons", f"{metrics['n_hexagons']:,}")
c4.metric("Back-test hexagons", f"{metrics['backtest']['n_hexagons']:,}")

st.subheader("What drives the prediction")
importance = (
    pd.Series(metrics["feature_importance"]).sort_values(ascending=True).rename("importance")
)
st.bar_chart(importance, horizontal=True)
st.caption(
    "`change_of_use_ratio` is the #1 predictor even after controlling for centrality — "
    "the gentrification signal carries information independent of location."
)

st.subheader("Back-test: does early undervaluation precede growth?")
backtest = metrics["backtest"]
st.markdown(
    f"Hexagons flagged undervalued **early** (2021–22) subsequently grew more. "
    f"Gap-vs-growth correlation: **{backtest['correlation']:.3f}** "
    f"(negative = undervalued grows more), monotonic across quintiles:"
)
quintiles = pd.Series(backtest["quintiles"]).rename("mean growth 2024–25 vs 2021–22 (%)")
st.bar_chart(quintiles)
st.dataframe(quintiles.round(2), use_container_width=True)

with st.expander("Honest caveats"):
    st.markdown(
        "- **Location confound is reduced, not eliminated** — one distance-to-centre "
        "feature can't fully model London's polycentric price surface; some outer "
        "boroughs look undervalued partly due to location. Candidate signal, not advice.\n"
        "- **Back-test is proof-of-concept, not causal proof** — features are not "
        "strictly frozen as of 2021, and 2-year price windows are thin.\n"
        "- **Coverage** — hexagons with enough planning activity *and* ≥30 pooled "
        "sales (~66% of London), not the whole city."
    )

build = metrics.get("build", {})
if build:
    versions = build.get("versions", {})
    st.caption(
        f"Trained {build.get('trained_at', '?')} · commit `{build.get('git_sha', '?')}` · "
        f"xgboost {versions.get('xgboost', '?')} · scikit-learn {versions.get('scikit_learn', '?')}"
    )
