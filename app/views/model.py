"""Model — metrics.json rendered honestly: R², importances, gaps, back-test, caveats."""

import pandas as pd
import streamlit as st
from shared import load_gap, load_metrics

FEATURE_LABELS = {
    "total_applications": "Planning applications (5 years)",
    "change_of_use_count": "Applications changing a building's use",
    "applications_recent": "Planning applications (last 12 months)",
    "change_of_use_ratio": "Share of applications changing a building's use",
    "planning_velocity": "Recent planning activity vs the area's own history",
    "total_cafe_count": "Cafés (total)",
    "independent_cafe_count": "Cafés (independent)",
    "cafe_to_application_ratio": "Cafés per planning application",
    "dist_to_centre_km": "Distance from central London (km)",
}

QUINTILE_LABELS = {
    "Q1 most undervalued": "1 · most underpriced",
    "Q2": "2",
    "Q3": "3",
    "Q4": "4",
    "Q5 most overvalued": "5 · most overpriced",
}

st.title("📈 Model")

metrics = load_metrics()
gap = load_gap()

st.markdown(
    "The model reads an area's activity signals (planning applications, building "
    "conversions, cafés) plus its distance from central London, and estimates what "
    "homes there should cost. This page shows how good that estimate is and what "
    "drives it."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("R² (XGBoost)", f"{metrics['r2_xgboost']:.3f}")
c2.metric("R² (linear baseline)", f"{metrics['r2_linear']:.3f}")
c3.metric("Modelled areas", f"{metrics['n_hexagons']:,}")
c4.metric("Back-tested areas", f"{metrics['backtest']['n_hexagons']:,}")
st.caption(
    f"R² of {metrics['r2_xgboost']:.2f} means the model explains about "
    f"{metrics['r2_xgboost']:.0%} of the variation in neighbourhood prices, measured "
    "on areas it never trained on. The rest is things it cannot see, like property "
    "condition or the exact street."
)

st.subheader("What drives the estimate")
importance = (
    pd.Series(metrics["feature_importance"])
    .rename(index=FEATURE_LABELS)
    .sort_values(ascending=True)
    .rename("importance")
)
st.bar_chart(importance, horizontal=True)
st.caption(
    "The strongest predictor is the share of applications changing a building's use "
    "(shops becoming cafés or flats), even after accounting for how central an area "
    "is. The change signal carries information beyond location."
)

st.subheader("How big are the gaps?")
gap_pct = (gap["valuation_gap"] * 100).clip(-75, 100)
bins = pd.cut(gap_pct, bins=range(-80, 101, 10))
hist = gap_pct.groupby(bins, observed=False).size()
hist.index = [f"{int(iv.left)} to {int(iv.right)}" for iv in hist.index]
st.bar_chart(hist.rename("areas"), x_label="price vs estimate (%)", y_label="areas")
st.caption(
    "Gaps of 30 to 40% either way are common: the model is a rough guide, not a "
    "valuation. A handful of extreme areas beyond this range are grouped into the "
    "outermost bars. Treat any single area's gap as a place to look closer, not a "
    "verdict."
)

st.subheader("Did the model's picks actually grow?")
backtest = metrics["backtest"]
q = backtest["quintiles"]
st.markdown(
    f"A back-test asks: did areas that looked underpriced early (2021 to 2022) grow "
    f"more by 2024 to 2025? On average, yes. The most underpriced fifth of areas grew "
    f"about {q['Q1 most undervalued']:.0f}% while the most overpriced fifth fell about "
    f"{abs(q['Q5 most overvalued']):.0f}%, with a steady gradient in between "
    f"(correlation {backtest['correlation']:.2f})."
)
quintiles = pd.Series(q).rename(index=QUINTILE_LABELS).rename("mean growth (%)")
st.bar_chart(quintiles, x_label="fifths by early gap", y_label="mean growth 2024–25 vs 2021–22 (%)")

with st.expander("What this cannot tell you"):
    st.markdown(
        "- **Location is only partly accounted for.** One distance-from-centre number "
        "cannot capture all of London's many centres, so some outer areas look "
        "underpriced partly because of where they are.\n"
        "- **The back-test is indicative, not proof.** The signals were not strictly "
        "frozen at 2021, and two-year windows of sales are thin at this map scale.\n"
        "- **Coverage is about two thirds of London.** An area needs planning activity "
        "and at least 30 home sales in 2021–2025 to be modelled.\n"
        "- **This is a research signal, not investment advice.**"
    )

build = metrics.get("build", {})
if build:
    versions = build.get("versions", {})
    st.caption(
        f"Trained {build.get('trained_at', '?')} · commit `{build.get('git_sha', '?')}` · "
        f"xgboost {versions.get('xgboost', '?')} · scikit-learn {versions.get('scikit_learn', '?')}"
    )
