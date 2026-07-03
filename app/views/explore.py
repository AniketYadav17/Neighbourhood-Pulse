"""Explore — the hero page: hexagon map, filters, detail panel, brief, what-if."""

import pydeck as pdk
import streamlit as st
from shared import derive_what_if_features, gap_colour, load_briefs, load_gap, reprice

from neighbourhood_pulse.model import feature_bounds

st.title("🗺️ The Neighbourhood Pulse")
st.markdown(
    "**A gentrification predictor for London.** Each hexagon's price is modelled from "
    "*pre-gentrification signals* — planning applications, change-of-use conversions, and "
    "independent-café density — plus a centrality control. The **valuation gap** is actual "
    "median sale price vs the price the signals predict: hexagons priced **below** their "
    "signal-implied level (red) are *candidate undervalued* neighbourhoods."
)

gap = load_gap()
briefs = load_briefs()
bounds = feature_bounds(gap)

with st.sidebar:
    st.header("Filters")
    boroughs = st.multiselect("Borough", sorted(gap["borough"].unique()))
    lo = float(gap["valuation_gap"].min() * 100)
    hi = float(gap["valuation_gap"].max() * 100)
    gap_lo, gap_hi = st.slider("Valuation gap (%)", lo, hi, (lo, hi))

view = gap
if boroughs:
    view = view[view["borough"].isin(boroughs)]
view = view[view["valuation_gap"].between(gap_lo / 100, gap_hi / 100)]

# Symmetric diverging scale anchored on the FULL dataset so filters don't recolour.
scale = float(gap["valuation_gap"].abs().quantile(0.90))
data = view[["h3_index", "borough", "median_price", "pred_price", "valuation_gap"]].copy()
data["fill"] = data["valuation_gap"].map(lambda g: gap_colour(g, scale) + [170])
data["gap_pct"] = (data["valuation_gap"] * 100).round(1)

left, right = st.columns([3, 2], gap="large")

with left:
    layer = pdk.Layer(
        "H3HexagonLayer",
        id="gap",
        data=data,
        get_hexagon="h3_index",
        get_fill_color="fill",
        pickable=True,
        stroked=False,
        extruded=False,
    )
    event = st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=pdk.ViewState(latitude=51.51, longitude=-0.10, zoom=9),
            map_style=None,
            tooltip={"text": "{borough}: gap {gap_pct}%"},
        ),
        on_select="rerun",
        selection_mode="single-object",
        height=560,
    )
    st.caption(
        f"{len(view):,} of {len(gap):,} modelled hexagons shown · "
        "red = undervalued, green = overvalued · click a hexagon for detail"
    )

selected_objects = event.selection.objects.get("gap", [])
selected = selected_objects[0] if selected_objects else None

with right:
    st.subheader("Hexagon detail")
    if selected is None:
        st.info("Click a hexagon on the map to see its signals, brief, and what-if repricing.")
        st.stop()

    row = gap.set_index("h3_index").loc[selected["h3_index"]]
    c1, c2, c3 = st.columns(3)
    c1.metric("Actual median £", f"£{row['median_price']:,.0f}")
    c2.metric("Predicted £", f"£{row['pred_price']:,.0f}")
    c3.metric("Gap", f"{row['valuation_gap'] * 100:+.1f}%")
    st.caption(
        f"{row['borough']} · {int(row['sales_count'])} pooled sales (2021–2025) · "
        f"{row['dist_to_centre_km']:.1f} km from Charing Cross"
    )

    brief = briefs.get(selected["h3_index"])
    if brief:
        st.markdown(f"**{brief['headline']}**")
        st.write(brief["brief"])
        st.caption(f"⚠️ {brief['caveat']}")

    st.markdown("#### What-if repricing")
    st.caption(
        "Move the base signals; ratios and velocity are recomputed and the trained "
        "model reprices the hexagon live. Sliders clamp to the observed data range."
    )

    def _slider(label: str, col: str) -> float:
        lo, hi = int(bounds[col][0]), int(bounds[col][1])
        return float(st.slider(label, lo, hi, int(row[col]), key=f"whatif_{col}"))

    edits = {
        "applications_recent": _slider("Applications (last 12 months)", "applications_recent"),
        "change_of_use_count": _slider("Change-of-use applications", "change_of_use_count"),
        "total_cafe_count": _slider("Cafés (total)", "total_cafe_count"),
    }
    edits["independent_cafe_count"] = float(
        st.slider(
            "Cafés (independent)",
            0,
            max(int(edits["total_cafe_count"]), 1),
            min(int(row["independent_cafe_count"]), int(edits["total_cafe_count"])),
            key="whatif_independent",
        )
    )

    try:
        price = reprice(derive_what_if_features(row, edits))
    except Exception as exc:  # API down under compose, etc. — degrade, don't crash
        st.error(f"Repricing unavailable: {exc}")
    else:
        new_gap = row["median_price"] / price - 1
        st.metric(
            "What-if predicted £",
            f"£{price:,.0f}",
            delta=f"{(price / row['pred_price'] - 1) * 100:+.1f}% vs current prediction",
        )
        st.caption(
            f"Gap would move from {row['valuation_gap'] * 100:+.1f}% to {new_gap * 100:+.1f}% "
            "(actual price held fixed)."
        )
