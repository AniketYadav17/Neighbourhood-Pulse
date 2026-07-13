"""Explore — the hero page: hexagon map, filters, detail panel, brief, what-if."""

import pydeck as pdk
import streamlit as st
from shared import (
    derive_what_if_features,
    fmt_gbp,
    gap_colour,
    load_briefs,
    load_gap,
    reprice,
)

from neighbourhood_pulse.model import feature_bounds

# UI-only filter presets (not config.py: this threshold is a display choice).
GAP_PRESET_THRESHOLD = 0.10
GAP_PRESETS = {
    "All areas": None,
    "Undervalued: price at least 10% below the model estimate": "under",
    "Overvalued: price at least 10% above the model estimate": "over",
}

st.title("🗺️ The Neighbourhood Pulse")
st.markdown(
    "**Can independent cafés predict house prices?** Gentrification tends to follow a "
    "script: planning applications pick up, shops turn into flats and cafés, and only "
    "later do prices move. This map reads those early signals for every London "
    "neighbourhood (each hexagon) and asks a model what homes *should* cost given the "
    "activity on the ground and how central the area is. **Red areas sell for less "
    "than the activity suggests**: possible early movers the market has not caught up "
    "with. Green areas sell for more.\n\n"
    "The idea holds up against history: areas this model flagged as underpriced in "
    "2021–22 went on to grow about 9 percentage points more by 2024–25 than the areas "
    "it flagged as overpriced."
)

gap = load_gap()
briefs = load_briefs()
bounds = feature_bounds(gap)

fcol1, fcol2 = st.columns(2)
boroughs = fcol1.multiselect("Borough", sorted(gap["borough"].unique()))
preset = fcol2.selectbox("Show", list(GAP_PRESETS))

view = gap
if boroughs:
    view = view[view["borough"].isin(boroughs)]
if GAP_PRESETS[preset] == "under":
    view = view[view["valuation_gap"] <= -GAP_PRESET_THRESHOLD]
elif GAP_PRESETS[preset] == "over":
    view = view[view["valuation_gap"] >= GAP_PRESET_THRESHOLD]

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
            tooltip={"text": "{borough}: price vs estimate {gap_pct}%"},
        ),
        on_select="rerun",
        selection_mode="single-object",
        height=560,
    )
    st.caption(
        f"{len(view):,} of {len(gap):,} modelled areas shown. Red = selling below the "
        "model estimate, green = above. Click a hexagon for detail. Blank areas are "
        "parks, industrial land, the river, or places with fewer than 30 home sales "
        "in 2021–2025; too few sales means no reliable price to model."
    )

selected_objects = event.selection.objects.get("gap", [])
selected = selected_objects[0] if selected_objects else None

with right:
    st.subheader("Area detail")
    if selected is None:
        st.info("Click a hexagon on the map to see its prices, AI brief, and what-if tool.")
        st.stop()

    row = gap.set_index("h3_index").loc[selected["h3_index"]]
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Median sale price",
        fmt_gbp(row["median_price"]),
        help=f"£{row['median_price']:,.0f}: median of actual sales, 2021–2025",
    )
    c2.metric(
        "Model estimate",
        fmt_gbp(row["pred_price"]),
        help=f"£{row['pred_price']:,.0f}: what the model expects from activity signals",
    )
    c3.metric("Price vs estimate", f"{row['valuation_gap'] * 100:+.1f}%")
    st.caption(
        f"{row['borough']} · {int(row['sales_count'])} sales pooled over 2021–2025 · "
        f"{row['dist_to_centre_km']:.1f} km from Charing Cross"
    )
    st.caption(
        "The estimate is what the model expects from activity signals alone. The model "
        "explains under half of price variation, so a large gap can mean model error "
        "rather than opportunity."
    )

    brief = briefs.get(selected["h3_index"])
    if brief:
        st.markdown("##### ✨ AI brief")
        st.markdown(f"**{brief['headline']}**")
        st.write(brief["brief"])
        st.caption(f"⚠️ {brief['caveat']}")
        st.caption(
            "Written by Gemini once at build time for the 50 most undervalued areas, "
            "then served from a cached file. The app makes no live API calls."
        )
    else:
        st.caption(
            "No AI brief for this area. Briefs are pre-generated for the 50 most "
            "undervalued areas only."
        )

    with st.expander("What if this area changed?"):
        st.caption(
            "Drag the sliders to imagine this area with more or less activity. The "
            "trained model reprices the area live, using the same prediction code as "
            "the project's API. Sliders are limited to the range seen in the real data."
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
                "What-if model estimate",
                fmt_gbp(price),
                delta=f"{(price / row['pred_price'] - 1) * 100:+.1f}% vs current estimate",
                help=f"£{price:,.0f}",
            )
            st.caption(
                f"Price vs estimate would move from {row['valuation_gap'] * 100:+.1f}% "
                f"to {new_gap * 100:+.1f}% (actual sale price held fixed)."
            )
