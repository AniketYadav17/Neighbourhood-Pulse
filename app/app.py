"""Neighbourhood Pulse — Streamlit viewer.

A THIN packaging layer over the trained valuation-gap model: it loads the single
precomputed artifact (`artifacts/hex_valuation_gap.parquet`), renders the
gap map and the most-undervalued table, and frames the result honestly. No model
training, no recomputation — the data science lives in the notebook; this is the
viewer a recruiter clicks.
"""

from pathlib import Path

import branca.colormap as cm
import folium
import h3
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

REPO = Path(__file__).resolve().parent.parent
GAP_PATH = REPO / "artifacts" / "hex_valuation_gap.parquet"

st.set_page_config(page_title="Neighbourhood Pulse", page_icon="🏙️", layout="wide")


@st.cache_data
def load_gap() -> pd.DataFrame:
    return pd.read_parquet(GAP_PATH)


@st.cache_data
def render_map_html() -> str:
    """Build the valuation-gap choropleth once and return it as embeddable HTML."""
    gap = load_gap()
    q = float(gap["valuation_gap"].abs().quantile(0.90))  # symmetric diverging scale
    cmap = cm.LinearColormap(
        ["#d73027", "#ffffbf", "#1a9850"],
        vmin=-q,
        vmax=q,
        caption="valuation gap   (red = undervalued, green = overvalued)",
    )
    m = folium.Map(location=[51.51, -0.10], zoom_start=10, tiles="cartodbpositron")
    for r in gap.itertuples():
        v = max(-q, min(q, r.valuation_gap))
        boundary = [list(p) for p in h3.cell_to_boundary(r.h3_index)]
        folium.Polygon(
            locations=boundary,
            weight=0,
            fill=True,
            fill_color=cmap(v),
            fill_opacity=0.65,
            tooltip=(
                f"{r.borough}: gap {r.valuation_gap * 100:.0f}%  |  "
                f"actual £{r.median_price:,.0f}  |  predicted £{r.pred_price:,.0f}"
            ),
        ).add_to(m)
    m.add_child(cmap)
    return m.get_root().render()


gap = load_gap()

st.title("🏙️ The Neighbourhood Pulse")
st.markdown(
    "**A gentrification predictor for London.** Each hexagon's price is modelled from "
    "*pre-gentrification signals* — planning applications, change-of-use conversions, and "
    "independent-café density — plus a centrality control. The **valuation gap** is the "
    "difference between a hexagon's actual median sale price and the price its signals predict. "
    "Hexagons priced **below** their signal-implied level (red) are *candidate undervalued* "
    "neighbourhoods — places showing the early markers of change before prices have caught up."
)

left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("Valuation-gap map")
    components.html(render_map_html(), height=560)

with right:
    st.subheader("Most undervalued neighbourhoods")
    top = gap.nsmallest(15, "valuation_gap")[
        ["borough", "median_price", "pred_price", "valuation_gap"]
    ].reset_index(drop=True)
    top["valuation_gap"] = (top["valuation_gap"] * 100).round(1)
    st.dataframe(
        top.rename(
            columns={
                "borough": "Borough",
                "median_price": "Actual median £",
                "pred_price": "Predicted £",
                "valuation_gap": "Gap %",
            }
        ),
        column_config={
            "Actual median £": st.column_config.NumberColumn(format="£%d"),
            "Predicted £": st.column_config.NumberColumn(format="£%d"),
            "Gap %": st.column_config.NumberColumn(format="%.1f%%"),
        },
        hide_index=True,
        use_container_width=True,
    )
    st.caption(f"Modelled on {len(gap):,} hexagons with ≥30 pooled sales (2021–2025).")

with st.expander("How to read this — and the honest caveats"):
    st.markdown(
        "- **Model:** XGBoost predicting `log(median price)` from planning + café signals + "
        "distance-to-centre (held-out R² ≈ 0.44). The gap is computed out-of-fold.\n"
        "- **Validated:** a back-test shows hexagons flagged undervalued *early* (2021–22) "
        "subsequently grew more — monotonically from **+7.3%** (most undervalued quintile) to "
        "**−1.5%** (most overvalued). The signal precedes growth.\n"
        "- **`change_of_use_ratio` is the #1 predictor** even after controlling for centrality — "
        "the gentrification signal carries information independent of location.\n"
        "- **Caveat — location confound is reduced, not eliminated:** a single distance-to-centre "
        "can't fully model London's polycentric, transport-driven price surface, so some outer "
        "boroughs still appear undervalued partly due to location. The gap is a **candidate** "
        "signal, not a buy recommendation.\n"
        "- **Caveat — coverage:** the modelled set is hexagons with sufficient planning activity "
        "*and* sales density (~66% of London hexagons), not the whole city."
    )
