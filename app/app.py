"""Neighbourhood Pulse — Streamlit app v2 (entry point).

Three pages over the committed serving artifacts (artifacts/): Explore (pydeck
hexagon map + detail + what-if), Model (metrics.json), Methodology. No model
training, no API calls at runtime — this is the packaging layer over frozen
research; the science lives in the pipeline and the notebook.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(page_title="Neighbourhood Pulse", page_icon="🏙️", layout="wide")

_GAP = Path(__file__).resolve().parent.parent / "artifacts" / "hex_valuation_gap.parquet"
if not _GAP.exists():
    st.error(
        "Missing `artifacts/hex_valuation_gap.parquet`. Run `pulse train` "
        "(see README quickstart) and restart the app."
    )
    st.stop()

pg = st.navigation(
    [
        st.Page("views/explore.py", title="Explore", icon="🗺️", default=True),
        st.Page("views/model.py", title="Model", icon="📈"),
        st.Page("views/methodology.py", title="Methodology", icon="🧭"),
    ]
)
pg.run()
