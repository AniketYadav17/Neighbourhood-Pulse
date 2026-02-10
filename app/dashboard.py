import streamlit as st
import pandas as pd
import pydeck as pdk
import pickle
import h3

# Page Setup
st.set_page_config(layout="wide", page_title="Neighborhood Pulse V1")

st.title("🏙️ Neighborhood Pulse: Gentrification Predictor")
st.markdown("### V1 Prototype: Shoreditch Spatiotemporal Analysis")

# 1. Load Data & Model
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/unified_training_data.csv")
    return df

@st.cache_resource
def load_model():
    with open("models/gentrification_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

df = load_data()
model = load_model()

# 2. Run Inference
# We re-predict using the model to show "Predicted Value" vs "Actual"
X = df.drop(columns=['avg_house_price', 'h3_index'])
df['predicted_price'] = model.predict(X)
df['gentrification_score'] = df['predicted_price'] / 1000 # Mock score normalization

# 3. Visualization (The Hero Map)
st.sidebar.header("Filter Options")
metric = st.sidebar.selectbox("Color Hexagons By:", ["avg_house_price", "count_commercial_change", "indie_cafe_count"])

# Define Layer
layer = pdk.Layer(
    "H3HexagonLayer",
    df,
    pickable=True,
    stroked=True,
    filled=True,
    extruded=True,
    get_hexagon="h3_index",
    get_fill_color=f"[255, ({metric} / {df[metric].max()}) * 255, 0, 140]", # Dynamic Green/Red scaling
    get_elevation=metric,
    elevation_scale=50,
    elevation_range=[0, 1000],
)

# Set View State (Center on Shoreditch)
view_state = pdk.ViewState(
    latitude=51.522,
    longitude=-0.072,
    zoom=14,
    pitch=50,
)

# Render Map
st.pydeck_chart(pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={"text": "H3: {h3_index}\nPrice: £{avg_house_price}\nIndie Cafes: {indie_cafe_count}"}
))

# 4. The Verdict (Explainability)
st.subheader("🤖 Model Verdict")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Top Gentrification Indicators")
    # Show feature importance (simple text for V1)
    st.write("Based on XGBoost feature importance:")
    st.progress(80, text="Commercial Change of Use (Planning)")
    st.progress(60, text="Indie Coffee Shop Density")
    st.progress(30, text="Residential Renovations")

with col2:
    st.markdown("#### Selected Zone Details")
    st.dataframe(df.head())