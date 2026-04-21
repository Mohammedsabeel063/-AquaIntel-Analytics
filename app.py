import os, sys, warnings, joblib
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import (
    load_all_csvs, generate_synthetic_cwc, preprocess,
    CORE_FEATURES, BIS_STANDARDS
)

from utils.model_utils import SoftVotingHybrid

# ─── Page config ─────────────────────────────────────────────
# sets layout + theme
st.set_page_config(
    page_title="AquaIntel Analytics",
    page_icon="💧",
    layout="wide",
)

# ─── Color mapping ───────────────────────────────────────────
# base colors for water quality categories
QUAL_COLORS = {
    "Excellent": "#27ae60",
    "Good":      "#f1c40f",
    "Poor":      "#e67e22",
    "Very Poor": "#e74c3c",
}

# returns only colors present in dataset
def get_valid_colors(df, column, color_map):
    present = df[column].dropna().astype(str).str.strip().unique()
    return {k: v for k, v in color_map.items() if k in present}


# ─── Data load ───────────────────────────────────────────────
# loads dataset (real or synthetic)
@st.cache_data
def load_data():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    try:
        raw = load_all_csvs(data_dir)
        source = "real"
    except:
        raw = generate_synthetic_cwc(n=8000)
        source = "synthetic"
    return preprocess(raw), source


# loads models once
@st.cache_resource
def load_models():
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    models = {}
    for f in ["rf_full.pkl", "xgb_full.pkl", "hybrid_soft.pkl"]:
        path = os.path.join(model_dir, f)
        if os.path.exists(path):
            models[f.replace(".pkl", "")] = joblib.load(path)
    return models


df, source = load_data()
models = load_models()

# ─── Sidebar ─────────────────────────────────────────────────
# user filters
st.sidebar.title("Filters")

states = sorted(df["state"].dropna().unique())
sel_states = st.sidebar.multiselect("States", states, default=states)

if "year" in df.columns:
    yr_min, yr_max = int(df["year"].min()), int(df["year"].max())
    sel_years = st.sidebar.slider("Year", yr_min, yr_max, (yr_min, yr_max))
else:
    sel_years = None

wqi_range = st.sidebar.slider("WQI", 0.0, 100.0, (0.0, 100.0))


# ─── Filtering ───────────────────────────────────────────────
# applies filters to dataset
filt = df.copy()

if sel_states:
    filt = filt[filt["state"].isin(sel_states)]

if sel_years and "year" in filt.columns:
    filt = filt[(filt["year"] >= sel_years[0]) & (filt["year"] <= sel_years[1])]

filt = filt[(filt["WQI"] >= wqi_range[0]) & (filt["WQI"] <= wqi_range[1])]

# clean labels (prevents bugs)
filt["water_quality"] = filt["water_quality"].astype(str).str.strip()

# downsample for performance
plot_df = filt.sample(min(3000, len(filt)), random_state=42)


# ─── Header ─────────────────────────────────────────────────
st.title("💧 AquaIntel Analytics")
st.markdown(f"**{len(filt):,} records** · Source: {source}")


# ─── Tabs ───────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "Risk Heatmap",
    "Trends",
    "Analysis",
    "Predict",
    "Upload"
])


# ════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════
with tab1:

    col1, col2, col3 = st.columns(3)

    col1.metric("Records", len(filt))
    col2.metric("States", filt["state"].nunique())
    col3.metric("Mean WQI", round(filt["WQI"].mean(), 2))

    st.markdown("### Distribution")

    # pie chart
    wq_counts = filt["water_quality"].value_counts().reset_index()
    wq_counts.columns = ["Category", "Count"]

    fig = px.pie(
        wq_counts,
        names="Category",
        values="Count",
        color="Category",
        color_discrete_map=get_valid_colors(wq_counts, "Category", QUAL_COLORS)
    )
    st.plotly_chart(fig, use_container_width=True)

    # histogram
    fig = px.histogram(plot_df, x="WQI", nbins=40)
    fig.add_vline(x=50, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)


## ─── Heatmap ─────────────────────────────────────────────
with tab2:

    st.markdown("### Risk Heatmap")

    if "latitude" in filt.columns:

        map_df = filt.dropna(subset=["latitude", "longitude", "WQI"])

        if len(map_df) == 0:
            st.warning("No valid coordinates after filtering")
        else:
            # separate samples for smoother rendering
            density_df = map_df.sample(min(3000, len(map_df)), random_state=42)
            scatter_df = map_df.sample(min(1500, len(map_df)), random_state=42)

            # ─── Density Map (softer colors)
            fig = px.density_mapbox(
                density_df,
                lat="latitude",
                lon="longitude",
                z="WQI",
                radius=18,
                center={"lat": 20.5, "lon": 80},
                zoom=4,
                mapbox_style="carto-positron",
                color_continuous_scale=[
                    [0.0, "#e3f2fd"],
                    [0.3, "#90caf9"],
                    [0.6, "#42a5f5"],
                    [1.0, "#1e88e5"]
                ]
            )

            fig.update_layout(height=550)
            st.plotly_chart(fig, use_container_width=True)

            # spacing so charts don’t visually overlap
            st.markdown("---")

            # ─── Scatter Map (stations clearly visible)
            st.markdown("### Stations")

            color_map = get_valid_colors(scatter_df, "water_quality", QUAL_COLORS)

            fig2 = px.scatter_mapbox(
                scatter_df,
                lat="latitude",
                lon="longitude",
                color="water_quality",
                color_discrete_map=color_map,
                size="WQI",
                size_max=12,
                center={"lat": 20.5, "lon": 80},
                zoom=4,
            )

            # improve marker visibility
            fig2.update_traces(
                marker=dict(
                opacity=0.9
                )
            )

            fig2.update_layout(height=520)

            st.plotly_chart(fig2, use_container_width=True)

# ─── Trends ─────────────────────────────────────────────
with tab3:

    st.markdown("### Trends")

    if "year" not in filt.columns:
        st.warning("No year column in dataset")
    else:
        trend = filt.groupby("year")["WQI"].mean().reset_index()

        if len(trend) == 0:
            st.warning("No data after filtering")
        else:
            fig = px.line(trend, x="year", y="WQI", markers=True)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
# ════════════════════════════════════════════════════════════
# TAB 4 — ANALYSIS
# ════════════════════════════════════════════════════════════
with tab4:

    avail = [f for f in CORE_FEATURES if f in filt.columns]

    if len(avail) >= 2:

        param_a = st.selectbox("Parameter A", avail)
        param_b = st.selectbox("Parameter B", avail, index=1)

        scatter_df = filt[[param_a, param_b, "water_quality"]].dropna()
        scatter_df = scatter_df.sample(min(2000, len(scatter_df)))

        fig = px.scatter(
            scatter_df,
            x=param_a,
            y=param_b,
            color="water_quality",
            color_discrete_map=get_valid_colors(scatter_df, "water_quality", QUAL_COLORS)
        )
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════
# TAB 5 — ML
# ════════════════════════════════════════════════════════════
with tab5:

    if models:

        st.markdown("### Water Quality Prediction")
        
        # Get features from RF model
        model_rf = models["rf_full"]["model"]
        features = models["rf_full"]["features"]

        inputs = {}

        cols = st.columns(3)

        for i, feat in enumerate(features):
            inputs[feat] = cols[i % 3].number_input(feat, value=0.0)

        if st.button("Predict"):
            input_df = pd.DataFrame([inputs])
            
            # Initialize prediction variables
            rf_pred = None
            xgb_pred = None
            hybrid_pred = None
            
            col1, col2, col3 = st.columns(3)
            
            # RF Full (Main)
            with col1:
                rf_pred = model_rf.predict(input_df)[0]
                if rf_pred == 1:
                    st.success("✅ SAFE (RF)")
                else:
                    st.error("⚠️ UNSAFE (RF)")
            
            # XGB Full
            if "xgb_full" in models:
                with col2:
                    model_xgb = models["xgb_full"]["model"]
                    xgb_pred = model_xgb.predict(input_df)[0]
                    if xgb_pred == 1:
                        st.success("✅ SAFE (XGB)")
                    else:
                        st.error("⚠️ UNSAFE (XGB)")
            
            # Soft Hybrid
            if "hybrid_soft" in models:
                with col3:
                    model_hybrid = models["hybrid_soft"]["model"]
                    hybrid_pred = model_hybrid.predict(input_df)[0]
                    if hybrid_pred == 1:
                        st.success("✅ SAFE (Hybrid)")
                    else:
                        st.error("⚠️ UNSAFE (Hybrid)")
            
            # Summary
            st.markdown("---")
            st.markdown("**Model Predictions Summary:**")
            pred_summary = {
                "RF (Main)": "Safe" if rf_pred == 1 else "Unsafe",
            }
            if xgb_pred is not None:
                pred_summary["XGB"] = "Safe" if xgb_pred == 1 else "Unsafe"
            if hybrid_pred is not None:
                pred_summary["Soft Hybrid"] = "Safe" if hybrid_pred == 1 else "Unsafe"
            
            summary_df = pd.DataFrame(list(pred_summary.items()), columns=["Model", "Prediction"])
            st.dataframe(summary_df, use_container_width=True)

    else:
        st.warning("Run model_dev.py first")


# ════════════════════════════════════════════════════════════
# TAB 6 — UPLOAD
# ════════════════════════════════════════════════════════════
with tab6:

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:

        new_df = pd.read_csv(uploaded)

        st.write(new_df.head())

        fig = px.histogram(new_df, x="WQI")
        st.plotly_chart(fig, use_container_width=True)