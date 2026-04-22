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
    except FileNotFoundError:
        st.warning("No data files found. Generating synthetic data...")
        raw = generate_synthetic_cwc(n=8000)
        source = "synthetic"
    except Exception as e:
        st.error(f"Error loading data: {str(e)}. Using synthetic data instead.")
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
            try:
                models[f.replace(".pkl", "")] = joblib.load(path)
            except Exception as e:
                st.warning(f"Could not load model {f}: {str(e)}")
    return models


df, source = load_data()
models = load_models()

# ─── Sidebar ─────────────────────────────────────────────────
# user filters
st.sidebar.title("Filters")

# Initialize session state for filters
if "sel_states" not in st.session_state:
    st.session_state.sel_states = None
if "sel_years" not in st.session_state:
    st.session_state.sel_years = None
if "wqi_range" not in st.session_state:
    st.session_state.wqi_range = (0.0, 100.0)

states = sorted(df["state"].dropna().unique())
sel_states = st.sidebar.multiselect("States", states, default=states, key="state_filter")

if "year" in df.columns:
    yr_min, yr_max = int(df["year"].min()), int(df["year"].max())
    sel_years = st.sidebar.slider("Year", yr_min, yr_max, (yr_min, yr_max), key="year_filter")
else:
    sel_years = None

wqi_range = st.sidebar.slider("WQI", 0.0, 100.0, (0.0, 100.0), key="wqi_filter")


# ─── Filtering ───────────────────────────────────────────────
# applies filters to dataset
@st.cache_data
def apply_filters(df, sel_states, sel_years, wqi_range):
    filt = df.copy()
    
    if sel_states:
        filt = filt[filt["state"].isin(sel_states)]
    
    if sel_years and "year" in filt.columns:
        filt = filt[(filt["year"] >= sel_years[0]) & (filt["year"] <= sel_years[1])]
    
    filt = filt[(filt["WQI"] >= wqi_range[0]) & (filt["WQI"] <= wqi_range[1])]
    
    # clean labels (prevents bugs)
    filt["water_quality"] = filt["water_quality"].astype(str).str.strip()
    
    return filt

filt = apply_filters(df, sel_states, sel_years, wqi_range)

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
    with st.spinner("Generating distribution chart..."):
        wq_counts = filt["water_quality"].value_counts().reset_index()
        wq_counts.columns = ["Category", "Count"]

        fig = px.pie(
            wq_counts,
            names="Category",
            values="Count",
            color="Category",
            color_discrete_map=get_valid_colors(wq_counts, "Category", QUAL_COLORS)
        )
        st.plotly_chart(fig, width='stretch')
    # ─ River Analysis ─

st.markdown("### 🌊 River Water Quality Analysis")

if "River" in filt.columns:

    river_df = filt.dropna(subset=["River", "WQI"])

    if len(river_df) > 0:

        river_summary = river_df.groupby("River").agg(
            mean_WQI=("WQI", "mean"),
            samples=("WQI", "count"),
            safe_pct=("is_safe", lambda x: x.mean() * 100)
        ).reset_index()

        river_summary = river_summary.sort_values("mean_WQI")

        # Risk classification
        river_summary["Risk"] = river_summary["mean_WQI"].apply(
            lambda x: "High Risk" if x > 70 else "Moderate" if x > 50 else "Low Risk"
        )

        st.dataframe(river_summary.head(10))

        fig = px.bar(
            river_summary.head(10),
            x="mean_WQI",
            y="River",
            orientation="h",
            color="mean_WQI",
            color_continuous_scale="RdYlGn_r",
            title="Top Polluted Rivers"
        )

        st.plotly_chart(fig, width='stretch')

    else:
        st.info("No river data after filtering.")

else:
    st.warning("River column not found in dataset.")

    # histogram
    fig = px.histogram(plot_df, x="WQI", nbins=40)
    fig.add_vline(x=50, line_dash="dash", line_color="red")
    st.plotly_chart(fig, width='stretch')


## ─── Heatmap ─────────────────────────────────────────────
with tab2:

    st.markdown("### Water Quality Spatial Analysis")

    if "latitude" in filt.columns:

        # Map configuration - merged controls
        with st.expander("Map Configuration", expanded=True):
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            
            with col1:
                basemap_style = st.selectbox(
                    "Basemap",
                    ["OpenStreetMap", "Carto Positron", "Carto Dark"],
                    index=0,
                    key="basemap_selector"
                )
            
            with col2:
                color_theme = st.selectbox(
                    "Color Theme",
                    ["Green-Yellow-Red", "Blue-Purple-Red", "Viridis", "Plasma", "Inferno"],
                    index=0,
                    key="color_theme_selector"
                )
            
            with col3:
                color_mode = st.selectbox(
                    "Station Color",
                    ["Safe/Unsafe", "WQI Gradient", "Quality Categories"],
                    index=0,
                    key="color_mode_selector"
                )
            
            with col4:
                show_density = st.checkbox(
                    "Show Density",
                    value=True,
                    key="density_toggle"
                )
            
            st.markdown("")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                density_radius = st.slider("Density Radius", 5, 30, 18, key="density_radius")
            
            with col2:
                marker_size = st.slider("Marker Size", 5, 25, 15, key="marker_size")
            
            with col3:
                enable_animation = st.checkbox("Enable Time Animation", value=True, key="animation_toggle")
        
        # Color theme mapping
        color_themes = {
            "Green-Yellow-Red": [[0.0, "#27ae60"], [0.5, "#f1c40f"], [1.0, "#e74c3c"]],
            "Blue-Purple-Red": [[0.0, "#3498db"], [0.5, "#9b59b6"], [1.0, "#e74c3c"]],
            "Viridis": [[0.0, "#440154"], [0.5, "#21918c"], [1.0, "#fde725"]],
            "Plasma": [[0.0, "#0d0887"], [0.5, "#cc4778"], [1.0, "#f0f921"]],
            "Inferno": [[0.0, "#000004"], [0.5, "#b53679"], [1.0, "#fcffa4"]]
        }
        selected_colors = color_themes[color_theme]
        
        # Map style mapping
        basemap_map = {
            "OpenStreetMap": "open-street-map",
            "Carto Positron": "carto-positron",
            "Carto Dark": "carto-darkmatter"
        }
        selected_basemap = basemap_map[basemap_style]

        with st.spinner("Loading spatial data..."):
            map_df = filt.dropna(subset=["latitude", "longitude", "WQI", "water_quality"])

            if len(map_df) == 0:
                st.warning("No valid coordinates found")
            else:
                # Custom animation control using Streamlit slider
                if enable_animation and "year" in map_df.columns:
                    available_years = sorted(map_df["year"].unique())
                    selected_year = st.selectbox("Select Year", available_years, index=len(available_years)-1, key="year_selector")
                    map_df = map_df[map_df["year"] == selected_year]
                    st.info(f"Showing data for year: {selected_year}")
                
                density_df = map_df.sample(min(2000, len(map_df)), random_state=42)
                scatter_df = map_df.sample(min(1000, len(map_df)), random_state=42)

                # Density Map with enhanced styling
                if show_density:
                    with st.spinner("Generating density heatmap..."):
                        fig = px.density_mapbox(
                            density_df,
                            lat="latitude",
                            lon="longitude",
                            z="WQI",
                            radius=density_radius,
                            center={"lat": 20.5, "lon": 80},
                            zoom=4,
                            mapbox_style="carto-positron",
                            color_continuous_scale=selected_colors,
                            title="Water Quality Density Heatmap",
                            labels={"z": "WQI Index"},
                            range_color=[density_df["WQI"].min(), density_df["WQI"].max()],
                            opacity=0.8
                        )
                        fig.update_layout(
                            height=500, 
                            margin=dict(l=0, r=0, t=30, b=0),
                            coloraxis_colorbar=dict(title="WQI", x=1.02, len=0.8)
                        )
                        fig.update_traces(opacity=0.9)
                        st.plotly_chart(fig, width='stretch', config={'scrollZoom': True, 'displayModeBar': True, 'displaylogo': False})

                st.markdown("")

                # Scatter Map with enhanced features
                if color_mode == "Safe/Unsafe":
                    scatter_df["status"] = scatter_df["is_safe"].map({1: "Safe", 0: "Unsafe"})
                    color_column = "status"
                    status_colors = {"Safe": "#27ae60", "Unsafe": "#e74c3c"}
                    color_discrete_map = status_colors
                elif color_mode == "WQI Gradient":
                    color_column = "WQI"
                    color_discrete_map = None
                    status_colors = None
                else:
                    color_column = "water_quality"
                    color_discrete_map = get_valid_colors(scatter_df, "water_quality", QUAL_COLORS)
                    status_colors = None

                with st.spinner("Rendering station map..."):
                    if color_mode == "WQI Gradient":
                        fig2 = px.scatter_mapbox(
                            scatter_df,
                            lat="latitude",
                            lon="longitude",
                            color="WQI",
                            size="WQI",
                            size_max=marker_size,
                            center={"lat": 20.5, "lon": 80},
                            zoom=4,
                            mapbox_style=selected_basemap,
                            title="Water Quality Monitoring Network",
                            color_continuous_scale=selected_colors,
                            hover_data={"WQI": ":.2f", "water_quality": True, "state": True}
                        )
                    else:
                        fig2 = px.scatter_mapbox(
                            scatter_df,
                            lat="latitude",
                            lon="longitude",
                            color=color_column,
                            color_discrete_map=color_discrete_map,
                            size="WQI",
                            size_max=marker_size,
                            center={"lat": 20.5, "lon": 80},
                            zoom=4,
                            mapbox_style=selected_basemap,
                            title="Water Quality Monitoring Network",
                            hover_data={"WQI": ":.2f", "water_quality": True, "state": True}
                        )

                    fig2.update_traces(marker=dict(opacity=0.85), selector=dict(mode='markers'))
                    fig2.update_layout(
                        height=500,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    st.plotly_chart(fig2, width='stretch', config={'scrollZoom': True, 'displayModeBar': True, 'displaylogo': False})

                # Enhanced statistics with visual indicators
                st.markdown("---")
                st.markdown("### Spatial Analytics Dashboard")
                
                col1, col2, col3, col4 = st.columns(4)
                safe_pct = (len(map_df[map_df["is_safe"] == 1]) / len(map_df)) * 100
                
                col1.metric("Total Stations", len(map_df), delta=f"{len(map_df)} locations")
                col2.metric("Safe Stations", len(map_df[map_df["is_safe"] == 1]), delta=f"{safe_pct:.1f}%")
                col3.metric("Unsafe Stations", len(map_df[map_df["is_safe"] == 0]), delta=f"{100-safe_pct:.1f}%")
                col4.metric("Mean WQI", round(map_df["WQI"].mean(), 2), delta=f"Range: {map_df['WQI'].min():.1f}-{map_df['WQI'].max():.1f}")

                # Export with options
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                csv_data = map_df.to_csv(index=False)
                col1.download_button(
                    "Download Dataset (CSV)",
                    data=csv_data,
                    file_name=f"water_quality_spatial_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="export_csv"
                )
                
                col2.markdown(f"**Data Summary:** {len(map_df)} stations across {map_df['state'].nunique()} states")

                # Advanced geospatial analysis
                st.markdown("---")
                st.markdown("### Risk Assessment")
                
                if len(map_df) > 10:
                    with st.spinner("Performing spatial analysis..."):
                        unsafe_df = map_df[map_df["is_safe"] == 0]
                        
                        if len(unsafe_df) > 0:
                            tab_a, tab_b = st.tabs(["High-Risk Areas", "Quality Distribution"])
                            
                            with tab_a:
                                st.markdown("#### States with Most Unsafe Stations")
                                state_risk = unsafe_df.groupby("state").agg(
                                    unsafe_count=("is_safe", "count"),
                                    avg_wqi=("WQI", "mean"),
                                    max_wqi=("WQI", "max"),
                                    total_stations=("state", "count")
                                ).sort_values("unsafe_count", ascending=False).head(10)
                                state_risk.columns = ["Unsafe Count", "Avg WQI", "Max WQI", "Total Stations"]
                                st.dataframe(state_risk, width='stretch')
                                
                                # Risk level visualization
                                st.markdown("#### Risk Level Heatmap")
                                risk_heatmap = state_risk[["Unsafe Count", "Avg WQI"]]
                                st.dataframe(risk_heatmap, width='stretch')
                            
                            with tab_b:
                                st.markdown("#### Water Quality Distribution")
                                risk_levels = map_df["water_quality"].value_counts()
                                
                                fig_dist = px.pie(
                                    values=risk_levels.values,
                                    names=risk_levels.index,
                                    title="Quality Category Distribution",
                                    hole=0.4
                                )
                                fig_dist.update_traces(textposition='inside', textinfo='percent+label')
                                fig_dist.update_layout(height=400)
                                st.plotly_chart(fig_dist, width='stretch', config={'displayModeBar': True, 'displaylogo': False})
                        else:
                            st.success("✅ All stations meet safety standards in current view")
                else:
                    st.warning("⚠️ Insufficient data for comprehensive analysis")

# ─── Trends ─────────────────────────────────────────────
with tab3:

    st.markdown("### Trends")

    if "year" not in filt.columns:
        st.warning("No year column in dataset")
    else:
        with st.spinner("Generating trend analysis..."):
            trend = filt.groupby("year")["WQI"].mean().reset_index()

            if len(trend) == 0:
                st.warning("No data after filtering")
            else:
                # Animation toggle
                show_animation = st.checkbox("Show Animation", value=False, key="trend_animation")
                
                if show_animation and len(trend) > 2:
                    fig = px.line(
                        trend, 
                        x="year", 
                        y="WQI", 
                        markers=True,
                        animation_frame="year",
                        range_y=[trend["WQI"].min() - 5, trend["WQI"].max() + 5]
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, width='stretch')
                else:
                    fig = px.line(trend, x="year", y="WQI", markers=True)
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, width='stretch')
                
                # Trend statistics
                st.markdown("### 📈 Trend Statistics")
                col1, col2, col3 = st.columns(3)
                col1.metric("Years Analyzed", len(trend))
                col2.metric("Avg WQI", round(trend["WQI"].mean(), 2))
                
                # Calculate trend direction
                if len(trend) >= 2:
                    first_year = trend.iloc[0]["WQI"]
                    last_year = trend.iloc[-1]["WQI"]
                    trend_change = last_year - first_year
                    trend_direction = "Improving ↓" if trend_change < 0 else "Deteriorating ↑"
                    col3.metric("Trend", trend_direction)
# ════════════════════════════════════════════════════════════
# TAB 4 — ANALYSIS
# ════════════════════════════════════════════════════════════
with tab4:

    avail = [f for f in CORE_FEATURES if f in filt.columns]

    if len(avail) >= 2:

        param_a = st.selectbox("Parameter A", avail)
        param_b = st.selectbox("Parameter B", avail, index=1)

        with st.spinner("Generating scatter plot..."):
            scatter_df = filt[[param_a, param_b, "water_quality"]].dropna()
            scatter_df = scatter_df.sample(min(1500, len(scatter_df)))

            fig = px.scatter(
                scatter_df,
                x=param_a,
                y=param_b,
                color="water_quality",
                color_discrete_map=get_valid_colors(scatter_df, "water_quality", QUAL_COLORS)
            )
            st.plotly_chart(fig, width='stretch')


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
            st.dataframe(summary_df)

    else:
        st.warning("Run model_dev.py first")


# ════════════════════════════════════════════════════════════
# TAB 6 — UPLOAD
# ════════════════════════════════════════════════════════════
with tab6:

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:

        with st.spinner("Processing uploaded file..."):
            new_df = pd.read_csv(uploaded)

            st.write(new_df.head())

            fig = px.histogram(new_df, x="WQI")
            st.plotly_chart(fig, width='stretch')
        
