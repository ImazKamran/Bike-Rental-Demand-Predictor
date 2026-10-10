import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Configuration - Wide layout
st.set_page_config(
    page_title="Velocita | Bike Demand Forecaster",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    .hero-container {
        padding: 1.8rem 2rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        color: #ffffff;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0;
        color: #f9fafb;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #9ca3af;
        margin-top: 0.4rem;
        margin-bottom: 0;
    }

    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6b7280;
    }
    .metric-value {
        font-size: 3.2rem;
        font-weight: 800;
        color: #2563eb;
        line-height: 1.1;
        margin: 0.3rem 0;
    }
    .metric-subvalue {
        font-size: 1.6rem;
        font-weight: 700;
        color: #111827;
        margin-top: 0.2rem;
    }
    .metric-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #eff6ff;
        color: #1d4ed8;
    }
</style>
""", unsafe_allow_html=True)

# 1. Load Assets
@st.cache_resource
def load_assets():
    model = joblib.load('bike_model.pkl')
    scaler = joblib.load('scaler.pkl')  # Remove or adjust if scaler was not used
    return model, scaler

try:
    model, scaler = load_assets()
except Exception:
    st.error("Model assets not found. Ensure 'bike_model.pkl' and 'scaler.pkl' are in the working directory.")
    st.stop()

# 2. Hero Banner Header
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🚲 Velocita Predictive Analytics</div>
    <div class="hero-subtitle">ML-Powered Hourly Fleet Demand Forecasting Engine</div>
</div>
""", unsafe_allow_html=True)

# 3. Sidebar - Operational Context
with st.sidebar:
    st.header("⚙️ Operational Context")
    
    yr = st.selectbox(
        "Operating Scale",
        options=[0, 1],
        format_func=lambda x: "2011 (Baseline Launch)" if x == 0 else "2012 (Established Operations)",
        index=1,
        help="2012 reflects full operational capacity with higher organic adoption."
    )
    
    st.divider()
    st.caption("ℹ️ **Model Architecture**")
    st.caption("Standard OLS Linear Regression with feature engineering (sin/cos temporal encoding & dummy categories) and post-processed zero-clipping.")

# 4. Main Control Dashboard
main_col, result_col = st.columns([1.6, 1], gap="large")

with main_col:
    # Container 1: Time & Schedule
    with st.container(border=True):
        st.markdown("### 📅 Temporal Parameters")
        
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            hr = st.slider("Hour of Day", 0, 23, 17, format="%d:00")
            mnth = st.slider("Month", 1, 12, 6, format="Month %d")
            season = st.select_slider(
                "Season", 
                options=[1, 2, 3, 4],
                format_func=lambda x: {1: "🌸 Spring", 2: "☀️ Summer", 3: "🍂 Fall", 4: "❄️ Winter"}[x],
                value=2
            )
        
        with t_col2:
            weekday = st.selectbox(
                "Day of Week", 
                options=[0, 1, 2, 3, 4, 5, 6],
                format_func=lambda x: ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"][x],
                index=3
            )
            holiday = st.toggle("Holiday", value=False)
            workingday = st.toggle("Working Day", value=True)

    st.write("") # Spacing

    # Container 2: Weather Dynamics & Buffer Configuration
    with st.container(border=True):
        st.markdown("### 🌤️ Environmental & Buffer Controls")
        
        w_col1, w_col2 = st.columns(2)
        with w_col1:
            weathersit = st.selectbox(
                "Weather Category",
                options=[1, 2, 3, 4],
                format_func=lambda x: {
                    1: "☀️ Clear / Few Clouds", 
                    2: "🌫️ Mist / Cloudy", 
                    3: "🌧️ Light Snow / Rain", 
                    4: "⛈️ Heavy Storm / Ice"
                }[x]
            )
            temp_c = st.slider("Temperature (°C)", -8.0, 39.0, 22.0, step=0.5)

        with w_col2:
            hum_pct = st.slider("Humidity (%)", 0, 100, 45)
            wind_kmh = st.slider("Wind Speed (km/h)", 0.0, 67.0, 12.0, step=0.5)

        st.divider()
        # Fleet Buffer Capacity Slider
        buffer_pct = st.slider(
            "Fleet Safety Buffer (%)", 
            min_value=0, 
            max_value=30, 
            value=15, 
            step=5,
            help="Additional fleet capacity added above predicted demand to absorb localized surges."
        )

# Feature Processing & Prediction Execution
temp_norm = (temp_c + 8) / 47
hum_norm = hum_pct / 100
wind_norm = wind_kmh / 67

hr_sin = np.sin(2 * np.pi * hr / 24.0)
hr_cos = np.cos(2 * np.pi * hr / 24.0)
mnth_sin = np.sin(2 * np.pi * mnth / 12.0)
mnth_cos = np.cos(2 * np.pi * mnth / 12.0)

input_df = pd.DataFrame([{
    'yr': int(yr),
    'holiday': int(holiday),
    'workingday': int(workingday),
    'temp': float(temp_norm),
    'hum': float(hum_norm),
    'windspeed': float(wind_norm),
    'hr_sin': float(hr_sin),
    'hr_cos': float(hr_cos),
    'mnth_sin': float(mnth_sin),
    'mnth_cos': float(mnth_cos),
    'season_2': (season == 2),
    'season_3': (season == 3),
    'season_4': (season == 4),
    'weathersit_2': (weathersit == 2),
    'weathersit_3': (weathersit == 3),
    'weathersit_4': (weathersit == 4),
    'weekday_1': (weekday == 1),
    'weekday_2': (weekday == 2),
    'weekday_3': (weekday == 3),
    'weekday_4': (weekday == 4),
    'weekday_5': (weekday == 5),
    'weekday_6': (weekday == 6)
}])

try:
    input_scaled = scaler.transform(input_df)
except Exception:
    input_scaled = input_df

raw_pred = model.predict(input_scaled)[0]
final_pred = int(np.clip(raw_pred, a_min=0, a_max=None))

# Recommended Fleet Calculation (Demand + Buffer)
recommended_fleet = int(final_pred * (1 + (buffer_pct / 100)))

# 5. Live Prediction & Fleet Output Display
with result_col:
    st.markdown("### 📊 Demand & Deployment")
    
    # Primary Metric Card showing Recommended Fleet & Base Demand
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Recommended Fleet Size</div>
        <div class="metric-value">{recommended_fleet:,}</div>
        <div class="metric-badge">Includes {buffer_pct}% Safety Buffer</div>
        <hr style="margin: 1rem 0; border: none; border-top: 1px solid #f3f4f6;">
        <div class="metric-label">Base Predicted Demand</div>
        <div class="metric-subvalue">{final_pred:,} bikes</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("") # Spacing

    with st.container(border=True):
        st.markdown("#### 💡 Operational Insights")
        
        if raw_pred < 0:
            st.warning(f"**Low-Demand Window Detected:** Raw OLS calculation yielded `{raw_pred:.1f}`. Zero-clipping override engaged to maintain physical non-negativity constraint.")
        elif final_pred > 400:
            st.error("🔥 **Peak Demand Surge:** High inventory allocation recommended. Ensure station rebalancing vans are active.")
        elif final_pred > 150:
            st.info("⚡ **Moderate Demand:** Standard fleet distribution required.")
        else:
            st.success("🌙 **Off-Peak Demand:** Ideal window for routine maintenance and station rebalancing.")

    with st.expander("🔍 View Raw Vector Inputs (22 Features)"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)