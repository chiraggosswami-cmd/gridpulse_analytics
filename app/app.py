import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="GridPulse Analytics Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom header styling
st.markdown("""
    <div style="background-color:#0E1117; padding:20px; border-radius:10px; border-bottom: 3px solid #FF4B4B; margin-bottom:30px;">
        <h1 style="color:white; margin:0;">⚡ GridPulse Dashboard</h1>
        <p style="color:#FAFAFA; margin:5px 0 0 0; opacity:0.7;">Production Smart Grid Anomaly Detection System</p>
    </div>
""", unsafe_allow_html=True)

# --- 2. ASSET LOADER (WITH ADVANCED PERFORMANCE CACHING) ---
@st.cache_resource
def load_production_artifacts():
    """
    Optimized loader that checks root or sibling folders for model files 
    and caches them into memory so your app doesn't slow down on re-renders.
    """
    model_paths = ["../models/xgb_anomaly_model.pkl", "models/xgb_anomaly_model.pkl", "xgb_anomaly_model.pkl"]
    lookup_paths = ["../models/user_lookup.pkl", "models/user_lookup.pkl", "user_lookup.pkl"]
    
    model, lookup = None, None
    
    for p in model_paths:
        if os.path.exists(p):
            with open(p, 'rb') as f: model = pickle.load(f)
            break
            
    for p in lookup_paths:
        if os.path.exists(p):
            with open(p, 'rb') as f: lookup = pickle.load(f)
            break
            
    return model, lookup

model, user_lookup = load_production_artifacts()

# Error handling placeholder if models aren't trained yet
if model is None or user_lookup is None:
    st.error("⚠️ Pipeline artifacts missing. Please train your model using 'src/models.py' first to generate files.")
    st.stop()

# --- 3. SIDEBAR INTERACTION COMMAND CENTER ---
st.sidebar.title("🎮 Control Console")
st.sidebar.markdown("---")

# Population of actual User ID values
user_list = user_lookup['UserId'].unique()
selected_user = st.sidebar.selectbox("🎯 Select Target User Profile", user_list)

st.sidebar.markdown("### 📊 Live Grid Injection Input")
input_consumption = st.sidebar.slider(
    "Today's Energy Meter Reading (kWh)", 
    min_value=0.0, max_value=150.0, value=25.0, step=0.1
)

calendar_context = st.sidebar.radio("📅 Temporal Frame", ["Weekday", "Weekend"], horizontal=True)
is_weekend_numeric = 1 if calendar_context == "Weekend" else 0
day_of_week_dummy = 5 if is_weekend_numeric == 1 else 1

# --- 4. DATA ENGINE AND HISTORICAL RECONSTRUCTION ---
# Pull user baseline metrics
user_historical_mean = user_lookup[user_lookup['UserId'] == selected_user]['UserMean'].values[0]
diff_from_mean = input_consumption - user_historical_mean

# Dynamic trailing time-series generation for line plotting
np.random.seed(42)  # Consistent mock generation per user profile seed
historical_days = pd.date_range(end=pd.Timestamp.now(), periods=14)
simulated_stream = np.random.normal(loc=user_historical_mean, scale=max(1.0, user_historical_mean*0.12), size=14)
simulated_stream[-1] = input_consumption  # Inject current active slider input into latest day

chart_df = pd.DataFrame({'Date': historical_days, 'Consumption (kWh)': simulated_stream}).set_index('Date')

# Compute short term rolling window features manually for inference
rolling_mean_7d = np.mean(simulated_stream[-7:])
diff_from_rolling = input_consumption - rolling_mean_7d

# Package exact feature matrix required by the model
features = np.array([[
    input_consumption, day_of_week_dummy, is_weekend_numeric,
    user_historical_mean, diff_from_mean, rolling_mean_7d, diff_from_rolling
]])

# --- 5. MAIN MONITORING INTERFACE LAYOUT ---
metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

with metrics_col1:
    st.metric(label="👤 Historical Mean Baseline", value=f"{user_historical_mean:.2f} kWh")
with metrics_col2:
    st.metric(label="⚡ Current Active Injected Value", value=f"{input_consumption:.2f} kWh")
with metrics_col3:
    st.metric(
        label="🔄 Footprint Shift (Delta)", 
        value=f"{diff_from_mean:+.2f} kWh",
        delta=f"{diff_from_mean:.2f} kWh", 
        delta_color="inverse"
    )

st.markdown("---")

left_panel, right_panel = st.columns([2, 1])

with left_panel:
    st.subheader("📊 14-Day Trailing Footprint Stream")
    # Dynamic line chart updates instantly when user adjusts the sidebar slider!
    st.line_chart(chart_df, color="#FF4B4B" if abs(diff_from_mean) > 25 else "#29B5E8")

with right_panel:
    st.subheader("🤖 ML Inference Engine")
    st.markdown("Run live diagnostic scan against the trained XGBoost model:")
    
    if st.button("🚀 Run Anomaly Diagnosis Scan", use_container_width=True):
        with st.spinner("Analyzing profile structures..."):
            time.sleep(0.3)  # Simulates production processing roundtrip latency
            
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            risk_score = probabilities[1] * 100
            
        if prediction == 1:
            st.markdown(f"""
                <div style="background-color:#FFEBEB; border-left: 6px solid #FF4B4B; padding:15px; border-radius:5px;">
                    <h4 style="color:#FF4B4B; margin:0 0 5px 0;">🚨 HIGH ANOMALY RISK</h4>
                    <p style="color:#1E1E1E; margin:0; font-size:14px;">
                        Theft Probability: <b>{risk_score:.1f}%</b><br>
                        Current footprint indicates sharp pattern deviations. Flagged for field audit inspection.
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="background-color:#EBFBEE; border-left: 6px solid #28A745; padding:15px; border-radius:5px;">
                    <h4 style="color:#28A745; margin:0 0 5px 0;">✅ BEHAVIOR NOMINAL</h4>
                    <p style="color:#1E1E1E; margin:0; font-size:14px;">
                        Normal Confidence: <b>{probabilities[0]*100:.1f}%</b><br>
                        Footprint signatures remain securely within expected distribution tracks.
                    </p>
                </div>
            """, unsafe_allow_html=True)