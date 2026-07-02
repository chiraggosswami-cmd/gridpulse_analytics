import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time

# --- 1. CONFIGURATION & DARK PREMIUM THEME INJECTION ---
st.set_page_config(
    page_title="GridPulse Analytics Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS injection for cards, neon borders, and clean fonts
st.markdown("""
    <style>
        /* Base page background tweaks if needed */
        .main {
            background-color: #0E1117;
        }
        
        /* Premium Container Cards */
        .metric-card {
            background: rgba(17, 22, 34, 0.85);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 15px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        }
        
        .metric-label {
            color: #8A94A6;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 6px;
        }
        
        .metric-value {
            color: #FFFFFF;
            font-size: 1.8rem;
            font-weight: 700;
        }
        
        /* Status Notification Banners */
        .status-box {
            border-radius: 12px;
            padding: 24px;
            margin-top: 15px;
            box-shadow: 0 4px 25px rgba(0,0,0,0.3);
        }
    </style>
""", unsafe_allow_html=True)

# Header Section with a sleek glow accent line
st.markdown("""
    <div style="padding: 10px 0px 20px 0px; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 30px;">
        <span style="background-color: rgba(255, 75, 75, 0.1); color: #FF4B4B; padding: 6px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.5px;">LIVE GRID SYSTEM</span>
        <h1 style="color: white; margin: 10px 0 5px 0; font-size: 2.2rem; font-weight: 700;">⚡ GridPulse Analytics</h1>
        <p style="color: #8A94A6; margin: 0; font-size: 0.95rem;">Smart Grid Machine Learning Anomaly Detection Interface</p>
    </div>
""", unsafe_allow_html=True)

# --- 2. CACHED ASSET LOADER ---
@st.cache_resource
def load_production_artifacts():
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

if model is None or user_lookup is None:
    st.error("⚠️ Pipeline artifacts missing. Run 'src/models.py' locally first to build your model weights.")
    st.stop()

# --- 3. SIDEBAR NAVIGATION & CONTROL HUB ---
st.sidebar.markdown("""
    <div style='padding-bottom: 10px;'>
        <h3 style='margin:0; color:white;'>🎯 Control Console</h3>
        <p style='margin:0; color:#8A94A6; font-size:0.8rem;'>Adjust input vectors live</p>
    </div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")

user_list = user_lookup['UserId'].unique()
selected_user = st.sidebar.selectbox("👤 Select Target Profile ID", user_list)

st.sidebar.markdown("<br><b>📊 Dynamic Input Stream</b>", unsafe_allow_html=True)
input_consumption = st.sidebar.slider(
    "Active Meter Reading (kWh)", 
    min_value=0.0, max_value=150.0, value=25.0, step=0.1
)

calendar_context = st.sidebar.radio("📅 Temporal Window", ["Weekday", "Weekend"], horizontal=True)
is_weekend_numeric = 1 if calendar_context == "Weekend" else 0
day_of_week_dummy = 5 if is_weekend_numeric == 1 else 1

# --- 4. CALCULATION ENGINE ---
user_historical_mean = user_lookup[user_lookup['UserId'] == selected_user]['UserMean'].values[0]
diff_from_mean = input_consumption - user_historical_mean

# Build chronological time stream
np.random.seed(42)
historical_days = pd.date_range(end=pd.Timestamp.now(), periods=14)
simulated_stream = np.random.normal(loc=user_historical_mean, scale=max(1.0, user_historical_mean*0.12), size=14)
simulated_stream[-1] = input_consumption
chart_df = pd.DataFrame({'Consumption (kWh)': simulated_stream}, index=historical_days)

rolling_mean_7d = np.mean(simulated_stream[-7:])
diff_from_rolling = input_consumption - rolling_mean_7d

features = np.array([[
    input_consumption, day_of_week_dummy, is_weekend_numeric,
    user_historical_mean, diff_from_mean, rolling_mean_7d, diff_from_rolling
]])

# --- 5. GRID OVERVIEW PANEL ---
metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

with metrics_col1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">👤 Historical Baseline</div>
            <div class="metric-value">{user_historical_mean:.2f} <span style="font-size:1rem; color:#8A94A6;">kWh</span></div>
        </div>
    """, unsafe_allow_html=True)

with metrics_col2:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">⚡ Active Injected Load</div>
            <div class="metric-value" style="color: #00F0FF;">{input_consumption:.2f} <span style="font-size:1rem; color:#8A94A6;">kWh</span></div>
        </div>
    """, unsafe_allow_html=True)

with metrics_col3:
    delta_color = "#FF4B4B" if diff_from_mean > 0 else "#00F0FF"
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🔄 Current Divergence</div>
            <div class="metric-value" style="color: {delta_color};">{diff_from_mean:+.2f} <span style="font-size:1rem; color:#8A94A6;">kWh</span></div>
        </div>
    """, unsafe_allow_html=True)

# --- 6. CHARTING & ML INTERACTION ---
left_panel, right_panel = st.columns([2, 1])

with left_panel:
    st.markdown("<h4 style='color:white; margin-bottom:15px;'>📈 14-Day Trailing Footprint Log</h4>", unsafe_allow_html=True)
    # Swaps chart color to match the dashboard's glowing palette dynamically
    chart_color = "#FF4B4B" if abs(diff_from_mean) > 20 else "#00F0FF"
    st.line_chart(chart_df, color=chart_color, height=280)

with right_panel:
    st.markdown("<h4 style='color:white; margin-bottom:15px;'>🤖 Core Inference Engine</h4>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8A94A6; font-size:0.9rem;'>Trigger out-of-sample real-time distribution mapping:</p>", unsafe_allow_html=True)
    
    if st.button("🚀 Run Anomaly Diagnosis Scan", use_container_width=True):
        with st.spinner("Analyzing parameters..."):
            time.sleep(0.3)
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            risk_score = probabilities[1] * 100
            
        if prediction == 1:
            st.markdown(f"""
                <div class="status-box" style="background: rgba(255, 75, 75, 0.08); border: 1px solid #FF4B4B;">
                    <h4 style="color:#FF4B4B; margin:0 0 8px 0; font-weight:700;">🚨 HIGH ANOMALY RISK</h4>
                    <p style="color:#E2E8F0; margin:0; font-size:0.9rem; line-height:1.5;">
                        Theft Confidence: <b style="font-size:1.1rem; color:#FF4B4B;">{risk_score:.1f}%</b><br>
                        Patterns show sharp short-term consumption drops compared to historic trends. Flagged for verification.
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="status-box" style="background: rgba(0, 240, 255, 0.05); border: 1px solid #00F0FF;">
                    <h4 style="color:#00F0FF; margin:0 0 8px 0; font-weight:700;">✅ BEHAVIOR NOMINAL</h4>
                    <p style="color:#E2E8F0; margin:0; font-size:0.9rem; line-height:1.5;">
                        Normal Confidence: <b style="font-size:1.1rem; color:#00F0FF;">{probabilities[0]*100:.1f}%</b><br>
                        Footprint signature maps perfectly within normal parameters.
                    </p>
                </div>
            """, unsafe_allow_html=True)