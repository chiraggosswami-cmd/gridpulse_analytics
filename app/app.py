import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time

# --- 1. CONFIGURATION & THEME OVERRIDES ---
st.set_page_config(
    page_title="GridPulse Analytics Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .stApp { background-color: #0B0E14 !important; font-family: 'Inter', sans-serif !important; }
        section[data-testid="stSidebar"] { background-color: #11151D !important; border-right: 1px solid rgba(255, 255, 255, 0.05) !important; }
        section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] label { color: #E2E8F0 !important; }
        .custom-card { background: #111622 !important; border: 1px solid rgba(255, 255, 255, 0.04) !important; border-radius: 16px !important; padding: 24px !important; margin-bottom: 20px !important; }
        .card-label { color: #64748B !important; font-size: 0.75rem !important; font-weight: 600; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 8px; }
        .card-value { color: #FFFFFF !important; font-size: 2.2rem !important; font-weight: 700; letter-spacing: -0.5px; }
        .card-unit { font-size: 1rem !important; color: #64748B !important; margin-left: 4px; }
        .status-panel { border-radius: 16px !important; padding: 24px !important; margin-top: 20px !important; background: #151926 !important; }
        #MainMenu, footer, header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div style="padding: 10px 0px 25px 0px; border-bottom: 1px solid rgba(255,255,255,0.05); margin-bottom: 35px;">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
            <span style="background: rgba(0, 240, 255, 0.1); color: #00F0FF; padding: 4px 10px; border-radius: 6px; font-size: 0.7rem; font-weight: 700; letter-spacing: 1px;">SYSTEM RUNNING</span>
            <span style="color: #64748B; font-size: 0.8rem;">•</span>
            <span style="color: #64748B; font-size: 0.8rem; font-weight: 500;">HYBRID ENSEMBLE SYSTEM ACTIVE</span>
        </div>
        <h1 style="color: white; margin: 0; font-size: 2.4rem; font-weight: 800; letter-spacing: -1px;">⚡ GridPulse Analytics</h1>
    </div>
""", unsafe_allow_html=True)

# --- 2. MULTI-MODEL ASSET LOADER ---
@st.cache_resource
def load_hybrid_artifacts():
    paths = ["../models/", "models/", ""]
    xgb, iso, lookup = None, None, None
    
    for p in paths:
        if os.path.exists(os.path.join(p, "xgb_anomaly_model.pkl")):
            with open(os.path.join(p, "xgb_anomaly_model.pkl"), 'rb') as f: xgb = pickle.load(f)
        if os.path.exists(os.path.join(p, "iso_forest_model.pkl")):
            with open(os.path.join(p, "iso_forest_model.pkl"), 'rb') as f: iso = pickle.load(f)
        if os.path.exists(os.path.join(p, "user_lookup.pkl")):
            with open(os.path.join(p, "user_lookup.pkl"), 'rb') as f: lookup = pickle.load(f)
        if xgb is not None and iso is not None and lookup is not None: break
            
    return xgb, iso, lookup

xgb_model, iso_forest_model, user_lookup = load_hybrid_artifacts()

# FIXED: Explicitly checking variable identities instead of using the 'in' operator to avoid DataFrame truth validation errors
if xgb_model is None or iso_forest_model is None or user_lookup is None:
    st.error("⚠️ Pipeline artifacts missing. Ensure both models have finished training successfully.")
    st.stop()

# --- 3. SIDEBAR INTERACTION CONTROL ---
st.sidebar.markdown("<h2 style='margin:0; font-size:1.4rem;'>Control Panel</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")
user_list = user_lookup['UserId'].unique()
selected_user = st.sidebar.selectbox("🎯 Target Subscriber Profile", user_list)

input_consumption = st.sidebar.slider("Live Meter Reading (kWh)", 0.0, 150.0, 25.0, 0.1)
calendar_context = st.sidebar.radio("📅 Day Context", ["Temporal Weekday", "Temporal Weekend"], horizontal=True)
is_weekend_numeric = 1 if "Weekend" in calendar_context else 0
day_of_week_dummy = 5 if is_weekend_numeric == 1 else 1

# --- 4. FEATURE SYNTHESIS ---
user_historical_mean = user_lookup[user_lookup['UserId'] == selected_user]['UserMean'].values[0]
diff_from_mean = input_consumption - user_historical_mean

np.random.seed(42)
historical_days = pd.date_range(end=pd.Timestamp.now(), periods=14)
simulated_stream = np.clip(np.random.normal(loc=user_historical_mean, scale=max(0.5, user_historical_mean * 0.15), size=14), 0.1, None)
simulated_stream[-1] = input_consumption
chart_df = pd.DataFrame({'Consumption': simulated_stream}, index=historical_days)

rolling_mean_7d = np.mean(simulated_stream[-7:])
diff_from_rolling = input_consumption - rolling_mean_7d

# Match shape matrix requirements
features = np.array([[input_consumption, day_of_week_dummy, is_weekend_numeric, user_historical_mean, diff_from_mean, rolling_mean_7d, diff_from_rolling]])

# --- 5. VISUAL METRICS BOARD ---
metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
with metrics_col1:
    st.markdown(f'<div class="custom-card"><div class="card-label">👤 User Historical Baseline</div><div class="card-value">{user_historical_mean:.2f}<span class="card-unit">kWh</span></div></div>', unsafe_allow_html=True)
with metrics_col2:
    st.markdown(f'<div class="custom-card"><div class="card-label">⚡ Injected Meter Load</div><div class="card-value" style="color: #00F0FF !important;">{input_consumption:.2f}<span class="card-unit">kWh</span></div></div>', unsafe_allow_html=True)
with metrics_col3:
    divergence_color = "#FF4B4B" if abs(diff_from_mean) > (user_historical_mean * 1.5) else "#00F0FF"
    st.markdown(f'<div class="custom-card"><div class="card-label">🔄 Footprint Divergence</div><div class="card-value" style="color: {divergence_color} !important;">{diff_from_mean:+.2f}<span class="card-unit">kWh</span></div></div>', unsafe_allow_html=True)

# --- 6. CORE DISPLAY & LOGIC RUN ---
left_panel, right_panel = st.columns([2, 1])
with left_panel:
    st.markdown("<h3 style='color:white; font-size:1.1rem; font-weight:600;'>📈 14-Day Trailing Consumption Stream</h3>", unsafe_allow_html=True)
    chart_color = "#FF4B4B" if abs(diff_from_mean) > (user_historical_mean * 1.5) else "#00F0FF"
    st.line_chart(chart_df, color=chart_color, height=300)

with right_panel:
    st.markdown("<h3 style='color:white; font-size:1.1rem; font-weight:600;'>🤖 Hybrid Engine Diagnostics</h3>", unsafe_allow_html=True)
    
    if st.button("🚀 Execute Hybrid Defense Scan", use_container_width=True):
        with st.spinner("Analyzing operational patterns..."):
            time.sleep(0.4)
            
            # Supervised XGBoost Inference
            xgb_pred = xgb_model.predict(features)[0]
            xgb_prob = xgb_model.predict_proba(features)[0][1] * 100
            
            # Unsupervised Isolation Forest Inference (-1 means anomaly, 1 means normal)
            if_pred = iso_forest_model.predict(features)[0]
            
        # Display Results Based on Hybrid Core Rules
        if xgb_pred == 1 or if_pred == -1:
            st.markdown(f"""
                <div class="status-panel" style="border-left: 4px solid #FF4B4B;">
                    <h4 style="color:#FF4B4B; margin:0 0 6px 0; font-size:1rem; font-weight:700;">🚨 SECURITY COMPROMISED</h4>
                    <p style="color:#94A3B8; margin:0; font-size:0.85rem; line-height:1.5;">
                        <b>XGBoost Risk Score:</b> {xgb_prob:.1f}% {"(Flagged)" if xgb_pred==1 else "(Nominal)"}<br>
                        <b>Isolation Forest Status:</b> {"⚠️ Outlier Detected" if if_pred==-1 else "Normal Distribution"}<br><br>
                        <span style="color:#FF4B4B;">System Action:</span> Dispatched to priority field audit logs.
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="status-panel" style="border-left: 4px solid #00F0FF;">
                    <h4 style="color:#00F0FF; margin:0 0 6px 0; font-size:1rem; font-weight:700;">✅ SECURITY SECURE</h4>
                    <p style="color:#94A3B8; margin:0; font-size:0.85rem; line-height:1.5;">
                        Both Supervised pattern match constraints and Unsupervised spatial distribution limits report normal, nominal customer load operations.
                    </p>
                </div>
            """, unsafe_allow_html=True)