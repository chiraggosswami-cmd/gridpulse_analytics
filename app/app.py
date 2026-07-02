import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time

# --- 1. CONFIGURATION & AGGRESSIVE PREMIUM THEME OVERRIDES ---
st.set_page_config(
    page_title="GridPulse Analytics Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global CSS injection to completely override Streamlit wrappers, metrics, and sidebars
st.markdown("""
    <style>
        /* Force Deep Cyber Dark Background */
        .stApp {
            background-color: #0B0E14 !important;
            font-family: 'Inter', -apple-system, sans-serif !important;
        }
        
        /* Premium Sidebar Overrides */
        section[data-testid="stSidebar"] {
            background-color: #11151D !important;
            border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
        }
        
        /* Text styling inside sidebar */
        section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] label {
            color: #E2E8F0 !important;
            font-weight: 500 !important;
        }

        /* Premium Custom Metric Card Container */
        .custom-card {
            background: #111622 !important;
            border: 1px solid rgba(255, 255, 255, 0.04) !important;
            border-radius: 16px !important;
            padding: 24px !important;
            margin-bottom: 20px !important;
            transition: all 0.3s ease;
        }
        .custom-card:hover {
            border-color: rgba(0, 240, 255, 0.2) !important;
            box-shadow: 0 10px 30px rgba(0, 240, 255, 0.03);
        }
        
        .card-label {
            color: #64748B !important;
            font-size: 0.75rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 1.5px !important;
            margin-bottom: 8px;
        }
        
        .card-value {
            color: #FFFFFF !important;
            font-size: 2.2rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.5px;
        }
        
        .card-unit {
            font-size: 1rem !important;
            color: #64748B !important;
            font-weight: 400 !important;
            margin-left: 4px;
        }

        /* Dynamic Status Indicator Panels */
        .status-panel {
            border-radius: 16px !important;
            padding: 24px !important;
            margin-top: 20px !important;
            background: #151926 !important;
        }
        
        /* Hide native Streamlit elements to maximize design purity */
        #MainMenu, footer, header {visibility: hidden;}
        div[data-testid="stMetricValue"] { display: none !important; }
    </style>
""", unsafe_allow_html=True)

# --- 2. HEADER INTERFACE ---
st.markdown("""
    <div style="padding: 10px 0px 25px 0px; border-bottom: 1px solid rgba(255,255,255,0.05); margin-bottom: 35px;">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
            <span style="background: rgba(0, 240, 255, 0.1); color: #00F0FF; padding: 4px 10px; border-radius: 6px; font-size: 0.7rem; font-weight: 700; letter-spacing: 1px;">SYSTEM ACTIVE</span>
            <span style="color: #64748B; font-size: 0.8rem;">•</span>
            <span style="color: #64748B; font-size: 0.8rem; font-weight: 500;">OOS TIME-SERIES MODEL V2</span>
        </div>
        <h1 style="color: white; margin: 0; font-size: 2.4rem; font-weight: 800; letter-spacing: -1px;">⚡ GridPulse Analytics</h1>
    </div>
""", unsafe_allow_html=True)

# --- 3. CACHED INFRASTRUCTURE LOADER ---
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
    st.error("⚠️ Model binaries missing. Run 'src/models.py' first.")
    st.stop()

# --- 4. SIDEBAR CONFIGURATION ---
st.sidebar.markdown("""
    <div style='padding: 10px 0 20px 0;'>
        <h2 style='margin:0; color:white; font-size:1.4rem; font-weight:700;'>Control Panel</h2>
        <p style='margin:4px 0 0 0; color:#64748B; font-size:0.85rem;'>Simulate grid telemetry arrays</p>
    </div>
""", unsafe_allow_html=True)

user_list = user_lookup['UserId'].unique()
selected_user = st.sidebar.selectbox("🎯 Target Subscriber Profile", user_list)

input_consumption = st.sidebar.slider(
    "Live Meter Transmission (kWh)", 
    min_value=0.0, max_value=150.0, value=25.0, step=0.1
)

calendar_context = st.sidebar.radio("📅 Day Context", ["Temporal Weekday", "Temporal Weekend"], horizontal=True)
is_weekend_numeric = 1 if "Weekend" in calendar_context else 0
day_of_week_dummy = 5 if is_weekend_numeric == 1 else 1

# --- 5. DATA SIMULATION & RECONSTRUCTION ENGINE ---
user_historical_mean = user_lookup[user_lookup['UserId'] == selected_user]['UserMean'].values[0]
diff_from_mean = input_consumption - user_historical_mean

# Generate a clean, realistic historical sequence (No abrupt layout box spikes)
np.random.seed(42)
historical_days = pd.date_range(end=pd.Timestamp.now(), periods=14)
simulated_stream = np.random.normal(loc=user_historical_mean, scale=max(0.5, user_historical_mean * 0.15), size=14)
# Keep variance smooth, then apply user input to current day
simulated_stream = np.clip(simulated_stream, 0.1, None)
simulated_stream[-1] = input_consumption
chart_df = pd.DataFrame({'Consumption': simulated_stream}, index=historical_days)

rolling_mean_7d = np.mean(simulated_stream[-7:])
diff_from_rolling = input_consumption - rolling_mean_7d

features = np.array([[
    input_consumption, day_of_week_dummy, is_weekend_numeric,
    user_historical_mean, diff_from_mean, rolling_mean_7d, diff_from_rolling
]])

# --- 6. METRIC BLOCKS OVERHAUL ---
metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

with metrics_col1:
    st.markdown(f"""
        <div class="custom-card">
            <div class="card-label">👤 User Historical Baseline</div>
            <div class="card-value">{user_historical_mean:.2f}<span class="card-unit">kWh</span></div>
        </div>
    """, unsafe_allow_html=True)

with metrics_col2:
    st.markdown(f"""
        <div class="custom-card">
            <div class="card-label">⚡ Injected Meter Load</div>
            <div class="card-value" style="color: #00F0FF !important;">{input_consumption:.2f}<span class="card-unit">kWh</span></div>
        </div>
    """, unsafe_allow_html=True)

with metrics_col3:
    # Use electric color indicators matching luxury dashboard aesthetics
    divergence_color = "#FF4B4B" if abs(diff_from_mean) > (user_historical_mean * 1.5) else "#00F0FF"
    st.markdown(f"""
        <div class="custom-card">
            <div class="card-label">🔄 Footprint Divergence</div>
            <div class="card-value" style="color: {divergence_color} !important;">{diff_from_mean:+.2f}<span class="card-unit">kWh</span></div>
        </div>
    """, unsafe_allow_html=True)

# --- 7. WORKFLOW GRAPHICS & PROCESSING INTERFACE ---
left_panel, right_panel = st.columns([2, 1])

with left_panel:
    st.markdown("<h3 style='color:white; font-size:1.1rem; font-weight:600; margin: 15px 0 15px 0;'>📈 14-Day Trailing Consumption Stream</h3>", unsafe_allow_html=True)
    chart_color = "#FF4B4B" if abs(diff_from_mean) > (user_historical_mean * 1.5) else "#00F0FF"
    st.line_chart(chart_df, color=chart_color, height=300)

with right_panel:
    st.markdown("<h3 style='color:white; font-size:1.1rem; font-weight:600; margin: 15px 0 15px 0;'>🤖 Diagnostics Interface</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B; font-size:0.85rem; margin-bottom:20px;'>Scan operational transmission vectors using trained XGBoost patterns:</p>", unsafe_allow_html=True)
    
    if st.button("🚀 Execute Anomaly Evaluation Scan", use_container_width=True):
        with st.spinner("Processing telemetry metrics..."):
            time.sleep(0.3)
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            risk_score = probabilities[1] * 100
            
        if prediction == 1:
            st.markdown(f"""
                <div class="status-panel" style="border-left: 4px solid #FF4B4B; box-shadow: 0 0 30px rgba(255,75,75,0.05);">
                    <h4 style="color:#FF4B4B; margin:0 0 6px 0; font-size:1rem; font-weight:700;">🚨 CRITICAL RISK DETECTED</h4>
                    <p style="color:#94A3B8; margin:0; font-size:0.85rem; line-height:1.5;">
                        Theft Probability: <span style="color:#FF4B4B; font-weight:700;">{risk_score:.1f}%</span><br>
                        Significant immediate drop patterns observed. Audit sequence flagged.
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="status-panel" style="border-left: 4px solid #00F0FF; box-shadow: 0 0 30px rgba(0,240,255,0.05);">
                    <h4 style="color:#00F0FF; margin:0 0 6px 0; font-size:1rem; font-weight:700;">✅ PATTERN NOMINAL</h4>
                    <p style="color:#94A3B8; margin:0; font-size:0.85rem; line-height:1.5;">
                        System Integrity: <span style="color:#00F0FF; font-weight:700;">{probabilities[0]*100:.1f}%</span><br>
                        Telemetry tracks safely inside expected historic density channels.
                    </p>
                </div>
            """, unsafe_allow_html=True)