from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import os
from scipy.stats import ks_2samp  # <-- NEW STATISTICAL DRIFT ENGINE

app = FastAPI(
    title="GridPulse ML Inference Engine",
    description="Production REST API with Real-time Data Drift Monitoring",
    version="3.0.0"
)

class TelemetryPacket(BaseModel):
    consumption: float
    day_of_week: int
    is_weekend: int
    user_mean: float
    diff_from_mean: float
    rolling_mean_7d: float
    diff_from_rolling: float

# --- LOADING BINARIES & HISTORICAL REFERENCE THEFT BASELINE ---
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))
xgb_path = os.path.join(MODELS_DIR, "xgb_anomaly_model.pkl")
iso_path = os.path.join(MODELS_DIR, "iso_forest_model.pkl")

# Live in-memory streaming window buffer to capture drift sequences
LIVE_STREAM_BUFFER = []
BUFFER_MAX_SIZE = 50  # Run a statistical check every 50 transmissions
DRIFT_STATUS = {"drift_detected": False, "p_value": 1.0, "status": "Baseline Stable"}

# Generate a mock training reference distribution matching original feature engineering scales
np.random.seed(42)
TRAINING_REFERENCE_DISTRIBUTION = np.random.normal(loc=15.2, scale=8.4, size=500)

try:
    with open(xgb_path, 'rb') as f:
        xgb_model = pickle.load(f)
    with open(iso_path, 'rb') as f:
        iso_forest_model = pickle.load(f)
    print("🚀 API Engine loaded ML model binaries successfully.")
except Exception as e:
    xgb_model, iso_forest_model = None, None
    print(f"⚠️ Critical Error loading models: {e}")


@app.get("/")
def health_check():
    if xgb_model is None or iso_forest_model is None:
        return {"status": "unhealthy", "error": "Model binaries missing on server"}
    return {
        "status": "healthy", 
        "service": "GridPulse Inference Hub",
        "drift_monitoring": DRIFT_STATUS
    }


@app.post("/predict")
def run_hybrid_inference(packet: TelemetryPacket):
    if xgb_model is None or iso_forest_model is None:
        raise HTTPException(status_code=503, detail="ML inference engines are offline.")
        
    # 1. Update live buffer for drift checking
    LIVE_STREAM_BUFFER.append(packet.consumption)
    if len(LIVE_STREAM_BUFFER) > BUFFER_MAX_SIZE:
        LIVE_STREAM_BUFFER.pop(0) # Maintain a sliding window queue
        
    # 2. Compute Kolmogorov-Smirnov Test if buffer is full
    if len(LIVE_STREAM_BUFFER) == BUFFER_MAX_SIZE:
        # Compare incoming sliding values with training baseline distributions
        stat, p_val = ks_2samp(LIVE_STREAM_BUFFER, TRAINING_REFERENCE_DISTRIBUTION)
        DRIFT_STATUS["p_value"] = float(p_val)
        
        if p_val < 0.05: # Statistical significance threshold
            DRIFT_STATUS["drift_detected"] = True
            DRIFT_STATUS["status"] = "🚨 DRIFT DETECTED - RETRAIN REQUIRED"
        else:
            DRIFT_STATUS["drift_detected"] = False
            DRIFT_STATUS["status"] = "✅ Distribution Stable"

    # 3. Model predictions
    features = np.array([[
        packet.consumption, packet.day_of_week, packet.is_weekend,
        packet.user_mean, packet.diff_from_mean, packet.rolling_mean_7d,
        packet.diff_from_rolling
    ]])
    
    xgb_pred = int(xgb_model.predict(features)[0])
    xgb_prob = float(xgb_model.predict_proba(features)[0][1])
    if_pred = int(iso_forest_model.predict(features)[0])
    
    is_anomaly = 1 if (xgb_pred == 1 or if_pred == -1) else 0
    
    return {
        "is_anomaly": is_anomaly,
        "xgb_prediction": xgb_pred,
        "xgb_theft_probability": xgb_prob,
        "isolation_forest_prediction": if_pred,
        "drift_metrics": DRIFT_STATUS
    }