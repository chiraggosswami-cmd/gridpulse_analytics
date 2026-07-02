from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import os

app = FastAPI(
    title="GridPulse ML Inference Engine",
    description="Production REST API for Hybrid Smart Grid Anomaly Detection",
    version="2.0.0"
)

# Define the structure of incoming operational telemetry packets
class TelemetryPacket(BaseModel):
    consumption: float
    day_of_week: int
    is_weekend: int
    user_mean: float
    diff_from_mean: float
    rolling_mean_7d: float
    diff_from_rolling: float

# Load binaries safely when the API stands up
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))
xgb_path = os.path.join(MODELS_DIR, "xgb_anomaly_model.pkl")
iso_path = os.path.join(MODELS_DIR, "iso_forest_model.pkl")

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
    """Confirms API status and system health."""
    if xgb_model is None or iso_forest_model is None:
        return {"status": "unhealthy", "error": "Model binaries missing on server"}
    return {"status": "healthy", "service": "GridPulse Inference Hub"}

@app.post("/predict")
def run_hybrid_inference(packet: TelemetryPacket):
    """Processes incoming feature vectors through both Supervised and Unsupervised engines."""
    if xgb_model is None or iso_forest_model is None:
        raise HTTPException(status_code=503, detail="ML inference engines are offline.")
        
    # Reconstruct the feature array exactly how the models expect it
    features = np.array([[
        packet.consumption, packet.day_of_week, packet.is_weekend,
        packet.user_mean, packet.diff_from_mean, packet.rolling_mean_7d,
        packet.diff_from_rolling
    ]])
    
    # Run predictions
    xgb_pred = int(xgb_model.predict(features)[0])
    xgb_prob = float(xgb_model.predict_proba(features)[0][1])
    if_pred = int(iso_forest_model.predict(features)[0])
    
    # Enforce hybrid security flag logic
    is_anomaly = 1 if (xgb_pred == 1 or if_pred == -1) else 0
    
    return {
        "is_anomaly": is_anomaly,
        "xgb_prediction": xgb_pred,
        "xgb_theft_probability": xgb_prob,
        "isolation_forest_prediction": if_pred
    }