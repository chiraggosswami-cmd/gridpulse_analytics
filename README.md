# ⚡ GridPulse: Advanced Smart Grid Analytics & Theft Detection

GridPulse is an end-to-end Machine Learning and data engineering pipeline designed to detect electricity theft and consumption anomalies in smart grids. Processing over 9 million daily data points, this project scales from raw time-series analysis to an interactive, production-ready monitoring dashboard.

## 🚀 Project Architecture
This project is systematically engineered into modular phases:
- **Modular Data Pipeline:** Automated cleaning and memory-optimized data reshaping.
- **Advanced Feature Engineering:** Historical user baselines and short-term 7-day rolling windows.
- **Ensemble Machine Learning:** Optimized XGBoost classifier with custom class balancing (`scale_pos_weight`) to handle heavy anomaly imbalances.
- **Operational Dashboard:** A live, interactive monitoring application built with Streamlit.

## 📁 Repository Structure
```text
├── src/                # Core Python backend engine
│   ├── data_pipeline.py# Data cleaning and reshaping
│   ├── feature_eng.py  # Rolling metrics & context features
│   └── models.py       # ML Training & evaluation configurations
├── app/                # Frontend dashboard layer
│   └── app.py          # Streamlit user interface
├── requirements.txt    # Production dependencies
└── README.md           # Documentation