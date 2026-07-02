import os
import pickle
import pandas as pd
import numpy as np
import kagglehub
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest  # <-- NEW IMPORT
from sklearn.metrics import classification_report

from data_pipeline import load_and_reshape_grid_data
from feature_eng import build_advanced_features

def train_production_model(data_path: str, output_dir: str = "models"):
    """
    Executes the end-to-end ML lifecycle: orchestrates extraction, 
    trains both Supervised (XGBoost) and Unsupervised (Isolation Forest) models.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    df_long = load_and_reshape_grid_data(data_path)
    df_featured = build_advanced_features(df_long)
    
    feature_cols = [
        'Consumption', 'DayOfWeek', 'IsWeekend', 'UserMean', 
        'Consumption_Diff_From_Mean', 'Rolling_Mean_7D', 'Consumption_Diff_From_Rolling'
    ]
    
    X = df_featured[feature_cols]
    y = df_featured['IsStealer']
    
    print("✂️ Step 3: Applying Out-of-Sample Time-Series Splitting...")
    split_date = pd.Timestamp('2016-01-01')
    train_mask = df_featured['Date'] < split_date
    test_mask = df_featured['Date'] >= split_date
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    # ----------------------------------------------------
    # MODEL 1: SUPERVISED XGBOOST
    # ----------------------------------------------------
    print("⚖️ Step 4: Computing target weight balances...")
    imbalance_ratio = y_train.value_counts()[0] / y_train.value_counts()[1]
    
    print("🏋️‍♂️ Step 5: Commencing XGBoost training sequence...")
    xgb_model = XGBClassifier(
        n_estimators=200, max_depth=5, scale_pos_weight=imbalance_ratio,
        random_state=42, eval_metric='logloss', n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    
    # ----------------------------------------------------
    # MODEL 2: UNSUPERVISED ISOLATION FOREST (NEW)
    # ----------------------------------------------------
    print("🌲 Step 6: Commencing Isolation Forest outlier training...")
    # contamination represents the expected percentage of anomalies in the grid (~8%)
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.08,
        random_state=42,
        n_jobs=-1
    )
    # Isolation Forest is unsupervised, so it only trains on feature distributions (X_train), ignoring y_train!
    iso_forest.fit(X_train)
    
    print("📊 Step 7: Generating validation performance diagnostics...")
    y_pred = xgb_model.predict(X_test)
    print("\n--- XGBoost Model Evaluation Results ---")
    print(classification_report(y_test, y_pred))
    
    print("📦 Step 8: Saving production model artifacts...")
    # Save XGBoost
    with open(os.path.join(output_dir, "xgb_anomaly_model.pkl"), 'wb') as f:
        pickle.dump(xgb_model, f)
        
    # Save Isolation Forest (NEW Artifact)
    with open(os.path.join(output_dir, "iso_forest_model.pkl"), 'wb') as f:
        pickle.dump(iso_forest, f)
        
    # Save Lookup Snapshot
    user_lookup = df_featured[['UserId', 'UserMean']].drop_duplicates().reset_index(drop=True)
    with open(os.path.join(output_dir, "user_lookup.pkl"), 'wb') as f:
        pickle.dump(user_lookup, f)
        
    print(f"🎉 Success! Hybrid core exported to: /{output_dir}")

if __name__ == "__main__":
    print("Model pipeline script compiled. Ready for cloud orchestration.")
    print("☁️ Downloading latest smart grid dataset directly from Kaggle cache...")
    download_path = kagglehub.dataset_download("ahmedrady66/theft-detection-scheme-in-smart-grids")
    csv_file_path = os.path.join(download_path, "AllData.csv")
    train_production_model(csv_file_path)