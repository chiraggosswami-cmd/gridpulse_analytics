import os
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import kagglehub

# Import the custom modules we built earlier!
from data_pipeline import load_and_reshape_grid_data
from feature_eng import build_advanced_features

def train_production_model(data_path: str, output_dir: str = "models"):
    """
    Executes the end-to-end ML lifecycle: orchestrates extraction, 
    applies time-series sorting, resolves class imbalances, and saves binaries.
    """
    # 1. Create target output directory if it doesn't exist yet
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Run background modules
    df_long = load_and_reshape_grid_data(data_path)
    df_featured = build_advanced_features(df_long)
    
    # 3. Define features matching your original notebook array configurations
    feature_cols = [
        'Consumption', 'DayOfWeek', 'IsWeekend', 'UserMean', 
        'Consumption_Diff_From_Mean', 'Rolling_Mean_7D', 'Consumption_Diff_From_Rolling'
    ]
    
    X = df_featured[feature_cols]
    y = df_featured['IsStealer']
    
    print("✂️ Step 3: Applying Out-of-Sample Time-Series Splitting...")
    # Establish a clean temporal boundary cut point (e.g., Jan 1st, 2016)
    split_date = pd.Timestamp('2016-01-01')
    
    # Isolate training indices into past history, and test indices into future logs
    train_mask = df_featured['Date'] < split_date
    test_mask = df_featured['Date'] >= split_date
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    print(f"📍 Train Range: {df_featured[train_mask]['Date'].min().strftime('%Y-%m-%d')} to {df_featured[train_mask]['Date'].max().strftime('%Y-%m-%d')}")
    print(f"📍 Test Range:  {df_featured[test_mask]['Date'].min().strftime('%Y-%m-%d')} to {df_featured[test_mask]['Date'].max().strftime('%Y-%m-%d')}")
    
    print("⚖️ Step 4: Computing target weight balances...")
    # Dynamic computation of scale_pos_weight to balance minority theft patterns
    imbalance_ratio = y_train.value_counts()[0] / y_train.value_counts()[1]
    
    print("🏋️‍♂️ Step 5: Commencing XGBoost training sequence...")
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        scale_pos_weight=imbalance_ratio,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1 # Uses all available CPU cores for lightning fast training local acceleration
    )
    xgb_model.fit(X_train, y_train)
    
    print("📊 Step 6: Generating validation performance diagnostics...")
    y_pred = xgb_model.predict(X_test)
    print("\n--- Model Evaluation Results ---")
    print(classification_report(y_test, y_pred))
    
    print("📦 Step 7: Saving model artifacts to storage directory...")
    # Save the core model binary
    model_out = os.path.join(output_dir, "xgb_anomaly_model.pkl")
    with open(model_out, 'wb') as f:
        pickle.dump(xgb_model, f)
        
    # Generate and save the user lookup snapshot table for Streamlit dropdown arrays
    lookup_out = os.path.join(output_dir, "user_lookup.pkl")
    user_lookup = df_featured[['UserId', 'UserMean']].drop_duplicates().reset_index(drop=True)
    with open(lookup_out, 'wb') as f:
        pickle.dump(user_lookup, f)
        
    print(f"🎉 Success! Production model and user lookup tables exported cleanly to: /{output_dir}")

if __name__ == "__main__":
    print("Model pipeline script compiled. Ready for cloud orchestration.")
    
    print("☁️ Downloading latest smart grid dataset directly from Kaggle cache...")
    # This automatically downloads the file into a temporary system cache folder,
    # keeping your local project folder 100% clean of massive CSV files!
    download_path = kagglehub.dataset_download("ahmedrady66/theft-detection-scheme-in-smart-grids")
    
    # Locate the AllData.csv inside the temporary cache path
    csv_file_path = os.path.join(download_path, "AllData.csv")
    
    # Fire off our out-of-sample production model training loop!
    train_production_model(csv_file_path)