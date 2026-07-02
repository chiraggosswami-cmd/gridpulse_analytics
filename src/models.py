import os
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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
    
    print("✂️ Step 3: Splitting datasets into evaluation segments...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
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
    print("Model pipeline script compiled. Ready for orchestration.")
    # Example local call pattern (uncomment to run locally once data path exists):
    # train_production_model("path/to/AllData.csv")