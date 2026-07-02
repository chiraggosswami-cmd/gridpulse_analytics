import pandas as pd

def build_advanced_features(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Takes clean long-format grid data, computes cumulative user baselines,
    sorts time-series arrays chronologically, and appends a 7-day moving window.
    """
    print("📈 Step 1: Calculating individual user historical baselines...")
    # Computes long-term behavior patterns
    df_long['UserMean'] = df_long.groupby('UserId')['Consumption'].transform('mean')
    df_long['Consumption_Diff_From_Mean'] = df_long['Consumption'] - df_long['UserMean']
    
    print("⏳ Step 2: Sorting data chronologically per subscriber...")
    # CRITICAL: Sorting prevents future information bleed and window shuffling
    df_long = df_long.sort_values(by=['UserId', 'Date']).reset_index(drop=True)
    
    print("🔄 Step 3: Generating short-term 7-day rolling window tracks...")
    # Rolling feature captures sudden drops (e.g. bypassing meters)
    df_long['Rolling_Mean_7D'] = df_long.groupby('UserId')['Consumption'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    df_long['Consumption_Diff_From_Rolling'] = df_long['Consumption'] - df_long['Rolling_Mean_7D']
    
    print("✅ Feature Engineering Completed successfully!")
    return df_long

if __name__ == "__main__":
    print("Feature engineering engine module initialized directly.")