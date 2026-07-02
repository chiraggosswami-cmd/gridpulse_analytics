import pandas as pd
import numpy as np
import os

def load_and_reshape_grid_data(file_path: str) -> pd.DataFrame:
    """
    Loads raw smart grid wide-format data, melts it into time-series long-format,
    optimizes memory signatures, and handles calendar flags.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw data file not found at: {file_path}")
        
    print("🚀 Step 1: Loading raw wide-format dataset...")
    df = pd.read_csv(file_path)
    
    print("🔄 Step 2: Melting dataset from Wide to Long format...")
    df_long = pd.melt(
        df, 
        id_vars=['UserId', 'IsStealer'], 
        var_name='Date', 
        value_name='Consumption'
    )
    
    print("📅 Step 3: Parsing datetime structures...")
    df_long['Date'] = pd.to_datetime(df_long['Date'])
    
    print("🛠️ Step 4: Extracting calendar variations...")
    df_long['DayOfWeek'] = df_long['Date'].dt.dayofweek
    df_long['IsWeekend'] = df_long['DayOfWeek'].isin([5, 6]).astype(int)
    
    # Memory optimization: standard float64 uses massive RAM over 9M rows
    print("💾 Step 5: Optimizing memory usage...")
    df_long['Consumption'] = df_long['Consumption'].astype(np.float32)
    df_long['DayOfWeek'] = df_long['DayOfWeek'].astype(np.int8)
    df_long['IsWeekend'] = df_long['IsWeekend'].astype(np.int8)
    df_long['IsStealer'] = df_long['IsStealer'].astype(np.int8)
    
    print("✅ Data Pipeline Execution Complete!")
    return df_long

if __name__ == "__main__":
    # This block allows us to easily test the pipeline file individually later
    print("Pipeline module initialized directly.")