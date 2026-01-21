import os
import pandas as pd
import numpy as np
import h3

def engineer_realtime_demand(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates sliding window demand without leakage.
    Fixes the 'Duplicate Index' error using sort + values assignment.
    """
    print("   -> ⚙️ Engineering Real-time Demand (H3 Rolling Window)...")
    
    # 1. Sort by Location AND Time to ensure deterministic grouping
    df = df.sort_values(by=['pickup_h3', 'created_at'])
    
    # 2. Use a temporary indexed DF for the rolling calc
    temp_df = df.set_index('created_at')
    
    # 3. Calculate Rolling Window (Closed='left' prevents leakage)
    # This returns a Series indexed by (pickup_h3, created_at)
    rolled_series = (
        temp_df.groupby('pickup_h3')['order_id']
        .rolling('60min', closed='left')
        .count()
    )
    
    # 4. Assign via .values (The Fix)
    # Because 'df' is sorted exactly like 'temp_df', the positions match 1:1.
    # We bypass index alignment to avoid "ValueError: duplicate axis".
    df['h3_demand_60m'] = rolled_series.values
    
    # 5. Fill NaNs (First items in window)
    df['h3_demand_60m'] = df['h3_demand_60m'].fillna(0)
    
    return df

def calculate_bayesian_smoothing(df: pd.DataFrame, C: int = 20) -> pd.DataFrame:
    """
    Calculates 'Driver Acceptance Rate' using Bayesian Smoothing (Expanding Mean).
    Solves the Cold Start problem without data leakage (Time-Split safe).
    
    Formula: (C * Global_Mean + Driver_History_Sum) / (C + Driver_History_Count)
    """
    print("   -> ⚙️ Engineering Driver History (Bayesian Smoothing)...")
    
    # Ensure time order
    df = df.sort_values('created_at')
    
    # Global Prior (The average acceptance rate of the whole platform)
    # In production, this would be a fixed constant from yesterday's data
    global_mean = df['is_accepted'].mean()
    
    # Group by driver
    grouped = df.groupby('driver_id')['is_accepted']
    
    # --- BUG FIX (CRITICAL) ---
    # CORRECT WAY: Calculate Inclusive Sum - Current Value
    # This is "Vectorized Previous Sum" that keeps boundaries safe.
    history_sum_inclusive = grouped.cumsum()
    history_sum_previous = history_sum_inclusive - df['is_accepted']
    
    history_count_previous = grouped.cumcount() # 0, 1, 2... (Correct n)
    # --------------------------
    
    # Apply Formula
    df['driver_global_accept_rate'] = (
        (C * global_mean) + history_sum_previous
    ) / (C + history_count_previous)
    
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw logs into Model-Ready Features.
    """
    print("--- 🛠️ Starting Feature Engineering ---")
    
    # 1. Spatial Features: H3 Indexing
    # R9 is ~0.1km² per cell, R8 is ~0.74km² per cell
    RESOLUTION = 8
    
    print("Mapping H3 Spatial Indices...")
    df['pickup_h3'] = df.apply(
        lambda row: h3.latlng_to_cell(row['pickup_lat'], row['pickup_lon'], RESOLUTION), 
        axis=1
    )
    
    # 2. Context Features: Supply/Demand Density
    df = engineer_realtime_demand(df)

    # 3. Temporal Features: Cyclical Encoding
    print("Encoding Cyclical Time...")
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
    
    # 4. Interaction Features: Price Sensitivity
    # Avoid division by zero
    df['price_per_km'] = df['shipping_fee'] / (df['distance_km'] + 0.01) 
    
    # 5. Driver History: "Rolling Acceptance Rate"
    df = calculate_bayesian_smoothing(df)

    # Cleanup: Drop raw IDs and raw spatial strings
    # CRITICAL CHANGE: We Keep 'order_id' now for Recall@k Evaluation!
    drop_cols = ['user_id', 'created_at', 'offered_at', 'driver_id', 
                 'pickup_lat', 'pickup_lon', 'driver_lat', 'driver_lon', 
                 'hour_of_day', 'dropoff_lat', 'dropoff_lon', 'pickup_h3']
    
    df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    print(f"✅ Finished. Feature Count: {len(df_clean.columns)}")
    print(f"   Features: {list(df_clean.columns)}")
    return df_clean

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    N = None

    # Simulate loading raw interaction logs
    interaction_df = pd.read_csv("data/raw/interaction_logs.csv", parse_dates=['offered_at'])
    if N is not None:
        sample_df = interaction_df.sample(n=N, random_state=42)
    else:
        sample_df = interaction_df.copy()

    order_df = pd.read_csv("data/raw/orders.csv", parse_dates=['created_at'])
    merged_df = sample_df.merge(order_df, on='order_id', how='left')

    feature_df = engineer_features(merged_df)

    if not os.path.exists("data/processed/"):
        os.makedirs("data/processed/")
    feature_df.to_csv("data/processed/feature_data.csv", index=False)
    print("Feature data saved to data/processed/feature_data.csv")