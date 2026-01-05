import os
import pandas as pd
import numpy as np
import h3

def engineer_realtime_demand(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Sort by Location AND Time
    # This ensures the order of rows is deterministic and grouped correctly.
    # We need 'pickup_h3' first because that's our grouping key.
    df = df.sort_values(by=['pickup_h3', 'created_at'])
    
    # 2. Use a temporary DataFrame for calculation
    # We do NOT set the index on the main 'df' to avoid losing our integer index or 
    # dealing with duplicates on the main object.
    temp_df = df.set_index('created_at')
    
    # 3. Calculate Rolling Window
    # The result here is a Series with a MultiIndex (pickup_h3, created_at).
    # Since 'df' (and thus 'temp_df') is already sorted by [pickup_h3, created_at],
    # the output of this groupby operation will match the row order of 'df' exactly.
    rolled_series = (
        temp_df.groupby('pickup_h3')['order_id']
        .rolling('60min', closed='left') # 'left' excludes current time (Anti-Leakage)
        .count()
    )
    
    # 4. Assign via .values (The Fix)
    # Instead of letting Pandas try to align by index (which fails on duplicates),
    # we extract the raw numpy array. Since the sort order matches, this is safe.
    df['h3_demand_60m'] = rolled_series.values
    
    # 5. Fill NaNs (First items in the window have no history)
    df['h3_demand_60m'] = df['h3_demand_60m'].fillna(0)
    
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
    # Count how many orders are happening in this H3 cell at this hour
    # (In production, this would come from Redis features)
    print("Calculating Spatiotemporal Density...")
    df = engineer_realtime_demand(df)

    # 3. Temporal Features: Cyclical Encoding
    # Maps 23:00 and 00:00 to be close to each other
    print("Encoding Cyclical Time...")
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
    
    # 4. Interaction Features: Price Sensitivity
    # Ratio of Shipping Fee to Distance (Price per KM)
    # High price/km = Drivers like it. Low price/km = Drivers reject it.
    df['price_per_km'] = df['shipping_fee'] / (df['distance_km'] + 0.01) # Avoid div/0
    
    # 5. Driver History: "Rolling Acceptance Rate"
    # (Simulating historical behavior - vital for personalization)
    # In production, this comes from the 'driver:{id}:state' Redis hash
    print("Calculating Driver History...")
    driver_stats = df.groupby('driver_id')['is_accepted'].mean().reset_index()
    driver_stats.rename(columns={'is_accepted': 'driver_global_accept_rate'}, inplace=True)
    df = df.merge(driver_stats, on='driver_id', how='left')

    # Cleanup: Drop raw IDs that leak info or cause overfitting
    drop_cols = ['user_id', 'created_at', 'offered_at'] # Keep order_id/driver_id for debugging
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    
    print(f"✅ Finished. Feature Count: {len(df.columns)}")
    return df

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