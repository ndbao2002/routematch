import redis
import pandas as pd
import os
import h3

# Connect to Redis
# If running outside Docker, host='localhost'. Inside, host='redis'
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
r = redis.Redis(host=REDIS_HOST, port=6379, db=0, decode_responses=True)

def hydrate():
    print(f"🔌 Connecting to Redis at {REDIS_HOST}...")
    try:
        r.ping()
    except redis.ConnectionError:
        print("❌ Redis not reachable. Is Docker running?")
        return

    print("💧 Hydrating Drivers...")
    df_drivers = pd.read_csv("data/raw/drivers.csv")
    
    # Fake initial locations (Just to make them queryable)
    # In real life, this comes from live GPS stream
    center_lat, center_lon = 10.762622, 106.660172
    
    pipe = r.pipeline()
    count = 0
    
    for index, row in df_drivers.iterrows():
        d_id = row['driver_id']
        
        # 1. Geo Index (Random start location near city center)
        import numpy as np
        lat = center_lat + np.random.normal(0, 0.05)
        lon = center_lon + np.random.normal(0, 0.05)
        
        # Add to Geo index per vehicle type
        # This should also include H3 indexing in a real system
        RESOLUTION = 8
        h3_index = h3.latlng_to_cell(lat, lon, RESOLUTION)
        pipe.zadd(f"drivers:h3:{h3_index}:{row['vehicle_type']}", {d_id: index})
        
        # 2. Profile (Static Features)
        pipe.hset(f"driver:{d_id}:profile", mapping={
            "vehicle_type": row['vehicle_type'],
            "max_load_kg": row['max_load_kg'],
            "joined_date": row['joined_date']
        })
        
        # 3. Initial State
        pipe.hset(f"driver:{d_id}:state", mapping={
            "status": "IDLE",
            "minutes_active": 0,
            "fatigue_index": 0.0,
            "cancel_rate": 0.0,
            "orders_completed": 0,
            "lat": lat,
            "lon": lon
        })
        
        count += 1
        if count % 500 == 0:
            pipe.execute()
            
    pipe.execute()
    print(f"✅ Hydrated {count} drivers.")

if __name__ == "__main__":
    hydrate()