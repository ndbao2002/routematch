import pandas as pd
import numpy as np
import uuid
from datetime import datetime, timedelta
import random
import os
import math

# ===========================
# CONFIGURATION
# ===========================
NUM_DRIVERS = 2000
NUM_ORDERS = 200000
START_DATE = datetime(2024, 1, 1)

# HCM City Coordinates (District 1 Center)
CENTER_LAT = 10.762622
CENTER_LON = 106.660172

# ===========================
# HELPER FUNCTIONS
# ===========================
def get_destination_point(lat, lon, distance_km, bearing_degrees):
    """
    Calculates destination lat/lon given a starting point, distance, and bearing.
    Uses Haversine formula approximation.
    """
    R = 6371  # Earth Radius in km
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    bearing_rad = math.radians(bearing_degrees)
    
    new_lat_rad = math.asin(math.sin(lat_rad) * math.cos(distance_km / R) +
                            math.cos(lat_rad) * math.sin(distance_km / R) * math.cos(bearing_rad))
    
    new_lon_rad = lon_rad + math.atan2(math.sin(bearing_rad) * math.sin(distance_km / R) * math.cos(lat_rad),
                                       math.cos(distance_km / R) - math.sin(lat_rad) * math.sin(new_lat_rad))
    
    return math.degrees(new_lat_rad), math.degrees(new_lon_rad)

# ===========================
# GENERATORS
# ===========================
def generate_drivers(n=1000):
    print(f"Generating {n} drivers...")
    drivers = []
    for _ in range(n):
        driver_id = f"D_{str(uuid.uuid4())[:8]}"
        
        # Vehicle Distribution: 80% Bikes, 15% Tricycles (500), 5% Trucks (1000)
        v_type = np.random.choice(
            ["bike", "truck_500", "truck_1000"], 
            p=[0.80, 0.15, 0.05]
        )
        
        capacity = 30 if v_type == "bike" else (500 if v_type == "truck_500" else 1000)
        
        drivers.append({
            "driver_id": driver_id,
            "vehicle_type": v_type,
            "max_load_kg": capacity,
            "joined_date": START_DATE - timedelta(days=random.randint(0, 365))
        })
    return pd.DataFrame(drivers)

def generate_orders(n=10000):
    print(f"Generating {n} orders with Realistic Logic...")
    orders = []
    
    # Demand Curve (Probability per hour)
    # 00-05: Dead
    # 06-08: Morning Ramp
    # 09-11: Morning Rush
    # 12-13: Lunch Dip
    # 14-17: Afternoon Rush (E-commerce push)
    # 18-21: Evening Taper
    # 22-23: Dead
    hours_prob = [
        0.001, 0.001, 0.001, 0.001, 0.001, 0.005, 
        0.02, 0.08, 0.10,                         
        0.10, 0.10, 0.08,                         
        0.05, 0.05,                               
        0.08, 0.10, 0.12, 0.05,                   
        0.03, 0.02, 0.01, 0.001,                  
        0.0005, 0.0005                            
    ]
    hours_prob = np.array(hours_prob)
    hours_prob /= hours_prob.sum()

    for i in range(n):
        order_id = f"O_{str(uuid.uuid4())[:13]}"
        
        # 1. Time Simulation
        hour = np.random.choice(range(24), p=hours_prob)
        day_offset = random.randint(0, 30) # Simulate 1 month of data
        minute_offset = random.randint(0, 59)
        created_at = START_DATE + timedelta(days=day_offset)
        created_at = created_at.replace(hour=hour, minute=minute_offset, second=random.randint(0,59))

        # 2. User Needs
        req_vehicle = np.random.choice(["bike", "truck_500", "truck_1000"], p=[0.85, 0.10, 0.05])
        service = np.random.choice(["standard", "fast", "prioritize"], p=[0.6, 0.3, 0.1])
        
        # 3. Context
        # Improved Rain Logic (Probabilistic)
        # Base probability for any hour (Ho Chi Minh City is tropical!)
        rain_prob = 0.05 
        
        if 14 <= hour <= 19:
            # Peak Rain: Afternoon/Evening squalls
            rain_prob += 0.25  # Total ~30%
        elif 12 <= hour < 14:
            # Early afternoon buildup
            rain_prob += 0.15  # Total ~20%
        elif 20 <= hour <= 23:
            # Lingering night rain
            rain_prob += 0.05  # Total ~10%

        is_raining = 1 if random.random() < rain_prob else 0
        
        # 4. Geography (Consistent)
        pickup_lat = CENTER_LAT + np.random.normal(0, 0.03)
        pickup_lon = CENTER_LON + np.random.normal(0, 0.03)
        
        dist_km = np.random.lognormal(mean=1.0, sigma=0.6)
        dist_km = max(0.5, min(dist_km, 30.0)) # Clip 0.5km - 30km
        
        bearing = random.uniform(0, 360)
        dropoff_lat, dropoff_lon = get_destination_point(pickup_lat, pickup_lon, dist_km, bearing)
        
        # 5. Pricing (Commission included)
        base_price = 15000 if req_vehicle == "bike" else 130000 if req_vehicle == "truck_500" else 200000
        per_km = 5000 if req_vehicle == "bike" else 15000
        price = base_price + (dist_km * per_km)
        
        if service == "prioritize": price *= 3.0 # Urgent = Expensive
        if service == "fast": price *= 2.0
        if is_raining: price *= 1.3
            
        # 6. COD (Cash On Delivery)
        cod_mean = 500000 if req_vehicle == "bike" else 2000000
        cod = 0 if random.random() < 0.4 else np.random.exponential(cod_mean)

        orders.append({
            "order_id": order_id,
            "user_id": f"U_{random.randint(1, 10000)}",
            "created_at": created_at,
            "pickup_lat": pickup_lat,
            "pickup_lon": pickup_lon,
            "dropoff_lat": dropoff_lat,
            "dropoff_lon": dropoff_lon,
            "distance_km": round(dist_km, 2),
            "shipping_fee": round(price, -3),
            "cod_amount": round(cod, -3),
            "requested_vehicle_type": req_vehicle,
            "service_type": service,
            "is_raining": is_raining,
            "hour_of_day": hour
        })
        
    # Sort orders by time so simulation can track state correctly
    return pd.DataFrame(orders).sort_values("created_at").reset_index(drop=True)

def simulate_market_interactions(drivers_df, orders_df):
    """
    Simulates the Market.
    Includes 'Driver Busy State' logic to prevent unrealistic availability.
    """
    print("Simulating Market Interactions...")
    interactions = []
    
    # State Tracker: driver_id -> datetime (When they become free)
    # Init everyone as free at start of simulation
    driver_next_free_time = {d_id: START_DATE for d_id in drivers_df["driver_id"]}
    
    # Pre-index drivers by type
    drivers_by_type = drivers_df.groupby("vehicle_type")
    
    for _, order in orders_df.iterrows():
        req_type = order["requested_vehicle_type"]
        current_time = order["created_at"]
        
        if req_type not in drivers_by_type.groups:
            continue
            
        # 1. RETRIEVAL: Find Compatible & Available Drivers
        potential_drivers = drivers_by_type.get_group(req_type)
        
        # This checks the "Busy" state
        available_drivers = potential_drivers[
            potential_drivers["driver_id"].map(driver_next_free_time) <= current_time
        ]
        
        if len(available_drivers) == 0:
            continue # Market Saturation (No drivers available)
            
        # Pick 5 nearby candidates
        candidates = available_drivers.sample(min(len(available_drivers), 5))
        
        # 2. RANKING & DECISION
        for _, driver in candidates.iterrows():
            score = 0.0
            
            # --- Features impacting decision ---
            dist_to_pickup = np.random.uniform(0.1, 3.0)
            score -= (dist_to_pickup * 0.4) # Unpaid miles
            
            score += (order["shipping_fee"] / 20000) * 0.8 # Earnings
            
            # COD Friction
            cod_friction = order["cod_amount"] / 1000000
            score -= (cod_friction * (1.5 if req_type == "bike" else 0.5))
            
            # Rain & Bike
            if order["is_raining"] and req_type == "bike":
                score -= 2.0
                
            # Service Pressure
            if order["service_type"] == "prioritize":
                score -= 0.5 # Stressful
            elif order["service_type"] == "standard":
                score += 0.2 # Relaxed
                
            # Fatigue
            fatigue = np.random.beta(2, 5)
            if fatigue > 0.75 and order["distance_km"] > 15:
                score -= 2.0 # Too tired for long trip
                
            # --- Outcome ---
            prob = 1 / (1 + np.exp(-(score + 0.1)))
            is_accepted = 1 if random.random() < prob else 0
            
            interactions.append({
                "order_id": order["order_id"],
                "driver_id": driver["driver_id"],
                "driver_lat": order["pickup_lat"] + np.random.normal(0, 0.005),
                "driver_lon": order["pickup_lon"] + np.random.normal(0, 0.005),
                "driver_distance_to_pickup": round(dist_to_pickup, 2),
                "driver_fatigue_index": round(fatigue, 2),
                "is_accepted": is_accepted,
                "offered_at": current_time
            })
            
            # 3. UPDATE STATE
            if is_accepted:
                # Driver is now busy.
                # Duration = (Dist / 25kmh) + 15 mins pickup/dropoff
                duration_hours = (order["distance_km"] / 25.0) + 0.25
                driver_next_free_time[driver["driver_id"]] = current_time + timedelta(hours=duration_hours)
                break # Order filled
                
    return pd.DataFrame(interactions)

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    
    # Generate
    df_drivers = generate_drivers(NUM_DRIVERS)
    df_orders = generate_orders(NUM_ORDERS)
    df_interactions = simulate_market_interactions(df_drivers, df_orders)
    
    # Save
    df_drivers.to_csv("data/raw/drivers.csv", index=False)
    df_orders.to_csv("data/raw/orders.csv", index=False)
    df_interactions.to_csv("data/raw/interaction_logs.csv", index=False)
    
    print("\n✅ Simulation Complete!")
    print(f"Stats: {len(df_drivers)} Drivers, {len(df_orders)} Orders")
    print(f"Interactions: {len(df_interactions)}")
    print(f"Global Acceptance Rate: {df_interactions['is_accepted'].mean():.2%}")