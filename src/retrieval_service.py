import redis
import time
from typing import List, Dict, Optional

# Connection Pool (Best practice for production to reuse connections)
pool = redis.ConnectionPool(host='localhost', port=6379, db=0, decode_responses=True)
r = redis.Redis(connection_pool=pool)

class RetrievalService:
    def __init__(self):
        # Config: Radius steps in km (1km -> 3km -> 5km -> 10km)
        self.radius_steps = [1, 3, 5, 10]
        self.max_candidates = 100

    def find_candidates(self, lat: float, lon: float, vehicle_type: str) -> List[Dict]:
        """
        Retrieves candidate drivers using Dynamic Radius Expansion + Pipelined Feature Fetching.
        """
        start_time = time.time()
        
        # 1. Dynamic Radius Expansion (The "Net")
        candidate_ids = []
        search_key = f"drivers:geo:{vehicle_type}"
        
        for radius in self.radius_steps:
            candidate_ids = r.geosearch(
                name=search_key,
                longitude=lon,
                latitude=lat,
                radius=radius,
                unit="km",
                sort="ASC", # Get closest drivers first
                count=self.max_candidates
            )
            
            if len(candidate_ids) >= 5: # Threshold: If we found enough, stop expanding
                print(f"  > Found {len(candidate_ids)} drivers within {radius}km")
                break
            else:
                print(f"  > Only found {len(candidate_ids)} in {radius}km. Expanding...")

        if not candidate_ids:
            return []

        # 2. Feature Fetching with Pipelining
        candidates_data = []
        pipeline = r.pipeline()
        
        for driver_id in candidate_ids:
            # We need the 'state' hash (Dynamic features like fatigue, orders today)
            pipeline.hgetall(f"driver:{driver_id}:state")
            
            # Optional: Fetch 'profile' if you need static features too
            # pipeline.hgetall(f"driver:{driver_id}:profile")

        # Execute all 200 fetches in one go
        features_list = pipeline.execute()

        # 3. Merge IDs with Features
        for driver_id, features in zip(candidate_ids, features_list):
            if features: # Ensure driver key actually existed
                # Convert strings to proper types (Redis stores everything as string)
                features['driver_id'] = driver_id
                features['minutes_active'] = int(features.get('minutes_active', 0))
                features['fatigue_index'] = float(features.get('fatigue_index', 0.0))
                features['orders_completed'] = int(features.get('orders_completed', 0)) 
                features['cancel_rate'] = float(features.get('cancel_rate', 0.0))
                # Add simplified distance placeholder (real distance calc happens in ranking)
                candidates_data.append(features)

        latency_ms = (time.time() - start_time) * 1000
        print(f"✅ Retrieved {len(candidates_data)} candidates in {latency_ms:.2f}ms")
        
        return candidates_data

# --- Quick Test Block ---
if __name__ == "__main__":
    service = RetrievalService()
    
    # Mock Input: User at Ben Thanh Market, asking for a Bike
    print("--- 🔍 Requesting Bike in District 1 ---")
    drivers = service.find_candidates(10.7721, 106.6983, "bike")
    
    if drivers:
        print(f"Top Candidate: {drivers[0]}")