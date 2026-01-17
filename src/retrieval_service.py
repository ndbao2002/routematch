import redis
import time
from typing import List, Dict, Optional
import h3

# Connection Pool (Best practice for production to reuse connections)
pool = redis.ConnectionPool(host='localhost', port=6379, db=0, decode_responses=True)
r = redis.Redis(connection_pool=pool)

class RetrievalService:
    def __init__(self):
        self.max_candidates = 100

    def find_candidates(self, lat: float, lon: float, vehicle_type: str) -> List[Dict]:
        """
        Retrieves candidate drivers using Dynamic Ring Expansion + Pipelined Feature Fetching.
        """
        start_time = time.time()
        
        # 1. Dynamic Ring Expansion (The "Net")
        # using H3 k-rings for geo-indexing
        RESOLUTION = 8
        h3_index_center = h3.latlng_to_cell(lat, lon, RESOLUTION)  # Resolution 8
        candidate_ids = []
        
        for k in range(0, 6):  # Expand k-ring from 0 to 5
            # Get H3 indexes in the current k
            h3_indexes = h3.grid_ring(h3_index_center, k)
            for h3_index in h3_indexes:
                search_key = f"drivers:h3:{h3_index}:{vehicle_type}"

                candidate_ids.extend(r.zrange(search_key, 0, self.max_candidates - 1, withscores=False))
                
            if len(candidate_ids) >= 5: # Threshold: If we found enough, stop expanding
                print(f"  > Found {len(candidate_ids)} drivers within {k}-ring.")
                break
            else:
                print(f"  > Only found {len(candidate_ids)} in {k}-ring. Expanding...")

        if not candidate_ids:
            return []

        # 2. Feature Fetching with Pipelining
        candidates_data = []
        pipeline = r.pipeline()
        
        for driver_id in candidate_ids:
            # We need the 'state' hash (Dynamic features like fatigue, orders today)
            pipeline.hgetall(f"driver:{driver_id}:state")
            
            # Optional: Fetch 'profile' if need static features as well
            # pipeline.hgetall(f"driver:{driver_id}:profile")

        # Execute all n fetches in one go
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
                features['lat'] = float(features.get('lat', 0.0))
                features['lon'] = float(features.get('lon', 0.0))
                features['distance_km'] = h3.great_circle_distance((lat, lon), (features['lat'], features['lon']), unit='km')
                # Add to final list
                candidates_data.append(features)

        latency_ms = (time.time() - start_time) * 1000
        print(f"Retrieved {len(candidates_data)} candidates in {latency_ms:.2f}ms")
        
        return candidates_data

# --- Quick Test Block ---
if __name__ == "__main__":
    service = RetrievalService()
    
    # Mock Input: User at Ben Thanh Market, asking for a Bike
    print("--- Requesting Bike in District 1 ---")
    drivers = service.find_candidates(10.7721, 106.6983, "bike")
    
    if drivers:
        print(f"Top Candidate: {drivers[0]}")