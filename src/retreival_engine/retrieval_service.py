import os
import redis
import h3
import time
import logging
from typing import List, Dict

GLOBAL_MEAN_ACCEPT_RATE = 0.60
logger = logging.getLogger("RedisService")

class RedisClient:
    def __init__(self, host=None, port=6379, db=0):
        # CHANGE: Read from env if not passed explicitly
        if host is None:
            host = os.getenv("REDIS_HOST", "localhost")
        # Use Connection Pool for production-grade performance
        # decode_responses=True ensures we get Strings, not Bytes
        self.pool = redis.ConnectionPool(host=host, port=port, db=db, decode_responses=True)
        self.r = redis.Redis(connection_pool=self.pool)
        self.EXPIRY_DEMAND = 7200 # 2 hours ttl
        self.MAX_CANDIDATES = 100

    # =============================================
    # 1. GEOSPATIAL: Find Drivers (H3 k-Ring)
    # =============================================
    def get_candidate_drivers(self, lat: float, lon: float, vehicle_type: str = "bike") -> List[Dict]:
        """
        Retrieves candidate drivers using Dynamic H3 Ring Expansion.
        Replaces GEORADIUS to solve Boundary Problems and align with Feature Engineering (Res 8).
        """
        # 1. Dynamic Ring Expansion (The "Net")
        RESOLUTION = 8
        h3_index_center = h3.latlng_to_cell(lat, lon, RESOLUTION)
        candidate_ids = []
        
        # Expand k-ring from 0 to 5 (~3-5km radius depending on k)
        for k in range(0, 6):
            h3_indexes = h3.grid_ring(h3_index_center, k)
            for h3_index in h3_indexes:
                search_key = f"drivers:h3:{h3_index}:{vehicle_type}"
                
                # Fetch top N drivers from this cell
                candidate_ids.extend(self.r.zrange(search_key, 0, self.MAX_CANDIDATES - 1, withscores=False))
                
            if len(candidate_ids) >= 25:
                break

        if not candidate_ids:
            return []

        # 2. Pipelined Feature Fetching
        pipeline = self.r.pipeline()
        for driver_id in candidate_ids:
            # We need the 'state' hash (Dynamic features like fatigue, orders today)
            pipeline.hgetall(f"driver:{driver_id}:state")

        # Execute all n fetches in one go
        features_list = pipeline.execute()

        # 3. Merge & Calculate
        candidates_data = []
        for driver_id, features in zip(candidate_ids, features_list):
            if not features: continue # Skip if driver key missing/expired
            
            # Map Redis Strings to Floats/Ints & Calculate Distance
            try:
                d_lat = float(features.get('lat', 0.0))
                d_lon = float(features.get('lon', 0.0))
                
                dist_km = h3.great_circle_distance((lat, lon), (d_lat, d_lon), unit='km')
                
                candidates_data.append({
                    "driver_id": driver_id,
                    # Geospatial
                    "driver_lat": d_lat,
                    "driver_lon": d_lon,
                    "driver_distance_to_pickup": dist_km,
                    
                    # Features for Model
                    "driver_fatigue_index": float(features.get('driver_fatigue_index', 0.0)),
                    "driver_global_accept_rate": float(features.get('driver_global_accept_rate', GLOBAL_MEAN_ACCEPT_RATE)),
                    "cod_amount": float(features.get('cod_amount', 0.0))
                })
            except ValueError:
                continue # Skip malformed data

        return candidates_data

    # =============================================
    # 2. FEATURE RETRIEVAL: H3 Demand
    # =============================================
    def get_h3_demand(self, lat: float, lon: float):
        """
        Calculates demand in the H3 cell (Resolution 8) for the last 60 mins.
        """
        RESOLUTION = 8 # Updated to match Retrieval
        h3_index = h3.latlng_to_cell(lat, lon, RESOLUTION)
        key = f"demand:h3:{h3_index}"
        
        now = time.time()
        window_start = now - 3600
        
        pipe = self.r.pipeline()
        pipe.zremrangebyscore(key, '-inf', window_start) # Clean old data
        pipe.zcard(key) # Count remaining
        results = pipe.execute()
        
        return float(results[1])

    # =============================================
    # 3. STATE UPDATES (Closing the Loop)
    # =============================================
    def update_driver_history(self, driver_id: str, accepted: bool, C: int = 20):
        """
        Updates Driver Stats (Bayesian Rate).
        """
        key = f"driver:{driver_id}:state"
        
        # Get current state
        current = self.r.hmget(key, ["total_offers", "total_accepts"])
        total_offers = float(current[0]) if current[0] else 0
        total_accepts = float(current[1]) if current[1] else 0
        
        total_offers += 1
        if accepted:
            total_accepts += 1
            
        # Recalculate Rate (Bayesian Smoothing)
        new_rate = (total_accepts + GLOBAL_MEAN_ACCEPT_RATE * C) / (total_offers + C)
        
        self.r.hset(key, mapping={
            "status": "BUSY" if accepted else "IDLE",
            "total_offers": total_offers,
            "total_accepts": total_accepts,
            "driver_global_accept_rate": new_rate
        })


    def record_demand(self, lat: float, lon: float, order_id: str):
        """
        Records order in H3 ZSET for future demand queries.
        """
        RESOLUTION = 8
        h3_index = h3.latlng_to_cell(lat, lon, RESOLUTION)
        key = f"demand:h3:{h3_index}"
        now = time.time()
        
        pipe = self.r.pipeline()
        pipe.zadd(key, {order_id: now})
        pipe.expire(key, self.EXPIRY_DEMAND)
        pipe.execute()

    # =============================================
    # 4. DISTRIBUTED LOCKING (Redlock)
    # =============================================
    def acquire_lock(self, driver_id: str, order_id: str, ttl=30) -> bool:
        """
        Attempts to lock a driver. Returns True if successful.
        """
        key = f"lock:driver:{driver_id}"
        # SET NX (Not Exists) EX (Expire)
        return self.r.set(key, order_id, nx=True, ex=ttl)