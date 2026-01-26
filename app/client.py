import os
import logging
import requests
import pandas as pd
import numpy as np
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# Metrics Integration (Prometheus)
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
from starlette.responses import Response

# Import our services
from src.retreival_engine.retrieval_service import RedisClient

# ==========================================
# 1. SETUP & CONFIG
# ==========================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DispatchService")

app = FastAPI(title="RouteMatch Dispatch Orchestrator")
redis_svc = RedisClient()

# Mount Prometheus ASGI app
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Configuration
MODEL_API_URL = os.getenv("MODEL_API_URL", "http://localhost:8000/predict/batch")

# ==========================================
# 2. PROMETHEUS METRICS DEFINITION
# ==========================================
# Counter: Total orders processed
METRIC_ORDER_COUNT = Counter('routematch_orders_total', 'Total incoming orders')

# Gauge: Current Drift Level (Example)
METRIC_DRIFT_PSI = Gauge('routematch_model_psi', 'Population Stability Index of Model')

# Histogram: Model Scores (to detect Concept Drift visualy)
METRIC_SCORE_DIST = Histogram('routematch_score_distribution', 'Distribution of ML scores', buckets=[0.1, 0.3, 0.5, 0.7, 0.9])

# Counter: Final Driver Decisions (Labels)
METRIC_DRIVER_RESPONSE = Counter('routematch_driver_response', 'Driver Accept/Reject', ['status'])

# ==========================================
# 3. REQUEST SCHEMA
# ==========================================
class OrderRequest(BaseModel):
    order_id: str
    user_id: str
    pickup_lat: float
    pickup_lon: float
    distance_km: float
    shipping_fee: float
    vehicle_type: str = "bike"
    service_type: str = "standard"
    is_raining: int = 0
    cod_amount: float = 0.0
    
    # Pre-calculated Time features (Client sends these usually)
    hour_sin: float
    hour_cos: float

# ==========================================
# 4. THE DISPATCH LOGIC (End-to-End Flow)
# ==========================================
@app.post("/order/submit")
def submit_order(order: OrderRequest):
    """
    1. Retrieve Candidates (Redis)
    2. Enrich Features (Redis)
    3. Score (ML API)
    4. Decide (Greedy)
    5. Simulate & Log
    """
    start_time = time.time()
    METRIC_ORDER_COUNT.inc()

    # Step A: Update Demand in Redis (Real-time Context)
    redis_svc.record_demand(order.pickup_lat, order.pickup_lon, order.order_id)
    
    # Step B: Get Candidate Drivers (Geo Search)
    candidates = redis_svc.get_candidate_drivers(order.pickup_lat, order.pickup_lon, order.vehicle_type)
    if not candidates:
        return {"status": "failed", "reason": "No drivers nearby"}

    # Step C: Enrich Candidates with Features
    batch_payload = []
    
    # Get H3 Demand once for the pickup location
    current_h3_demand = redis_svc.get_h3_demand(order.pickup_lat, order.pickup_lon)

    for cand in candidates:
        # Build the feature row matching 'app.py' schema
        feature_row = {
            "driver_id": cand['driver_id'],
            "order_id": order.order_id,
            
            # Context
            "distance_km": order.distance_km,
            "shipping_fee": order.shipping_fee,
            "requested_vehicle_type": order.vehicle_type,
            "service_type": order.service_type,
            "is_raining": order.is_raining,
            "hour_sin": order.hour_sin,
            "hour_cos": order.hour_cos,
            "h3_demand_60m": current_h3_demand,
            
            # Driver
            "driver_distance_to_pickup": cand['driver_distance_to_pickup'],
            "driver_fatigue_index": cand['driver_fatigue_index'],
            "driver_global_accept_rate": cand['driver_global_accept_rate'],
            "cod_amount": order.cod_amount
        }
        batch_payload.append(feature_row)

    # Step D: Call Model API
    try:
        response = requests.post(MODEL_API_URL, json={"requests": batch_payload})
        response.raise_for_status()
        predictions = response.json() # List of {driver_id, prob_accept}
    except Exception as e:
        logger.error(f"Model API failed: {e}")
        # Fallback: Random scores or distance-based
        return {"status": "error", "message": "Scoring Engine Unavailable"}

    # Step E: Make Decision (Greedy Strategy)
    # Sort by Probability Descending
    predictions.sort(key=lambda x: x['prob_accept'], reverse=True)
    best_driver_id = None
    best_prob = 0.0

    # Loop through candidates and try to acquire lock
    for candidate in predictions:
        candidate_driver_id = candidate['driver_id']
        if redis_svc.acquire_lock(candidate_driver_id, order.order_id):
            best_driver_id = candidate_driver_id
            best_prob = candidate['prob_accept']
            break
    else:
        # No drivers could be locked
        return {"status": "failed", "reason": "All drivers busy"}
        
    # Log distribution to Prometheus
    METRIC_SCORE_DIST.observe(best_prob)

    # Step G: Simulate Driver Response (Closing the Loop)
    best_match_record = next(item for item in batch_payload if item["driver_id"] == best_driver_id)
    did_accept = accepct_order_simulation(best_match_record, best_prob)
    
    # Step H: Update System State
    if did_accept:
        status = "accepted"
        redis_svc.update_driver_history(best_driver_id, accepted=True)
        METRIC_DRIVER_RESPONSE.labels(status='accepted').inc()
    else:
        status = "rejected"
        redis_svc.update_driver_history(best_driver_id, accepted=False)
        METRIC_DRIVER_RESPONSE.labels(status='rejected').inc()

    logger.info(f"Order {order.order_id} -> Driver {best_driver_id} (Score: {best_prob:.2f}) -> {status.upper()}")

    return {
        "status": status,
        "driver_id": best_driver_id,
        "score": best_prob,
        "processing_time": time.time() - start_time
    }

def accepct_order_simulation(candidate: dict, prob_accept: float) -> bool:
    """
    Simulates whether a driver accepts the order based on how we simulate the data at first.
    """
    score = 0.0
    # --- Features impacting decision ---
    dist_to_pickup = candidate["driver_distance_to_pickup"]
    req_type = candidate["requested_vehicle_type"]
    score -= (dist_to_pickup * 0.4) # Unpaid miles
    
    score += (candidate["shipping_fee"] / 20000) * 0.8 # Earnings
    
    # COD Friction
    cod_friction = candidate["cod_amount"] / 1000000
    score -= (cod_friction * (1.5 if req_type == "bike" else 0.5))
    
    # Rain & Bike
    if candidate["is_raining"] and req_type == "bike":
        score -= 2.0
        
    # Service Pressure
    if candidate["service_type"] == "prioritize":
        score -= 0.5 # Stressful
    elif candidate["service_type"] == "standard":
        score += 0.2 # Relaxed
        
    # Fatigue
    fatigue = candidate["driver_fatigue_index"]
    if fatigue > 0.75 and candidate["distance_km"] > 15:
        score -= 2.0 # Too tired for long trip
        
    # --- Outcome ---
    prob = 1 / (1 + np.exp(-(score + 0.1)))
    print(f"Simulated Score: {score:.2f} -> Prob: {prob:.2f}")
    is_accepted = 1 if np.random.random() < prob else 0

    return is_accepted