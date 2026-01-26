import logging
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RouteMatchAPI")

app = FastAPI(
    title="RouteMatch Dispatch API",
    description="Real-time Scoring Engine for Order Dispatching",
    version="1.0.0"
)

# Global model store
model_pipeline = None

# Ordered columns used during training
TRAINING_COLUMNS = ['cod_amount', 'distance_km', 'driver_distance_to_pickup', 'driver_fatigue_index', 
                    'driver_global_accept_rate', 'driver_id', 'h3_demand_60m', 'hour_cos', 'hour_sin', 
                    'is_raining', 'order_id', 'price_per_km', 'requested_vehicle_type', 
                    'service_type', 'shipping_fee']

class DispatchRequest(BaseModel):
    """
    Represents a single candidate pair (Driver + Order context).
    """
    # Identifier
    driver_id: str = Field(..., description="Unique Driver Identifier")
    order_id: str = Field(..., description="Unique Order Identifier")

    # Context Features (Order)
    distance_km: float = Field(..., gt=0, description="Trip distance in km")
    shipping_fee: float = Field(..., gt=0, description="Shipping fee in VND")
    requested_vehicle_type: str = Field(..., description="'bike' or 'truck_500' or 'truck_1000'")
    service_type: str = Field(..., description="'standard' or 'fast' or 'prioritize'")
    is_raining: int = Field(..., ge=0, le=1, description="1 if raining, 0 else")
    
    # Real-time Context (Time/Space)
    hour_sin: float = Field(..., ge=-1, le=1)
    hour_cos: float = Field(..., ge=-1, le=1)
    h3_demand_60m: float = Field(..., ge=0, description="Orders in this cell last hour")
    
    # Driver Features
    driver_distance_to_pickup: float = Field(..., ge=0, description="Distance from driver to pickup in km")
    driver_fatigue_index: float = Field(..., ge=0, le=1, description="0=Fresh, 1=Exhausted")
    driver_global_accept_rate: float = Field(..., ge=0, le=1, description="Bayesian smoothed acceptance rate")
    cod_amount: float = Field(0, ge=0, description="Cash on Delivery amount")

class DispatchResponse(BaseModel):
    driver_id: str
    prob_accept: float

class BatchRequest(BaseModel):
    # We include IDs to map responses back to requests
    requests: List[DispatchRequest] # List of dictionaries containing 'driver_id' + fields above

# ==========================================
# 1. LIFECYCLE MANAGEMENT
# ==========================================

@app.on_event("startup")
def load_model():
    """
    Loads the model once when the server starts.
    Prevents high latency on first request (Cold Start).
    """
    global model_pipeline
    model_path = "models/xgb_scoring_model.pkl"
    try:
        logger.info(f"Loading model from {model_path}...")
        model_pipeline = joblib.load(model_path)
        logger.info("✅ Model loaded successfully!")
    except FileNotFoundError:
        logger.error(f"❌ Model file not found at {model_path}. Please run training first.")
        raise RuntimeError("Model missing")

# ==========================================
# 2. ENDPOINTS
# ==========================================

@app.get("/health")
def health_check():
    """K8s Liveness Probe"""
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_version": "v1"}

@app.post("/predict/batch", response_model=List[DispatchResponse])
def predict_batch(payload: BatchRequest):
    """
    Scores a batch of candidate drivers for an order.
    Optimized for vectorization (Pandas).
    """
    if not model_pipeline:
        raise HTTPException(status_code=503, detail="Model not ready")
    
    try:
        # 1. Parse Input
        # Convert list of dicts to DataFrame directly
        df_batch = pd.DataFrame([req.model_dump() for req in payload.requests])
        
        if df_batch.empty:
            return []

        # 2. Dynamic Feature Engineering (On-the-fly)
        # Replicate logic from feature_engineering.py
        # Avoid division by zero safety check
        df_batch['price_per_km'] = df_batch['shipping_fee'] / (df_batch['distance_km'] + 1e-8)
        
        # 3. Validation
        # Ensure strict column ordering matches training
        assert sorted(df_batch.columns) == sorted(TRAINING_COLUMNS), "Missing columns or extra columns in input data"
        df_batch = df_batch.reindex(columns=TRAINING_COLUMNS)

        # Drop order_id before prediction
        df_batch = df_batch.drop(columns=['order_id'], errors='ignore')
        
        # 4. Inference
        # predict_proba returns [P(0), P(1)]. We want P(1).
        probs = model_pipeline.predict_proba(df_batch)[:, 1]
        
        # 5. Format Response
        results = []
        for i, row in df_batch.iterrows():
            results.append({
                "driver_id": row.get("driver_id", f"unknown_{i}"),
                "prob_accept": float(probs[i])
            })
            
        return results

    except Exception as e:
        logger.error(f"Inference Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))