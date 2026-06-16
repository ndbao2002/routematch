import math
import os
import uuid
import logging
import time
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import redis
import grpc

from app.config import Config
from pb import driver_state_service_pb2 as pb2
from pb import driver_state_service_pb2_grpc as pb2_grpc

logger = logging.getLogger("ml-batch-optimizer")

def calculate_distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0  # Earth radius in meters
    rad_lat1 = math.radians(lat1)
    rad_lat2 = math.radians(lat2)
    diff_lat = math.radians(lat2 - lat1)
    diff_lon = math.radians(lon2 - lon1)
    
    a = (math.sin(diff_lat / 2) ** 2 +
         math.cos(rad_lat1) * math.cos(rad_lat2) * (math.sin(diff_lon / 2) ** 2))
    c = 2 * math.asin(math.sqrt(a))
    return R * c

class MatchOptimizer:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            socket_timeout=2.0
        )
        self.grpc_channel = grpc.insecure_channel(Config.DRIVER_STATE_GRPC_ADDR)
        self.grpc_stub = pb2_grpc.DriverStateServiceStub(self.grpc_channel)
        self.model = self._load_model()

    def _load_model(self):
        # 1. Try to load from MLflow Registry
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
            mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
            
            # Try to query latest versions dynamically via MlflowClient
            try:
                client = MlflowClient()
                versions = client.get_latest_versions(Config.MODEL_NAME, stages=["None", "Production", "Staging"])
                if versions:
                    # Sort versions descending (newest first)
                    versions.sort(key=lambda x: int(x.version), reverse=True)
                    for v in versions:
                        try:
                            uri = f"models:/{Config.MODEL_NAME}/{v.version}"
                            logger.info("Attempting to load model from MLflow registry: %s", uri)
                            model = mlflow.sklearn.load_model(uri)
                            logger.info("Successfully loaded model from MLflow registry version %s!", v.version)
                            return model
                        except Exception as inner_e:
                            logger.debug("Could not load model version %s: %s", v.version, inner_e)
            except Exception as client_e:
                logger.warning("Failed to retrieve latest versions via MlflowClient: %s", client_e)

            # Fallback to static list of stages/versions if dynamic check failed
            for stage in ["Production", "latest", "3", "2", "1"]:
                try:
                    uri = f"models:/{Config.MODEL_NAME}/{stage}"
                    logger.info("Attempting to load model from MLflow registry fallback: %s", uri)
                    model = mlflow.sklearn.load_model(uri)
                    logger.info("Successfully loaded model from MLflow registry fallback %s!", stage)
                    return model
                except Exception as e:
                    logger.debug("Could not load from registry Stage/Version %s: %s", stage, e)
        except Exception as e:
            logger.warning("Failed to connect/query MLflow tracking server: %s", e)

        # 2. Try loading local fallback
        local_paths = [
            "models/xgb_scoring_model.pkl",
            "/app/models/xgb_scoring_model.pkl",
            "../models/xgb_scoring_model.pkl",
            "../../models/xgb_scoring_model.pkl"
        ]
        for path in local_paths:
            if os.path.exists(path):
                try:
                    import joblib
                    logger.info("Loading local fallback model from: %s", path)
                    model = joblib.load(path)
                    logger.info("Successfully loaded local fallback model!")
                    return model
                except Exception as e:
                    logger.warning("Failed to load local model at %s: %s", path, e)
        
        raise RuntimeError("No model found. MLflow registry is unavailable and local pkl files are missing.")

    def get_rejected_drivers(self, order_id: str) -> List[str]:
        try:
            key = f"order:rejected:{order_id}"
            members = self.redis_client.smembers(key)
            return [m.decode("utf-8") if isinstance(m, bytes) else m for m in members]
        except Exception as e:
            logger.error("Redis failed to retrieve rejected drivers for %s: %s", order_id, e)
            return []

    def get_h3_demand(self, lat: float, lon: float) -> float:
        try:
            import h3
            RESOLUTION = 8
            h3_index = h3.latlng_to_cell(lat, lon, RESOLUTION)
            key = f"demand:h3:{h3_index}"
            now = time.time()
            window_start = now - 3600
            
            pipe = self.redis_client.pipeline()
            pipe.zremrangebyscore(key, '-inf', window_start)
            pipe.zcard(key)
            results = pipe.execute()
            return float(results[1])
        except Exception as e:
            logger.error("Redis failed to get H3 demand: %s", e)
            return 0.0

    def record_demand(self, lat: float, lon: float, order_id: str):
        try:
            import h3
            RESOLUTION = 8
            h3_index = h3.latlng_to_cell(lat, lon, RESOLUTION)
            key = f"demand:h3:{h3_index}"
            now = time.time()
            
            pipe = self.redis_client.pipeline()
            pipe.zadd(key, {order_id: now})
            pipe.expire(key, 3600)
            pipe.execute()
        except Exception as e:
            logger.error("Redis failed to record demand for order %s: %s", order_id, e)

    def optimize_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Executes Bipartite Matching for a batch of Orders.
        """
        if not batch:
            return {}

        # 1. Group orders by vehicle type
        grouped_orders: Dict[str, List[Dict[str, Any]]] = {}
        for order in batch:
            vt = order.get("requested_vehicle_type", "bike")
            grouped_orders.setdefault(vt, []).append(order)
            
            # Record demand dynamically
            self.record_demand(order["pickup_lat"], order["pickup_lon"], order["order_id"])

        assignments = []
        batch_id = str(uuid.uuid4())

        # 2. Optimize each vehicle group independently
        for vehicle_type, group in grouped_orders.items():
            try:
                group_assignments = self._optimize_group(vehicle_type, group)
                assignments.extend(group_assignments)
            except Exception as e:
                logger.exception("Failed to optimize group %s: %s", vehicle_type, e)

        if not assignments:
            return {}

        return {
            "batch_id": batch_id,
            "assignments": assignments
        }

    def _optimize_group(self, vehicle_type: str, orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # A. Gather rejections and build gRPC request queries
        queries = []
        order_map = {}
        h3_demands = {}

        for order in orders:
            oid = order["order_id"]
            order_map[oid] = order
            rejected_ids = self.get_rejected_drivers(oid)
            
            # Cache H3 demand per order location
            h3_demands[oid] = self.get_h3_demand(order["pickup_lat"], order["pickup_lon"])

            queries.append(pb2.OrderSearchQuery(
                order_id=oid,
                pickup_lat=order["pickup_lat"],
                pickup_lon=order["pickup_lon"],
                search_radius_meters=3000.0, # 3km search radius
                rejected_driver_ids=rejected_ids
            ))

        # B. Call BatchGetCandidates gRPC RPC
        try:
            grpc_req = pb2.BatchGetCandidatesRequest(
                correlation_id=str(uuid.uuid4()),
                requested_vehicle_type=vehicle_type,
                max_candidates=150, # Retrieve sufficient candidates
                orders=queries
            )
            grpc_resp = self.grpc_stub.BatchGetCandidates(grpc_req, timeout=3.0)
            if grpc_resp.status == "error":
                logger.error("gRPC BatchGetCandidates returned error: %s", grpc_resp.message)
                return []
        except Exception as e:
            logger.error("gRPC call to driver-state failed: %s", e)
            return []

        candidates = grpc_resp.candidates
        if not candidates:
            logger.info("No candidates found for vehicle type: %s", vehicle_type)
            return []

        # C. Filter pairs and build feature dataframe for inference
        eligible_pairs = []
        feature_rows = []

        for order in orders:
            oid = order["order_id"]
            rejected_ids = set(self.get_rejected_drivers(oid))
            
            for cand in candidates:
                # 1. Skip if driver rejected
                if cand.driver_id in rejected_ids:
                    continue
                
                # 2. Check radius
                dist_m = calculate_distance_meters(
                    order["pickup_lat"], order["pickup_lon"],
                    cand.driver_lat, cand.driver_lon
                )
                if dist_m > 3000.0:
                    continue

                dist_km = dist_m / 1000.0
                attempt_count = int(order.get("attempt_count", 1))

                # Escalation logic: pricing bonus for attempt > 2
                shipping_fee = float(order["shipping_fee"])
                if attempt_count > 2:
                    shipping_fee += 10000.0 # Add 10,000 VND bonus to fee

                price_per_km = shipping_fee / (float(order["distance_km"]) + 0.01)
                
                # Temporal encoding
                hour_of_day = int(order.get("hour_of_day", 12))
                hour_sin = math.sin(2 * math.pi * hour_of_day / 24)
                hour_cos = math.cos(2 * math.pi * hour_of_day / 24)

                # Default cold start parameters
                accept_rate = cand.accept_rate if cand.accept_rate > 0.0 else 0.35
                fatigue = cand.fatigue_index if cand.fatigue_index >= 0.0 else 0.0

                # Form row matching the expected features
                row = {
                    "cod_amount": float(order["cod_amount"]),
                    "distance_km": float(order["distance_km"]),
                    "driver_distance_to_pickup": dist_km,
                    "driver_fatigue_index": fatigue,
                    "driver_global_accept_rate": accept_rate,
                    "h3_demand_60m": h3_demands[oid],
                    "hour_cos": hour_cos,
                    "hour_sin": hour_sin,
                    "is_raining": 1 if order.get("is_raining") in [True, 1, "1"] else 0,
                    "price_per_km": price_per_km,
                    "requested_vehicle_type": vehicle_type,
                    "service_type": order["service_type"],
                    "shipping_fee": shipping_fee
                }

                eligible_pairs.append((oid, cand.driver_id, dist_km, attempt_count))
                feature_rows.append(row)

        if not eligible_pairs:
            return []

        # D. Batch Model Inference
        df_features = pd.DataFrame(feature_rows)
        # Reindex features alphabetically to align with the training script
        df_features = df_features.reindex(sorted(df_features.columns), axis=1)

        try:
            probabilities = self.model.predict_proba(df_features)[:, 1]
        except Exception as e:
            logger.error("Model prediction failed: %s", e)
            return []

        # E. Construct Hungarian Cost Matrix
        # Map orders and drivers to indices
        unique_order_ids = list(order_map.keys())
        unique_driver_ids = list(set(cand.driver_id for cand in candidates))
        
        order_to_idx = {oid: idx for idx, oid in enumerate(unique_order_ids)}
        driver_to_idx = {did: idx for idx, did in enumerate(unique_driver_ids)}

        num_orders = len(unique_order_ids)
        num_drivers = len(unique_driver_ids)
        
        # Initialize matrix with large penalty
        cost_matrix = np.full((num_orders, num_drivers), 1e6)
        prob_matrix = np.zeros((num_orders, num_drivers))

        for (oid, did, dist_km, attempt_count), prob in zip(eligible_pairs, probabilities):
            o_idx = order_to_idx[oid]
            d_idx = driver_to_idx[did]
            
            # Base cost: w1 * (1 - P(Accept)) + w2 * distance_to_pickup
            cost = Config.WEIGHT_PROB * (1.0 - prob) + Config.WEIGHT_DISTANCE * dist_km
            
            # Escalation logic: Age priority penalty for attempt > 1
            if attempt_count > 1:
                cost -= 2.0 * (attempt_count - 1)

            cost_matrix[o_idx, d_idx] = cost
            prob_matrix[o_idx, d_idx] = prob

        # F. Hungarian Optimization
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assignments = []
        for r, c in zip(row_ind, col_ind):
            cost = cost_matrix[r, c]
            if cost >= 1e5:  # Skip invalid/ineligible pairs
                continue
                
            oid = unique_order_ids[r]
            did = unique_driver_ids[c]
            prob = prob_matrix[r, c]
            order = order_map[oid]

            assignments.append({
                "order_id": oid,
                "driver_id": did,
                "score": float(prob),
                "attempt_count": int(order.get("attempt_count", 1)),
                "rank": 1
            })

        return assignments
