import uuid
from datetime import datetime
import random
import os
import math
import requests
import random
import time
import numpy as np

CENTER_LAT = 10.762622
CENTER_LON = 106.660172

ROUTEMATCH_DISPATCH_URL = os.getenv("ROUTEMATCH_DISPATCH_URL", "http://localhost:9000/order/submit")

MAX_ORDERS_PER_SEC = 1

if __name__ == "__main__":
   while True:
        num_orders = random.randint(0, MAX_ORDERS_PER_SEC)

        for _ in range(num_orders):
            order_id = str(uuid.uuid4())
            user_id = str(uuid.uuid4())
            vehicle_type = random.choice(["bike", "truck_500", "truck_1000"])
            service_type = random.choice(["standard", "fast", "prioritize"])
            
            # Random location within ~5km radius
            pickup_lat = CENTER_LAT + np.random.normal(0, 0.03)
            pickup_lon = CENTER_LON + np.random.normal(0, 0.03)
            
            distance_km = np.random.lognormal(mean=1.0, sigma=0.6)
            distance_km = max(0.5, min(distance_km, 30.0)) # Clip 0.5km - 30km

            # 5. Pricing (Commission included)
            base_price = 15000 if vehicle_type == "bike" else 130000 if vehicle_type == "truck_500" else 200000
            per_km = 5000 if vehicle_type == "bike" else 15000
            price = base_price + (distance_km * per_km)
            
            is_raining = random.choice([0, 1])
            if service_type == "prioritize": price *= 3.0 # Urgent = Expensive
            if service_type == "fast": price *= 2.0
            if is_raining: price *= 1.3 # Rain surcharge
            shipping_fee = int(price)
            
            now = datetime.utcnow()
            hour_sin = math.sin(2 * math.pi * now.hour / 24)
            hour_cos = math.cos(2 * math.pi * now.hour / 24)

            cod_mean = 500000 if vehicle_type == "bike" else 2000000
            cod = 0 if random.random() < 0.4 else np.random.exponential(cod_mean)
            
            order_payload = {
                "order_id": order_id,
                "user_id": user_id,
                "pickup_lat": pickup_lat,
                "pickup_lon": pickup_lon,
                "distance_km": distance_km,
                "shipping_fee": shipping_fee,
                "vehicle_type": vehicle_type,
                "service_type": service_type,
                "is_raining": is_raining,
                "cod_amount": cod,
                "hour_sin": hour_sin,
                "hour_cos": hour_cos,
            }
            
            try:
                response = requests.post(ROUTEMATCH_DISPATCH_URL, json=order_payload)
                if response.status_code == 200:
                    print(f"Dispatched Order {order_id} successfully.")
                else:
                    print(f"Failed to dispatch Order {order_id}. Status Code: {response.status_code}")
            except Exception as e:
                print(f"Error dispatching Order {order_id}: {e}")
            
        time.sleep(1)