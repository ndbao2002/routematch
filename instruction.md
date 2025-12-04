```bash
docker-compose up -d
    *Wait 10s for Postgres to initialize.*
```

```bash
uv run src/data_simulator.py
```

**Load DB (Manual Copy for now):**
```bash
docker cp data/raw/drivers.csv routematch_db:/tmp/drivers.csv
docker cp data/raw/orders.csv routematch_db:/tmp/orders.csv
docker cp data/raw/interaction_logs.csv routematch_db:/tmp/interaction_logs.csv

docker exec -i routematch_db psql -U postgres -d routematch -c "\copy drivers(driver_id, vehicle_type, max_load_kg, joined_date) FROM '/tmp/drivers.csv' DELIMITER ',' CSV HEADER;"
docker exec -i routematch_db psql -U postgres -d routematch -c "\copy orders(order_id, user_id, created_at, pickup_lat, pickup_lon, dropoff_lat, dropoff_lon, distance_km, shipping_fee, cod_amount, requested_vehicle_type, service_type, is_raining, hour_of_day) FROM '/tmp/orders.csv' DELIMITER ',' CSV HEADER;"
docker exec -i routematch_db psql -U postgres -d routematch -c "\copy interaction_logs(order_id, driver_id, driver_lat, driver_lon, driver_distance_to_pickup, driver_fatigue_index, is_accepted, offered_at) FROM '/tmp/interaction_logs.csv' DELIMITER ',' CSV HEADER;"
```

**Hydrate Redis:**
```bash
uv run src/hydrate_redis.py
```