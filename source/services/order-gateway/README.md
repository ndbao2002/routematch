# Order Gateway Service

Stateless HTTP edge service for order submission and status polling.
Owns Redis status cache updates via the sync worker and publishes OrderRequested events.


Run locally (uses Redis at localhost:6379 by default):

```bash
cd source/services/order-gateway
uvicorn app.api:app --host 0.0.0.0 --port 8080
```

Docker (build image then run) using `uv` to manage dependencies:

```bash
cd source/services/order-gateway
docker build -t routematch/order-gateway .

# Recommended (run the service on the same Docker network as the redis container):
docker run --network routematch-net -e REDIS_HOST=routematch_ms_redis -p 8080:8080 routematch/order-gateway

# If you are running Redis on the Docker host (Docker Desktop), you can use:
# docker run -e REDIS_HOST=host.docker.internal -p 8080:8080 routematch/order-gateway
```

Notes:
- The service stores a short-lived Redis key `order:status:{order_id}` with TTL configured by `REDIS_TTL`.
- If `PANDAPROXY_URL` is set (e.g. `http://kafka:18082`), the service will attempt to publish `Orders_Requested` events.
- Startup and shutdown are handled with FastAPI lifespan, not `app.on_event`.
- On a Redis cache miss, `GET /order/status/{order_id}` falls back to Postgres using the `orders` table and latest `interaction_logs` row.
