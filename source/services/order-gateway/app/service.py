from __future__ import annotations

import asyncio
import os
import json
import logging
from typing import Optional

import asyncpg
import redis.asyncio as aioredis
import requests

LOG = logging.getLogger("order_gateway.service")
logging.basicConfig(level=logging.INFO)

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_TTL = int(os.getenv("REDIS_TTL", "300"))
PANDAPROXY_URL = os.getenv("PANDAPROXY_URL")
DB_HOST = os.getenv("DB_HOST", "db")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "routematch")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")


class ServiceState:
    redis_client: Optional[aioredis.Redis] = None
    db_pool: Optional[asyncpg.Pool] = None


state = ServiceState()


async def init_redis() -> None:
    state.redis_client = aioredis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    LOG.info("Initialized Redis client %s:%s", REDIS_HOST, REDIS_PORT)


async def close_redis() -> None:
    if state.redis_client:
        await state.redis_client.close()
        LOG.info("Closed Redis client")


async def init_db() -> None:
    state.db_pool = await asyncpg.create_pool(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        min_size=1,
        max_size=5,
    )
    LOG.info("Initialized Postgres pool %s:%s/%s", DB_HOST, DB_PORT, DB_NAME)


async def close_db() -> None:
    if state.db_pool:
        await state.db_pool.close()
        LOG.info("Closed Postgres pool")


def publish_event(topic: str, payload: dict) -> None:
    if not PANDAPROXY_URL:
        LOG.info("[event:%s] %s", topic, payload)
        return

    url = f"{PANDAPROXY_URL.rstrip('/')}/v1/produce?topic={topic}"
    body = {"records": [{"value": payload}]}
    try:
        res = requests.post(url, json=body, timeout=2)
        res.raise_for_status()
        LOG.info("Published event to %s", topic)
    except Exception as e:
        LOG.warning("Failed to publish to Pandaproxy: %s", e)


async def set_order_status(order_id: str, status: dict) -> None:
    key = f"order:status:{order_id}"
    client = state.redis_client
    if not client:
        raise RuntimeError("Redis client not initialized")
    await client.setex(key, REDIS_TTL, json.dumps(status))


async def create_order_record(
    order_id: str,
    user_id: str,
    pickup_lat: float,
    pickup_lon: float,
    dropoff_lat: float,
    dropoff_lon: float,
    requested_vehicle_type: str,
    service_type: str,
    cod_amount: float,
    shipping_fee: float,
    distance_km: Optional[float],
    is_raining: bool,
    hour_of_day: int
) -> None:
    pool = state.db_pool
    if not pool:
        raise RuntimeError("Database pool not initialized")

    query = """
        INSERT INTO orders (
            order_id,
            user_id,
            created_at,
            status,
            status_updated_at,
            pickup_lat,
            pickup_lon,
            dropoff_lat,
            dropoff_lon,
            shipping_fee,
            cod_amount,
            distance_km,
            requested_vehicle_type,
            service_type,
            is_raining,
            hour_of_day
        ) VALUES (
            $1, $2, NOW(), $3, NOW(), $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14
        )
        ON CONFLICT (order_id) DO NOTHING
    """
    async with pool.acquire() as conn:
        await conn.execute(
            query,
            order_id,
            user_id,
            "accumulating",
            pickup_lat,
            pickup_lon,
            dropoff_lat,
            dropoff_lon,
            shipping_fee,
            cod_amount,
            distance_km,
            requested_vehicle_type,
            service_type,
            is_raining,
            hour_of_day
        )


async def load_order_status_from_db(order_id: str) -> Optional[dict]:
    pool = state.db_pool
    if not pool:
        raise RuntimeError("Database pool not initialized")

    query = """
        SELECT
            o.status,
            o.status_updated_at,
            COALESCE(il.attempt_count, 1) AS attempt_count,
            il.driver_id
        FROM orders o
        LEFT JOIN LATERAL (
            SELECT driver_id, attempt_count
            FROM interaction_logs
            WHERE order_id = o.order_id
            ORDER BY attempt_count DESC, interaction_id DESC
            LIMIT 1
        ) il ON TRUE
        WHERE o.order_id = $1
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, order_id)
    if not row:
        return None
    return {
        "status": row["status"],
        "attempt": row["attempt_count"],
        "driver_id": row["driver_id"],
        "source": "database",
    }


async def get_order_status(order_id: str) -> Optional[dict]:
    key = f"order:status:{order_id}"
    client = state.redis_client
    if not client:
        raise RuntimeError("Redis client not initialized")
    raw = await client.get(key)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return {"status": "unknown"}
