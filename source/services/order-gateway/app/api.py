from __future__ import annotations

from contextlib import asynccontextmanager, suppress
import uuid
import logging
import asyncio

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .service import (
    init_redis,
    close_redis,
    init_db,
    close_db,
    set_order_status,
    get_order_status,
    publish_event,
    create_order_record,
    load_order_status_from_db,
)
from .worker import run_sync_worker

from prometheus_fastapi_instrumentator import Instrumentator

LOG = logging.getLogger("order_gateway.api")
logging.basicConfig(level=logging.INFO)

@asynccontextmanager
async def lifespan(_: FastAPI):
    await init_redis()
    await init_db()
    
    consumer_task = asyncio.create_task(run_sync_worker())
    
    try:
        yield
    finally:
        consumer_task.cancel()
        with suppress(asyncio.CancelledError):
            await consumer_task
            
        await close_db()
        await close_redis()


app = FastAPI(title="RouteMatch - Order Gateway", lifespan=lifespan)
Instrumentator().instrument(app).expose(app)


class OrderSubmit(BaseModel):
    user_id: str
    pickup_lat: float
    pickup_lon: float
    dropoff_lat: float
    dropoff_lon: float
    requested_vehicle_type: str = Field(..., pattern="^(bike|truck_500|truck_1000)$")
    service_type: str = Field(..., pattern="^(standard|fast|prioritize)$")
    cod_amount: float = 0.0
    shipping_fee: float = 0.0
    distance_km: float = None
    is_raining: bool = False
    hour_of_day: int = Field(..., ge=0, le=23)


@app.post("/order/submit", status_code=202)
async def submit_order(order: OrderSubmit):
    order_id = uuid.uuid4().hex[:20]
    state = {"status": "accumulating", "attempt": 1}
    await set_order_status(order_id, state)
    await create_order_record(
        order_id=order_id,
        user_id=order.user_id,
        pickup_lat=order.pickup_lat,
        pickup_lon=order.pickup_lon,
        dropoff_lat=order.dropoff_lat,
        dropoff_lon=order.dropoff_lon,
        requested_vehicle_type=order.requested_vehicle_type,
        service_type=order.service_type,
        cod_amount=order.cod_amount,
        shipping_fee=order.shipping_fee,
        distance_km=order.distance_km,
        is_raining=order.is_raining,
        hour_of_day=order.hour_of_day
    )

    event = {
        "order_id": order_id,
        "user_id": order.user_id,
        "pickup_lat": order.pickup_lat,
        "pickup_lon": order.pickup_lon,
        "dropoff_lat": order.dropoff_lat,
        "dropoff_lon": order.dropoff_lon,
        "requested_vehicle_type": order.requested_vehicle_type,
        "service_type": order.service_type,
        "cod_amount": order.cod_amount,
        "shipping_fee": order.shipping_fee,
        "distance_km": order.distance_km,
        "is_raining": order.is_raining,
        "hour_of_day": order.hour_of_day,
        "attempt_count": 1,
    }
    # best-effort
    publish_event("Orders_Requested", event)
    return {"order_id": order_id, "message": "Finding a driver..."}


@app.get("/order/status/{order_id}")
async def get_status(order_id: str):
    data = await get_order_status(order_id)
    if not data:
        data = await load_order_status_from_db(order_id)
    if not data:
        raise HTTPException(status_code=404, detail="Order not found")
    return data
