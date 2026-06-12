import os
import json
import asyncio
import logging
from aiokafka import AIOKafkaConsumer
from .service import set_order_status

LOG = logging.getLogger("order_gateway.worker")

KAFKA_BROKERS = os.getenv("KAFKA_BROKERS", "localhost:19092")
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "order-gateway-sync-worker")

async def run_sync_worker():
    LOG.info("Starting State Sync Worker...")
    topics = ["Orders_Requested", "Offers_Created", "Orders_Finalized", "Orders_Failed"]
    
    consumer = None
    while True:
        try:
            consumer = AIOKafkaConsumer(
                *topics,
                bootstrap_servers=KAFKA_BROKERS,
                group_id=KAFKA_GROUP_ID,
                value_deserializer=lambda m: json.loads(m.decode('utf-8'))
            )
            await consumer.start()
            LOG.info("✅ State Sync Worker connected to Kafka brokers at %s", KAFKA_BROKERS)
            break
        except Exception as e:
            LOG.warning("Kafka broker connection failed: %s. Retrying in 5 seconds...", e)
            if consumer:
                await consumer.stop()
            await asyncio.sleep(5)
            
    try:
        async for msg in consumer:
            try:
                event = msg.value
                topic = msg.topic
                
                payload = event.get("payload")
                if not payload:
                    LOG.warning("Event missing payload: %s", event)
                    continue
                    
                if topic == "Offers_Created":
                    await handle_offer_created(payload)
                elif topic == "Orders_Requested":
                    await handle_orders_requested(payload)
                elif topic == "Orders_Finalized":
                    await handle_orders_finalized(payload)
                elif topic == "Orders_Failed":
                    await handle_orders_failed(payload)
                else:
                    LOG.warning("Received event from unexpected topic %s", topic)
                    
            except json.JSONDecodeError as jde:
                LOG.error("Failed to parse JSON message: %s", jde)
            except Exception as e:
                LOG.error("Error processing message from topic %s: %s", msg.topic, e)
    except asyncio.CancelledError:
        LOG.info("State Sync Worker received cancellation signal")
    finally:
        LOG.info("Stopping State Sync Worker consumer...")
        if consumer:
            await consumer.stop()

async def handle_offer_created(payload: dict):
    order_id = payload.get("order_id")
    driver_id = payload.get("driver_id")
    attempt = payload.get("attempt_count", 1)
    if not order_id:
        LOG.warning("Offers_Created missing order_id: %s", payload)
        return
        
    status_state = {
        "status": "offering",
        "driver_id": driver_id,
        "attempt": attempt
    }
    await set_order_status(order_id, status_state)
    LOG.info("Updated order %s status to OFFERING (driver: %s, attempt: %d)", order_id, driver_id, attempt)

async def handle_orders_requested(payload: dict):
    order_id = payload.get("order_id")
    attempt = payload.get("attempt_count", 1)
    if not order_id:
        LOG.warning("Orders_Requested missing order_id: %s", payload)
        return
        
    status_state = {
        "status": "accumulating",
        "attempt": attempt
    }
    await set_order_status(order_id, status_state)
    LOG.info("Updated order %s status to ACCUMULATING (attempt: %d)", order_id, attempt)

async def handle_orders_finalized(payload: dict):
    order_id = payload.get("order_id")
    driver_id = payload.get("driver_id")
    attempt = payload.get("attempt_count", 1)
    if not order_id:
        LOG.warning("Orders_Finalized missing order_id: %s", payload)
        return
        
    status_state = {
        "status": "matched",
        "driver_id": driver_id,
        "attempt": attempt
    }
    await set_order_status(order_id, status_state)
    LOG.info("Updated order %s status to MATCHED (driver: %s, attempt: %d)", order_id, driver_id, attempt)

async def handle_orders_failed(payload: dict):
    order_id = payload.get("order_id")
    attempt = payload.get("attempt_count", 1)
    reason = payload.get("reason", "unknown")
    if not order_id:
        LOG.warning("Orders_Failed missing order_id: %s", payload)
        return
        
    status_state = {
        "status": "failed",
        "attempt": attempt,
        "reason": reason
    }
    await set_order_status(order_id, status_state)
    LOG.info("Updated order %s status to FAILED (attempt: %d, reason: %s)", order_id, attempt, reason)
