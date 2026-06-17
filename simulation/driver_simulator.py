import asyncio
import json
import logging
import os
import random
import uuid
from datetime import datetime
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("driver-simulator")

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
OFFERS_TOPIC = "Offers_Created"
RESPONSES_TOPIC = "Driver_Responses"

async def main():
    logger.info("Starting Driver Response Simulator...")
    logger.info("Connecting to Kafka at %s", KAFKA_BOOTSTRAP_SERVERS)
    
    consumer = AIOKafkaConsumer(
        OFFERS_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id="driver-simulator-group",
        auto_offset_reset="latest",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )
    
    producer = AIOKafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda m: json.dumps(m).encode("utf-8"),
    )
    
    await consumer.start()
    await producer.start()
    logger.info("✅ Connected to Kafka. Subscribed to %s. Ready to process offers.", OFFERS_TOPIC)
    
    try:
        async for msg in consumer:
            event = msg.value
            if not isinstance(event, dict):
                continue
                
            payload = event.get("payload")
            if not payload or not isinstance(payload, dict):
                logger.warning("Received event with missing/invalid payload: %s", event)
                continue
                
            order_id = payload.get("order_id")
            driver_id = payload.get("driver_id")
            offer_id = payload.get("offer_id")
            attempt = payload.get("attempt_count", 1)
            
            if not order_id or not driver_id:
                logger.warning("Payload missing order_id or driver_id: %s", payload)
                continue
                
            logger.info("📬 Received Offer %s for Order %s assigned to Driver %s (Attempt %d)", 
                        offer_id, order_id, driver_id, attempt)
            
            # Simulate decision logic:
            # 80% Accept
            # 15% Reject
            # 5% Timeout (ignore the offer, causing the 15-second tracking in dispatch-engine to expire)
            roll = random.random()
            if roll < 0.80:
                decision = "accept"
            elif roll < 0.95:
                decision = "reject"
            else:
                decision = "timeout"
                
            if decision == "timeout":
                logger.info("⏰ Driver %s chooses to TIMEOUT on Offer %s (doing nothing)", driver_id, offer_id)
                continue
                
            # Simulate driver reaction time
            reaction_time = random.uniform(1.0, 4.0)
            logger.info("🤔 Driver %s is thinking for %.2f seconds...", driver_id, reaction_time)
            await asyncio.sleep(reaction_time)
            
            # Construct response envelope
            resp_payload = {
                "order_id": order_id,
                "driver_id": driver_id,
                "response": decision
            }
            
            resp_envelope = {
                "event_id": str(uuid.uuid4()),
                "event_type": "Driver_Responses",
                "event_version": 1,
                "occurred_at": datetime.utcnow().isoformat() + "Z",
                "correlation_id": event.get("correlation_id", str(uuid.uuid4())),
                "payload": resp_payload
            }
            
            # Publish response
            await producer.send_and_wait(RESPONSES_TOPIC, resp_envelope)
            logger.info("📤 Published Driver Response: %s for Order %s from Driver %s", 
                        decision.upper(), order_id, driver_id)
            
    except asyncio.CancelledError:
        logger.info("Shutting down driver simulator...")
    finally:
        await consumer.stop()
        await producer.stop()
        logger.info("Driver simulator stopped.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
