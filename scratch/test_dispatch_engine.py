import asyncio
import json
import uuid
import sys
import os
import time
import psycopg2
import redis
from datetime import datetime, timezone
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer

KAFKA_BROKERS = os.getenv("KAFKA_BROKERS", "localhost:19092")
DB_CONN = os.getenv("DB_CONN", "dbname=routematch user=postgres password=postgres host=localhost port=5433")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

ORDER_ID = "test_disp_order_999"
DRIVER_ID = "D_disp_drv_999"

async def publish_event(topic: str, payload: dict, correlation_id: str):
    producer = AIOKafkaProducer(bootstrap_servers=KAFKA_BROKERS)
    await producer.start()
    try:
        envelope = {
            "event_id": str(uuid.uuid4()),
            "event_type": topic,
            "event_version": 1,
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "correlation_id": correlation_id,
            "payload": payload
        }
        value_bytes = json.dumps(envelope).encode('utf-8')
        await producer.send_and_wait(topic, value_bytes)
        print(f"Published envelope to topic {topic}")
    finally:
        await producer.stop()

async def assert_event_emitted(consumer, expected_topic, order_id, timeout=8):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            msg = await asyncio.wait_for(consumer.getone(), timeout=1.0)
            if msg.topic == expected_topic:
                envelope = json.loads(msg.value.decode('utf-8'))
                # Handle both wrapped and unwrapped event values
                payload = envelope.get("payload") if isinstance(envelope, dict) and "payload" in envelope else envelope
                if isinstance(payload, dict) and payload.get("order_id") == order_id:
                    print(f"🎯 Consumed expected event on topic '{expected_topic}': {envelope}")
                    return envelope
        except asyncio.TimeoutError:
            continue
    raise AssertionError(f"❌ Timeout exceeded waiting for event on topic '{expected_topic}' for order '{order_id}'")

async def main():
    # 1. Clean up existing records in Postgres and Redis
    print(f"Connecting to database with: {DB_CONN}")
    conn = psycopg2.connect(DB_CONN)
    cur = conn.cursor()
    cur.execute("DELETE FROM interaction_logs WHERE order_id = %s;", (ORDER_ID,))
    cur.execute("DELETE FROM order_state_transitions WHERE order_id = %s;", (ORDER_ID,))
    cur.execute("DELETE FROM orders WHERE order_id = %s;", (ORDER_ID,))
    
    # Insert driver if not exists
    cur.execute("INSERT INTO drivers (driver_id, vehicle_type, max_load_kg, joined_date) VALUES (%s, 'bike', 30, NOW()) ON CONFLICT (driver_id) DO NOTHING;", (DRIVER_ID,))
    
    # Insert order
    cur.execute("""
        INSERT INTO orders (
            order_id, user_id, created_at, status, status_updated_at,
            pickup_lat, pickup_lon, dropoff_lat, dropoff_lon,
            shipping_fee, cod_amount, distance_km, requested_vehicle_type,
            service_type, is_raining, hour_of_day
        ) VALUES (
            %s, 'U_test_user', NOW(), 'accumulating', NOW(),
            10.762622, 106.660172, 10.772622, 106.670172,
            15000.0, 100000.0, 1.5, 'bike',
            'standard', FALSE, 14
        );
    """, (ORDER_ID,))
    conn.commit()
    
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
    r.delete(f"lock:driver:{DRIVER_ID}")
    r.delete(f"order:rejected:{ORDER_ID}")
    
    print("🧹 Database and Redis cleaned and initialized.")

    # Start Kafka Producer and Consumer to monitor events
    producer = AIOKafkaProducer(bootstrap_servers=KAFKA_BROKERS)
    await producer.start()
    
    consumer = AIOKafkaConsumer(
        "Offers_Created", "Orders_Finalized", "Orders_Failed", "Orders_Requested",
        bootstrap_servers=KAFKA_BROKERS,
        auto_offset_reset="latest"
    )
    await consumer.start()
    
    # Short sleep to allow consumers to fetch partitions
    await asyncio.sleep(1)

    print("\n--- Test Case 1: Driver Accepts Match ---")
    
    # 2. Publish match assignment to Matched_Assignments
    asg_event = {
        "batch_id": "test-batch-uuid",
        "assignments": [
            {
                "order_id": ORDER_ID,
                "driver_id": DRIVER_ID,
                "score": 0.85,
                "attempt_count": 1,
                "rank": 1,
                "driver_lat": 10.7630,
                "driver_lon": 106.6605,
                "driver_distance_to_pickup": 0.5,
                "driver_fatigue_index": 0.1
            }
        ]
    }
    
    print("Publishing matched assignment...")
    await producer.send_and_wait("Matched_Assignments", json.dumps(asg_event).encode('utf-8'))
    
    # Wait for processing
    await asyncio.sleep(2)
    
    # Check DB status
    cur.execute("SELECT status FROM orders WHERE order_id = %s;", (ORDER_ID,))
    status = cur.fetchone()[0]
    print(f"Postgres Order Status: {status} (Expected: offering)")
    assert status == "offering"
    
    # Check Redis lock
    lock_val = r.get(f"lock:driver:{DRIVER_ID}")
    print(f"Redis Driver Lock Value: {lock_val} (Expected: b'order:test_disp_order_999')")
    assert lock_val == f"order:{ORDER_ID}".encode('utf-8')
    
    # Consume Offers_Created to assert it was emitted
    await assert_event_emitted(consumer, "Offers_Created", ORDER_ID)
    
    # 3. Publish Driver Response: ACCEPT
    resp_event = {
        "event_id": str(uuid.uuid4()),
        "event_type": "Driver_Responses",
        "payload": {
            "order_id": ORDER_ID,
            "driver_id": DRIVER_ID,
            "response": "accept"
        }
    }
    print("Publishing driver response (accept)...")
    await producer.send_and_wait("Driver_Responses", json.dumps(resp_event).encode('utf-8'))
    
    # Wait for processing
    await asyncio.sleep(2)
    
    # Check DB status
    cur.execute("SELECT status FROM orders WHERE order_id = %s;", (ORDER_ID,))
    status = cur.fetchone()[0]
    print(f"Postgres Order Status: {status} (Expected: matched)")
    assert status == "matched"
    
    # Check Redis lock is released
    lock_val = r.get(f"lock:driver:{DRIVER_ID}")
    print(f"Redis Driver Lock Value: {lock_val} (Expected: None)")
    assert lock_val is None
    
    # Consume Orders_Finalized
    await assert_event_emitted(consumer, "Orders_Finalized", ORDER_ID)
    
    print("✅ Test Case 1 Passed!")

    print("\n--- Test Case 2: Driver Rejects, Escalates/Retries ---")
    # Reset status in DB to accumulating for Case 2
    cur.execute("UPDATE orders SET status = 'accumulating' WHERE order_id = %s;", (ORDER_ID,))
    conn.commit()
    r.delete(f"lock:driver:{DRIVER_ID}")
    
    # Publish match again (attempt_count = 1)
    asg_event["assignments"][0]["attempt_count"] = 1
    print("Publishing matched assignment for attempt 1...")
    await producer.send_and_wait("Matched_Assignments", json.dumps(asg_event).encode('utf-8'))
    await asyncio.sleep(2)
    
    # Publish reject response
    resp_event = {
        "event_id": str(uuid.uuid4()),
        "event_type": "Driver_Responses",
        "payload": {
            "order_id": ORDER_ID,
            "driver_id": DRIVER_ID,
            "response": "reject"
        }
    }
    print("Publishing driver response (reject)...")
    await producer.send_and_wait("Driver_Responses", json.dumps(resp_event).encode('utf-8'))
    await asyncio.sleep(2)
    
    # Confirm retry request was emitted with attempt_count = 2
    cur.execute("SELECT status FROM orders WHERE order_id = %s;", (ORDER_ID,))
    status = cur.fetchone()[0]
    print(f"Postgres Order Status after reject: {status} (Expected: accumulating)")
    assert status == "accumulating"
    
    # Check that driver is in the rejected list
    is_rejected = r.sismember(f"order:rejected:{ORDER_ID}", DRIVER_ID)
    print(f"Driver in Redis rejected list: {is_rejected} (Expected: True)")
    assert is_rejected
    
    # Check retry Orders_Requested event
    await assert_event_emitted(consumer, "Orders_Requested", ORDER_ID)
    
    print("✅ Test Case 2 Passed!")

    print("\n--- Test Case 3: Timeout Exceeded, Escalates ---")
    # Reset status in DB to accumulating for Case 3
    cur.execute("UPDATE orders SET status = 'accumulating' WHERE order_id = %s;", (ORDER_ID,))
    conn.commit()
    r.delete(f"order:rejected:{ORDER_ID}")
    r.delete(f"lock:driver:{DRIVER_ID}")
    
    # Publish match again (attempt_count = 2)
    asg_event["assignments"][0]["attempt_count"] = 2
    print("Publishing matched assignment for attempt 2...")
    await producer.send_and_wait("Matched_Assignments", json.dumps(asg_event).encode('utf-8'))
    await asyncio.sleep(2)
    
    print("Waiting 16 seconds for timeout to occur...")
    await asyncio.sleep(17)
    
    # Confirm retry request was emitted with attempt_count = 3
    cur.execute("SELECT status FROM orders WHERE order_id = %s;", (ORDER_ID,))
    status = cur.fetchone()[0]
    print(f"Postgres Order Status after timeout: {status} (Expected: accumulating)")
    assert status == "accumulating"
    
    # Check retry Orders_Requested event (attempt 3)
    await assert_event_emitted(consumer, "Orders_Requested", ORDER_ID)
    
    print("✅ Test Case 3 Passed!")

    print("\n--- Test Case 4: Max Attempts Exceeded, Order Fails ---")
    # Reset status in DB to accumulating
    cur.execute("UPDATE orders SET status = 'accumulating' WHERE order_id = %s;", (ORDER_ID,))
    conn.commit()
    r.delete(f"order:rejected:{ORDER_ID}")
    r.delete(f"lock:driver:{DRIVER_ID}")
    
    # Publish match again (attempt_count = 3, which is MAX_ATTEMPTS)
    asg_event["assignments"][0]["attempt_count"] = 3
    print("Publishing matched assignment for attempt 3...")
    await producer.send_and_wait("Matched_Assignments", json.dumps(asg_event).encode('utf-8'))
    await asyncio.sleep(2)
    
    # Publish reject response
    resp_event["payload"]["response"] = "reject"
    print("Publishing driver response (reject) for attempt 3...")
    await producer.send_and_wait("Driver_Responses", json.dumps(resp_event).encode('utf-8'))
    await asyncio.sleep(2)
    
    # Confirm order fails
    cur.execute("SELECT status FROM orders WHERE order_id = %s;", (ORDER_ID,))
    status = cur.fetchone()[0]
    print(f"Postgres Order Status: {status} (Expected: failed)")
    assert status == "failed"
    
    # Consume Orders_Failed
    await assert_event_emitted(consumer, "Orders_Failed", ORDER_ID)
    
    print("✅ Test Case 4 Passed!")
    
    # Cleanup
    await producer.stop()
    await consumer.stop()
    cur.close()
    conn.close()
    print("\n🎉 All Dispatch Engine Integration Tests Passed Successfully!")

if __name__ == "__main__":
    asyncio.run(main())
