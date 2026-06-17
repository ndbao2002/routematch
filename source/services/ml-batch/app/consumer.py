import asyncio
import json
import logging
from typing import List, Dict, Any
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from app.config import Config
from app.optimizer import MatchOptimizer

logger = logging.getLogger("ml-batch-consumer")

class KafkaBatchConsumer:
    def __init__(self, optimizer: MatchOptimizer):
        self.optimizer = optimizer
        self.consumer = None
        self.producer = None
        self.running = False
        self.buffer: List[Dict[str, Any]] = []
        self.buffer_lock = asyncio.Lock()
        self.loop_task = None
        self.timeout_task = None

    async def start(self):
        self.running = True
        logger.info("Initializing Kafka Consumer on topic: %s", Config.KAFKA_INPUT_TOPIC)
        self.consumer = AIOKafkaConsumer(
            Config.KAFKA_INPUT_TOPIC,
            bootstrap_servers=Config.KAFKA_BOOTSTRAP_SERVERS,
            group_id=Config.KAFKA_CONSUMER_GROUP,
            auto_offset_reset="latest",
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            enable_auto_commit=True
        )
        
        logger.info("Initializing Kafka Producer on topic: %s", Config.KAFKA_OUTPUT_TOPIC)
        self.producer = AIOKafkaProducer(
            bootstrap_servers=Config.KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode("utf-8")
        )
        
        await self.consumer.start()
        await self.producer.start()
        logger.info("Kafka consumer and producer started successfully.")
        
        self.loop_task = asyncio.create_task(self.consume_loop())

    async def stop(self):
        self.running = False
        if self.loop_task:
            self.loop_task.cancel()
            try:
                await self.loop_task
            except asyncio.CancelledError:
                pass
        if self.timeout_task:
            self.timeout_task.cancel()
            try:
                await self.timeout_task
            except asyncio.CancelledError:
                pass
        
        # Flush remaining buffer on shutdown
        async with self.buffer_lock:
            if self.buffer:
                logger.info("Shutting down - flushing remaining %d orders in buffer", len(self.buffer))
                await self.flush_buffer()

        if self.consumer:
            await self.consumer.stop()
        if self.producer:
            await self.producer.stop()
        logger.info("Kafka consumer and producer stopped.")

    async def consume_loop(self):
        try:
            while self.running:
                msg_set = await self.consumer.getmany(timeout_ms=1000)
                if not msg_set:
                    continue
                
                async with self.buffer_lock:
                    for tp, messages in msg_set.items():
                        for msg in messages:
                            val = msg.value
                            if not val:
                                continue
                            payload = val.get("payload") if isinstance(val, dict) and "payload" in val else val
                            if not isinstance(payload, dict) or "order_id" not in payload:
                                logger.warning("Received invalid order requested message: %s", val)
                                continue
                            self.buffer.append(payload)
                    
                    if len(self.buffer) > 0:
                        logger.info("Buffer size: %d/%d", len(self.buffer), Config.BATCH_MAX_SIZE)
                    
                    # If buffer is not empty and no timer is running, start one
                    if len(self.buffer) > 0 and not self.timeout_task:
                        self.timeout_task = asyncio.create_task(self.schedule_timeout())
                    
                    # If max size reached, trigger immediately
                    if len(self.buffer) >= Config.BATCH_MAX_SIZE:
                        logger.info("Max batch size reached (%d). Flushing immediately.", Config.BATCH_MAX_SIZE)
                        if self.timeout_task:
                            self.timeout_task.cancel()
                            self.timeout_task = None
                        await self.flush_buffer()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception("Unexpected error in Kafka consume loop: %s", e)

    async def schedule_timeout(self):
        try:
            await asyncio.sleep(Config.BATCH_TIMEOUT_SECONDS)
            async with self.buffer_lock:
                if len(self.buffer) > 0:
                    logger.info("Window timeout (%ds) reached. Flushing batch.", Config.BATCH_TIMEOUT_SECONDS)
                    await self.flush_buffer()
        except asyncio.CancelledError:
            pass
        finally:
            self.timeout_task = None

    async def flush_buffer(self):
        if not self.buffer:
            return
        
        batch = list(self.buffer)
        self.buffer.clear()
        
        # Run CPU-heavy model inference, gRPC, and Hungarian solver in an executor thread
        loop = asyncio.get_running_loop()
        try:
            matched_assignments = await loop.run_in_executor(None, self.optimizer.optimize_batch, batch)
            if matched_assignments and matched_assignments.get("assignments"):
                logger.info("Publishing matching assignments to %s", Config.KAFKA_OUTPUT_TOPIC)
                await self.producer.send_and_wait(Config.KAFKA_OUTPUT_TOPIC, matched_assignments)
                logger.info("Successfully published %d matches.", len(matched_assignments["assignments"]))
            else:
                logger.info("No matches generated for this batch.")
        except Exception as e:
            logger.exception("Error executing batch optimization: %s", e)
