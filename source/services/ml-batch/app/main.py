import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.config import Config
from app.optimizer import MatchOptimizer
from app.consumer import KafkaBatchConsumer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ml-batch-service")

consumer_worker = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing ML Batch & Optimization Service...")
    try:
        optimizer = MatchOptimizer()
        global consumer_worker
        consumer_worker = KafkaBatchConsumer(optimizer)
        await consumer_worker.start()
        logger.info("ML Batch & Optimization Service successfully started!")
    except Exception as e:
        logger.exception("Failed to initialize or start background matching consumer: %s", e)
        raise e

    yield
    
    # Shutdown
    logger.info("Stopping ML Batch & Optimization Service background workers...")
    if consumer_worker:
        await consumer_worker.stop()
    logger.info("ML Batch & Optimization Service shutdown complete.")

app = FastAPI(
    title="ML Batch & Optimization Service",
    description="Consumes OrderRequested events, matches drivers using model scoring and Hungarian bipartite optimization.",
    lifespan=lifespan
)

@app.get("/health")
def health_check():
    return {"status": "healthy"}
