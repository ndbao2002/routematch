import os

class Config:
    KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
    KAFKA_CONSUMER_GROUP = os.getenv("KAFKA_CONSUMER_GROUP", "ml-batch-group")
    KAFKA_INPUT_TOPIC = os.getenv("KAFKA_INPUT_TOPIC", "Orders_Requested")
    KAFKA_OUTPUT_TOPIC = os.getenv("KAFKA_OUTPUT_TOPIC", "Matched_Assignments")
    
    REDIS_HOST = os.getenv("REDIS_HOST", "redis")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    
    DRIVER_STATE_GRPC_ADDR = os.getenv("DRIVER_STATE_GRPC_ADDR", "driver-state:50051")
    
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    MODEL_NAME = os.getenv("MODEL_NAME", "RouteMatchScoring")
    MODEL_STAGE = os.getenv("MODEL_STAGE", "None")  # Can be None, Staging, Production
    
    WEIGHT_PROB = float(os.getenv("WEIGHT_PROB", "1.0"))
    WEIGHT_DISTANCE = float(os.getenv("WEIGHT_DISTANCE", "0.1"))
    
    PORT = int(os.getenv("PORT", "8080"))
    HOST = os.getenv("HOST", "0.0.0.0")
    
    BATCH_TIMEOUT_SECONDS = float(os.getenv("BATCH_TIMEOUT_SECONDS", "5.0"))
    BATCH_MAX_SIZE = int(os.getenv("BATCH_MAX_SIZE", "50"))
