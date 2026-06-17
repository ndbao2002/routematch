package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"

	"routematch/dispatch-engine/internal/config"
	"routematch/dispatch-engine/internal/db"
	"routematch/dispatch-engine/internal/kafka"
	"routematch/dispatch-engine/internal/redis"
	"routematch/dispatch-engine/internal/worker"
)

func main() {
	log.Printf("Starting Dispatch & State Machine Engine...")

	// Start Prometheus metrics server
	go func() {
		mux := http.NewServeMux()
		mux.Handle("/metrics", promhttp.Handler())
		log.Printf("Starting Prometheus metrics server on :9091")
		if err := http.ListenAndServe(":9091", mux); err != nil {
			log.Printf("Metrics server error: %v", err)
		}
	}()

	cfg := config.LoadConfig()

	// 1. Initialize Postgres Client with Retries
	var dbClient *db.DbClient
	var err error
	for i := 0; i < 10; i++ {
		dbClient, err = db.NewDbClient(cfg.DbConnStr)
		if err == nil {
			err = dbClient.Ping(context.Background())
		}
		if err == nil {
			log.Printf("✅ Connected to Postgres database")
			break
		}
		log.Printf("Postgres database connection failed: %v. Retrying in 3 seconds...", err)
		time.Sleep(3 * time.Second)
	}
	if err != nil {
		log.Fatalf("❌ Failed to connect to Postgres: %v", err)
	}
	defer dbClient.Close()

	// 2. Initialize Redis Client with Retries
	var redisClient *redis.RedisClient
	for i := 0; i < 10; i++ {
		redisClient = redis.NewRedisClient(cfg.RedisAddr)
		err = redisClient.Ping(context.Background())
		if err == nil {
			log.Printf("✅ Connected to Redis at %s", cfg.RedisAddr)
			break
		}
		log.Printf("Redis connection failed: %v. Retrying in 3 seconds...", err)
		time.Sleep(3 * time.Second)
	}
	if err != nil {
		log.Fatalf("❌ Failed to connect to Redis: %v", err)
	}

	// 3. Initialize Kafka Client
	brokers := strings.Split(cfg.KafkaBrokers, ",")
	kafkaClient := kafka.NewKafkaClient(brokers)
	defer kafkaClient.Close()
	log.Printf("✅ Initialized Kafka producer/consumer for brokers: %s", cfg.KafkaBrokers)

	// 4. Initialize State Machine Worker
	w := worker.NewWorker(cfg, redisClient, dbClient, kafkaClient)

	// Context for graceful shutdown
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	w.Start(ctx)

	log.Printf("🚀 Dispatch & State Machine Engine is running. Press CTRL+C to stop.")

	// Wait for terminate signal
	<-ctx.Done()
	log.Printf("Shutdown signal received. Stopping background workers...")

	// Graceful shutdown
	w.Stop()
	log.Printf("👋 Dispatch & State Machine Engine shutdown complete.")
}
