package config

import (
	"os"
	"strconv"
)

type Config struct {
	KafkaBrokers         string
	RedisAddr            string
	DbConnStr            string
	OfferTimeoutSeconds  int
	MaxAttempts          int
}

func LoadConfig() *Config {
	offerTimeout := 15
	if val := os.Getenv("OFFER_TIMEOUT_SECONDS"); val != "" {
		if i, err := strconv.Atoi(val); err == nil {
			offerTimeout = i
		}
	}

	maxAttempts := 3
	if val := os.Getenv("MAX_ATTEMPTS"); val != "" {
		if i, err := strconv.Atoi(val); err == nil {
			maxAttempts = i
		}
	}

	return &Config{
		KafkaBrokers:        getEnv("KAFKA_BROKERS", "localhost:19092"),
		RedisAddr:           getEnv("REDIS_ADDR", "localhost:6379"),
		DbConnStr:           getEnv("DB_CONN_STR", "postgres://postgres:postgres@localhost:5433/routematch?sslmode=disable"),
		OfferTimeoutSeconds: offerTimeout,
		MaxAttempts:         maxAttempts,
	}
}

func getEnv(key, fallback string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return fallback
}
