package config

import (
	"os"
)

type Config struct {
	Port      string
	RedisHost string
	RedisPort string
}

func LoadConfig() *Config {
	port := os.Getenv("PORT")
	if port == "" {
		port = "50051"
	}
	redisHost := os.Getenv("REDIS_HOST")
	if redisHost == "" {
		redisHost = "localhost"
	}
	redisPort := os.Getenv("REDIS_PORT")
	if redisPort == "" {
		redisPort = "6379"
	}
	return &Config{
		Port:      port,
		RedisHost: redisHost,
		RedisPort: redisPort,
	}
}
