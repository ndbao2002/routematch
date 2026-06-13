package main

import (
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"syscall"

	"routematch/driver-state/internal/config"
	rdb "routematch/driver-state/internal/redis"
	"routematch/driver-state/internal/server"
	"routematch/shared/libs/go/pb"

	"google.golang.org/grpc"
)

func main() {
	log.Println("Initializing Driver State & Location Service...")
	cfg := config.LoadConfig()

	redisClient := rdb.NewRedisClient(cfg.RedisHost, cfg.RedisPort)
	defer func() {
		if err := redisClient.Close(); err != nil {
			log.Printf("Error closing Redis client: %v", err)
		}
	}()
	log.Printf("Connected to Redis at %s:%s", cfg.RedisHost, cfg.RedisPort)

	lis, err := net.Listen("tcp", fmt.Sprintf(":%s", cfg.Port))
	if err != nil {
		log.Fatalf("Failed to listen on port %s: %v", cfg.Port, err)
	}
	log.Printf("gRPC server listening on port %s", cfg.Port)

	grpcServer := grpc.NewServer()
	driverServer := server.NewDriverStateServer(redisClient)
	pb.RegisterDriverStateServiceServer(grpcServer, driverServer)

	go func() {
		sigChan := make(chan os.Signal, 1)
		signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
		<-sigChan
		log.Println("Shutting down gRPC server...")
		grpcServer.GracefulStop()
	}()

	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("Failed to serve gRPC: %v", err)
	}
}
