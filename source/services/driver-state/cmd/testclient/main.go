package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"routematch/shared/libs/go/pb"

	"github.com/redis/go-redis/v9"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func main() {
	log.Println("Starting integration test client for Driver State service...")

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	// Hydrate Redis with driver profiles and states directly
	rdbClient := redis.NewClient(&redis.Options{
		Addr: "localhost:6379",
	})
	defer rdbClient.Close()

	log.Println("Flushing Redis database...")
	if err := rdbClient.FlushDB(ctx).Err(); err != nil {
		log.Fatalf("Failed to flush Redis: %v", err)
	}

	// We will register three test drivers:
	// - D_test_go_1: bike, IDLE, near Pickup A
	// - D_test_go_2: bike, IDLE, near Pickup B
	// - D_test_go_3: bike, IDLE, very far away
	drivers := []struct {
		id          string
		vehicleType string
		maxLoad     int
		status      string
		lat         float64
		lon         float64
	}{
		{"D_test_go_1", "bike", 30, "IDLE", 10.762622, 106.660172},
		{"D_test_go_2", "bike", 50, "IDLE", 10.768000, 106.670000},
		{"D_test_go_3", "bike", 60, "IDLE", 11.200000, 107.200000},
	}

	log.Println("Hydrating mock driver profiles and states in Redis...")
	for _, d := range drivers {
		// Profile Hash
		err := rdbClient.HSet(ctx, fmt.Sprintf("driver:%s:profile", d.id), map[string]interface{}{
			"vehicle_type": d.vehicleType,
			"max_load_kg":  d.maxLoad,
			"joined_date":  "2024-01-01",
		}).Err()
		if err != nil {
			log.Fatalf("Failed to write mock profile for %s to Redis: %v", d.id, err)
		}

		// State Hash
		err = rdbClient.HSet(ctx, fmt.Sprintf("driver:%s:state", d.id), map[string]interface{}{
			"status":                    d.status,
			"lat":                       d.lat,
			"lon":                       d.lon,
			"driver_fatigue_index":      0.1,
			"driver_global_accept_rate": 0.8,
			"cancel_rate":               0.05,
			"minutes_active":            120,
		}).Err()
		if err != nil {
			log.Fatalf("Failed to write mock state for %s to Redis: %v", d.id, err)
		}
	}
	log.Println("Redis hydration complete.")

	// Connect to gRPC server
	conn, err := grpc.Dial("localhost:50051", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("Did not connect: %v", err)
	}
	defer conn.Close()
	client := pb.NewDriverStateServiceClient(conn)

	// Update locations via gRPC so H3 indexing keys are created in Redis
	log.Println("Updating locations via gRPC to index driver positions...")
	for _, d := range drivers {
		upRes, err := client.UpdateLocation(ctx, &pb.UpdateLocationRequest{
			DriverId: d.id,
			Lat:      d.lat,
			Lon:      d.lon,
		})
		if err != nil {
			log.Fatalf("UpdateLocation failed for %s: %v", d.id, err)
		}
		if !upRes.Success {
			log.Fatalf("UpdateLocation response failed for %s: %s", d.id, upRes.Message)
		}
	}
	log.Println("Driver locations indexed successfully.")

	// Test 1: Single order candidate query (GetCandidates)
	log.Println("--- Running Test 1: GetCandidates (Single Order) ---")
	candRes, err := client.GetCandidates(ctx, &pb.GetCandidatesRequest{
		CorrelationId:        "test-correlation-single",
		OrderId:              "order-single",
		PickupLat:            10.762622,
		PickupLon:            106.660172,
		SearchRadiusMeters:   1000.0,
		RequestedVehicleType: "bike",
		Limit:                5,
	})
	if err != nil {
		log.Fatalf("GetCandidates failed: %v", err)
	}
	log.Printf("GetCandidates Response: ReturnedCount=%d", candRes.ReturnedCount)
	if candRes.ReturnedCount != 1 || candRes.Candidates[0].DriverId != "D_test_go_1" {
		log.Fatalf("Test 1 Verification failed! Expected exactly D_test_go_1, got count=%d and data=%+v", candRes.ReturnedCount, candRes.Candidates)
	}
	log.Println("Test 1 passed successfully!")

	// Test 2: Batch candidate query (BatchGetCandidates)
	log.Println("--- Running Test 2: BatchGetCandidates (Multiple Orders) ---")
	batchRes, err := client.BatchGetCandidates(ctx, &pb.BatchGetCandidatesRequest{
		CorrelationId:        "test-correlation-batch",
		RequestedVehicleType: "bike",
		MaxCandidates:        5,
		Orders: []*pb.OrderSearchQuery{
			{
				OrderId:            "order-A",
				PickupLat:          10.762622,
				PickupLon:          106.660172,
				SearchRadiusMeters: 1000.0,
			},
			{
				OrderId:            "order-B",
				PickupLat:          10.768000,
				PickupLon:          106.670000,
				SearchRadiusMeters: 1000.0,
			},
		},
	})
	if err != nil {
		log.Fatalf("BatchGetCandidates failed: %v", err)
	}

	log.Printf("BatchGetCandidates Response: Status=%s, Message=%s, ReturnedCount=%d", batchRes.Status, batchRes.Message, batchRes.ReturnedCount)
	
	// Map to easily verify the returned candidate IDs
	returnedDriversMap := make(map[string]bool)
	for _, c := range batchRes.Candidates {
		returnedDriversMap[c.DriverId] = true
		fmt.Printf("Candidate Driver ID: %s (Status: %s, Lat: %f, Lon: %f)\n", c.DriverId, c.Status, c.DriverLat, c.DriverLon)
	}

	// Assert that D_test_go_1 and D_test_go_2 are returned
	if !returnedDriversMap["D_test_go_1"] || !returnedDriversMap["D_test_go_2"] {
		log.Fatalf("Test 2 Verification failed! Expected D_test_go_1 and D_test_go_2 in the pool. Got map: %+v", returnedDriversMap)
	}

	// Assert that D_test_go_3 is NOT returned
	if returnedDriversMap["D_test_go_3"] {
		log.Fatalf("Test 2 Verification failed! D_test_go_3 is too far and should not be in the pool.")
	}

	// Assert that the returned count is exactly 2
	if batchRes.ReturnedCount != 2 {
		log.Fatalf("Test 2 Verification failed! Expected ReturnedCount to be 2, got %d", batchRes.ReturnedCount)
	}

	log.Println("Test 2 passed successfully!")
	log.Println("All integration test checks completed successfully!")
}
