package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"routematch/shared/libs/go/pb"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func main() {
	log.Println("Starting integration test client for Driver State service...")

	// Connect to gRPC server
	conn, err := grpc.Dial("localhost:50051", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("Did not connect: %v", err)
	}
	defer conn.Close()
	client := pb.NewDriverStateServiceClient(conn)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// 1. Update location for test driver D_test_go_123
	driverID := "D_test_go_123"
	lat := 10.762622
	lon := 106.660172

	log.Printf("Updating location for driver %s to (lat: %f, lon: %f)...", driverID, lat, lon)
	upRes, err := client.UpdateLocation(ctx, &pb.UpdateLocationRequest{
		DriverId: driverID,
		Lat:      lat,
		Lon:      lon,
	})
	if err != nil {
		log.Fatalf("UpdateLocation failed: %v", err)
	}
	log.Printf("UpdateLocation Response: Success=%t, Message=%s", upRes.Success, upRes.Message)

	// 2. Query candidates around that location
	// Note: since center is at 10.762622, 106.660172, we search at the exact same location
	log.Printf("Retrieving candidates for pickup at (lat: %f, lon: %f) with vehicle type 'bike'...", lat, lon)
	candRes, err := client.GetCandidates(ctx, &pb.GetCandidatesRequest{
		CorrelationId:        "test-correlation-id",
		OrderId:              "test-order-id",
		PickupLat:            lat,
		PickupLon:            lon,
		SearchRadiusMeters:   1000.0,
		RequestedVehicleType: "bike",
		Limit:                5,
	})
	if err != nil {
		log.Fatalf("GetCandidates failed: %v", err)
	}

	log.Printf("GetCandidates Response: Status=%s, Message=%s, ReturnedCount=%d", candRes.Status, candRes.Message, candRes.ReturnedCount)
	for i, cand := range candRes.Candidates {
		fmt.Printf("Candidate %d:\n", i+1)
		fmt.Printf("  Driver ID: %s\n", cand.DriverId)
		fmt.Printf("  Vehicle Type: %s\n", cand.VehicleType)
		fmt.Printf("  Max Load KG: %d\n", cand.MaxLoadKg)
		fmt.Printf("  Distance to Pickup: %.2f meters\n", cand.DistanceToPickupMeters)
		fmt.Printf("  Status: %s\n", cand.Status)
		fmt.Printf("  Fatigue Index: %.2f\n", cand.FatigueIndex)
		fmt.Printf("  Accept Rate: %.2f\n", cand.AcceptRate)
		fmt.Printf("  Cancel Rate: %.2f\n", cand.CancelRate)
		fmt.Printf("  Minutes Active: %d\n", cand.MinutesActive)
	}

	if candRes.ReturnedCount != 1 || candRes.Candidates[0].DriverId != driverID {
		log.Fatalf("Verification failed! Expected candidate %s to be returned, got %+v", driverID, candRes.Candidates)
	}

	log.Println("Verification check passed successfully!")
}
