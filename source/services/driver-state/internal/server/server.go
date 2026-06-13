package server

import (
	"context"
	"math"

	rdb "routematch/driver-state/internal/redis"
	"routematch/shared/libs/go/pb"
)

type DriverStateServer struct {
	pb.UnimplementedDriverStateServiceServer
	redisClient *rdb.RedisClient
}

func NewDriverStateServer(redisClient *rdb.RedisClient) *DriverStateServer {
	return &DriverStateServer{
		redisClient: redisClient,
	}
}

func (s *DriverStateServer) GetCandidates(ctx context.Context, req *pb.GetCandidatesRequest) (*pb.GetCandidatesResponse, error) {
	candidates, err := s.redisClient.GetCandidateDrivers(ctx, req.PickupLat, req.PickupLon, req.RequestedVehicleType)
	if err != nil {
		return nil, err
	}

	rejectedMap := make(map[string]bool)
	for _, id := range req.RejectedDriverIds {
		rejectedMap[id] = true
	}

	var eligibleCandidates []*pb.DriverCandidate
	for _, cand := range candidates {
		if cand.Status != "IDLE" {
			continue
		}
		if rejectedMap[cand.DriverID] {
			continue
		}

		distMeters := calculateDistanceMeters(req.PickupLat, req.PickupLon, cand.Lat, cand.Lon)

		if req.SearchRadiusMeters > 0 && distMeters > req.SearchRadiusMeters {
			continue
		}

		eligibleCandidates = append(eligibleCandidates, &pb.DriverCandidate{
			DriverId:               cand.DriverID,
			VehicleType:            cand.VehicleType,
			MaxLoadKg:              cand.MaxLoadKG,
			DriverLat:              cand.Lat,
			DriverLon:              cand.Lon,
			DistanceToPickupMeters: distMeters,
			FatigueIndex:           cand.FatigueIndex,
			AcceptRate:             cand.AcceptRate,
			CancelRate:             cand.CancelRate,
			MinutesActive:          cand.MinutesActive,
			Status:                 cand.Status,
		})
	}

	if req.Limit > 0 && int32(len(eligibleCandidates)) > req.Limit {
		eligibleCandidates = eligibleCandidates[:req.Limit]
	}

	return &pb.GetCandidatesResponse{
		CorrelationId:      req.CorrelationId,
		OrderId:            req.OrderId,
		Candidates:         eligibleCandidates,
		ReturnedCount:      int32(len(eligibleCandidates)),
		RequestedLimit:     req.Limit,
		SearchRadiusMeters: int32(req.SearchRadiusMeters),
		Status:             "success",
		Message:            "Successfully retrieved candidates",
	}, nil
}

func (s *DriverStateServer) UpdateLocation(ctx context.Context, req *pb.UpdateLocationRequest) (*pb.UpdateLocationResponse, error) {
	err := s.redisClient.UpdateDriverLocation(ctx, req.DriverId, req.Lat, req.Lon)
	if err != nil {
		return &pb.UpdateLocationResponse{
			Success: false,
			Message: err.Error(),
		}, nil
	}

	return &pb.UpdateLocationResponse{
		Success: true,
		Message: "Successfully updated driver location",
	}, nil
}

func calculateDistanceMeters(lat1, lon1, lat2, lon2 float64) float64 {
	const R = 6371000.0
	radLat1 := lat1 * math.Pi / 180.0
	radLat2 := lat2 * math.Pi / 180.0
	diffLat := (lat2 - lat1) * math.Pi / 180.0
	diffLon := (lon2 - lon1) * math.Pi / 180.0

	a := math.Sin(diffLat/2)*math.Sin(diffLat/2) +
		math.Cos(radLat1)*math.Cos(radLat2)*
			math.Sin(diffLon/2)*math.Sin(diffLon/2)
	c := 2 * math.Asin(math.Sqrt(a))
	return R * c
}
