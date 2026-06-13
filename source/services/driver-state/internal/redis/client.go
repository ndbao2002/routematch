package redis

import (
	"context"
	"fmt"
	"math"
	"strconv"
	"time"

	"github.com/redis/go-redis/v9"
	"github.com/uber/h3-go/v3"
)

type RedisClient struct {
	rdb *redis.Client
}

func NewRedisClient(host, port string) *RedisClient {
	rdb := redis.NewClient(&redis.Options{
		Addr: fmt.Sprintf("%s:%s", host, port),
	})
	return &RedisClient{rdb: rdb}
}

func (c *RedisClient) Close() error {
	return c.rdb.Close()
}

type DriverCandidate struct {
	DriverID      string
	Lat           float64
	Lon           float64
	FatigueIndex  float64
	AcceptRate    float64
	CancelRate    float64
	MinutesActive int32
	Status        string
	VehicleType   string
	MaxLoadKG     int32
}

func (c *RedisClient) GetCandidateDrivers(ctx context.Context, lat, lon float64, vehicleType string) ([]DriverCandidate, error) {
	latRad := lat * math.Pi / 180.0
	lonRad := lon * math.Pi / 180.0
	coord := h3.GeoCoord{Latitude: latRad, Longitude: lonRad}
	centerCell := h3.FromGeo(coord, 8)

	var candidateIDs []string
	visitedCells := make(map[h3.H3Index]bool)

	// Expand rings from k=0 to k=5
	for k := 0; k <= 5; k++ {
		cells := h3.KRing(centerCell, k)
		
		var newCells []h3.H3Index
		for _, cell := range cells {
			if !visitedCells[cell] {
				visitedCells[cell] = true
				newCells = append(newCells, cell)
			}
		}

		if len(newCells) == 0 {
			continue
		}

		pipe := c.rdb.Pipeline()
		var cmds []*redis.StringSliceCmd
		for _, cell := range newCells {
			h3Str := fmt.Sprintf("%x", cell)
			key := fmt.Sprintf("drivers:h3:%s:%s", h3Str, vehicleType)
			cmds = append(cmds, pipe.ZRange(ctx, key, 0, 99))
		}

		_, err := pipe.Exec(ctx)
		if err != nil && err != redis.Nil {
			return nil, err
		}

		for _, cmd := range cmds {
			ids, _ := cmd.Result()
			candidateIDs = append(candidateIDs, ids...)
		}

		if len(candidateIDs) >= 25 {
			break
		}
	}

	if len(candidateIDs) == 0 {
		return nil, nil
	}

	pipe := c.rdb.Pipeline()
	var stateCmds []*redis.MapStringStringCmd
	var profileCmds []*redis.MapStringStringCmd

	for _, id := range candidateIDs {
		stateCmds = append(stateCmds, pipe.HGetAll(ctx, fmt.Sprintf("driver:%s:state", id)))
		profileCmds = append(profileCmds, pipe.HGetAll(ctx, fmt.Sprintf("driver:%s:profile", id)))
	}

	_, err := pipe.Exec(ctx)
	if err != nil && err != redis.Nil {
		return nil, err
	}

	var candidates []DriverCandidate
	for i, id := range candidateIDs {
		state, _ := stateCmds[i].Result()
		profile, _ := profileCmds[i].Result()

		if len(state) == 0 {
			continue
		}

		dLat, _ := strconv.ParseFloat(state["lat"], 64)
		dLon, _ := strconv.ParseFloat(state["lon"], 64)
		fatigue, _ := strconv.ParseFloat(state["driver_fatigue_index"], 64)
		acceptRate, _ := strconv.ParseFloat(state["driver_global_accept_rate"], 64)
		cancelRate, _ := strconv.ParseFloat(state["cancel_rate"], 64)
		minutesActive, _ := strconv.ParseInt(state["minutes_active"], 10, 32)
		maxLoad, _ := strconv.ParseInt(profile["max_load_kg"], 10, 32)

		candidates = append(candidates, DriverCandidate{
			DriverID:      id,
			Lat:           dLat,
			Lon:           dLon,
			FatigueIndex:  fatigue,
			AcceptRate:    acceptRate,
			CancelRate:    cancelRate,
			MinutesActive: int32(minutesActive),
			Status:        state["status"],
			VehicleType:   profile["vehicle_type"],
			MaxLoadKG:     int32(maxLoad),
		})
	}

	return candidates, nil
}

func (c *RedisClient) UpdateDriverLocation(ctx context.Context, driverID string, lat, lon float64) error {
	latRad := lat * math.Pi / 180.0
	lonRad := lon * math.Pi / 180.0
	coord := h3.GeoCoord{Latitude: latRad, Longitude: lonRad}
	cell := h3.FromGeo(coord, 8)
	h3Str := fmt.Sprintf("%x", cell)

	profileKey := fmt.Sprintf("driver:%s:profile", driverID)
	vehicleType, err := c.rdb.HGet(ctx, profileKey, "vehicle_type").Result()
	if err != nil {
		return err
	}

	stateKey := fmt.Sprintf("driver:%s:state", driverID)
	
	pipe := c.rdb.Pipeline()
	
	oldCell, _ := c.rdb.HGet(ctx, stateKey, "h3_cell").Result()
	if oldCell != "" && oldCell != h3Str {
		oldKey := fmt.Sprintf("drivers:h3:%s:%s", oldCell, vehicleType)
		pipe.ZRem(ctx, oldKey, driverID)
	}
	
	newKey := fmt.Sprintf("drivers:h3:%s:%s", h3Str, vehicleType)
	pipe.ZAdd(ctx, newKey, redis.Z{
		Score:  float64(time.Now().Unix()),
		Member: driverID,
	})

	pipe.HSet(ctx, stateKey, map[string]interface{}{
		"lat":     strconv.FormatFloat(lat, 'f', -1, 64),
		"lon":     strconv.FormatFloat(lon, 'f', -1, 64),
		"h3_cell": h3Str,
	})

	_, err = pipe.Exec(ctx)
	return err
}
