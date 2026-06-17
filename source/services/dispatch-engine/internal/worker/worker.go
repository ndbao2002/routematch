package worker

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/google/uuid"
	kafkaGo "github.com/segmentio/kafka-go"
	"routematch/dispatch-engine/internal/config"
	"routematch/dispatch-engine/internal/db"
	"routematch/dispatch-engine/internal/kafka"
	"routematch/dispatch-engine/internal/redis"
)

type OfferContext struct {
	OrderID       string
	DriverID      string
	AttemptCount  int
	ResponseChan  chan string // receives "accept" or "reject"
	CorrelationID string
}

type Worker struct {
	cfg         *config.Config
	redisClient *redis.RedisClient
	dbClient    *db.DbClient
	kafkaClient *kafka.KafkaClient

	activeOffers   map[string]*OfferContext
	activeOffersMu sync.RWMutex
	
	wg sync.WaitGroup
}

func NewWorker(cfg *config.Config, rdb *redis.RedisClient, db *db.DbClient, kf *kafka.KafkaClient) *Worker {
	return &Worker{
		cfg:          cfg,
		redisClient:  rdb,
		dbClient:     db,
		kafkaClient:  kf,
		activeOffers: make(map[string]*OfferContext),
	}
}

func (w *Worker) Start(ctx context.Context) {
	w.wg.Add(2)
	go w.consumeMatchedAssignments(ctx)
	go w.consumeDriverResponses(ctx)
}

func (w *Worker) Stop() {
	w.wg.Wait()
}

func (w *Worker) consumeMatchedAssignments(ctx context.Context) {
	defer w.wg.Done()
	log.Printf("Starting matched assignments consumer on topic: Matched_Assignments")
	reader := w.kafkaClient.NewReader("Matched_Assignments", "dispatch-engine-group")
	defer reader.Close()

	for {
		select {
		case <-ctx.Done():
			return
		default:
			msg, err := reader.ReadMessage(ctx)
			if err != nil {
				if ctx.Err() != nil {
					return
				}
				log.Printf("Error reading matched assignments message: %v", err)
				continue
			}

			var batch kafka.MatchedAssignmentsBatch
			if err := json.Unmarshal(msg.Value, &batch); err != nil {
				log.Printf("Error deserializing MatchedAssignmentsBatch: %v", err)
				continue
			}

			log.Printf("Received matching batch %s with %d assignments", batch.BatchID, len(batch.Assignments))
			
			// Process each assignment in the batch
			for _, asg := range batch.Assignments {
				go w.processAssignment(ctx, asg, msg.Headers)
			}
		}
	}
}

func (w *Worker) processAssignment(ctx context.Context, asg kafka.MatchedAssignment, headers []kafkaGo.Header) {
	// 1. Get correlation ID from headers or generate one
	correlationID := ""
	for _, h := range headers {
		if h.Key == "correlation_id" {
			correlationID = string(h.Value)
			break
		}
	}
	if correlationID == "" {
		correlationID = uuid.New().String()
	}

	log.Printf("[Order: %s] Processing assignment to driver %s (attempt: %d)", asg.OrderID, asg.DriverID, asg.AttemptCount)

	// 2. Attempt Redis Redlock on driver
	locked, err := w.redisClient.AcquireDriverLock(ctx, asg.DriverID, asg.OrderID)
	if err != nil {
		log.Printf("[Order: %s] Redis lock error for driver %s: %v", asg.OrderID, asg.DriverID, err)
		return
	}
	if !locked {
		log.Printf("[Order: %s] Driver %s is already locked. Skipping assignment.", asg.OrderID, asg.DriverID)
		return
	}

	// 3. Verify current order status in DB (must be accumulating)
	status, err := w.dbClient.GetCurrentOrderStatus(ctx, asg.OrderID)
	if err != nil {
		log.Printf("[Order: %s] DB error retrieving status: %v. Releasing lock.", asg.OrderID, err)
		w.redisClient.ReleaseDriverLock(ctx, asg.DriverID)
		return
	}

	if status != "accumulating" {
		log.Printf("[Order: %s] Order is in '%s' state (not 'accumulating'). Skipping and releasing lock.", asg.OrderID, status)
		w.redisClient.ReleaseDriverLock(ctx, asg.DriverID)
		return
	}

	// 4. Update order state in PostgreSQL: accumulating -> offering
	err = w.dbClient.UpdateOrderStatus(ctx, asg.OrderID, "offering")
	if err != nil {
		log.Printf("[Order: %s] DB error updating status: %v. Releasing lock.", asg.OrderID, err)
		w.redisClient.ReleaseDriverLock(ctx, asg.DriverID)
		return
	}

	// Write transition
	_ = w.dbClient.InsertStateTransition(ctx, asg.OrderID,
		sql.NullString{String: "accumulating", Valid: true},
		sql.NullString{String: "offering", Valid: true},
		sql.NullString{String: asg.DriverID, Valid: true},
		sql.NullInt32{Int32: int32(asg.AttemptCount), Valid: true},
		sql.NullString{String: "match_offered", Valid: true},
	)

	// Log interaction (is_accepted = 0 initially / pending)
	err = w.dbClient.InsertInteractionLog(ctx, asg.OrderID, asg.DriverID, asg.AttemptCount,
		asg.DriverLat, asg.DriverLon, asg.DriverDistance, asg.DriverFatigue, 0)
	if err != nil {
		log.Printf("[Order: %s] Warning - failed to write interaction log: %v", asg.OrderID, err)
	}

	// 5. Publish Offers_Created event to Kafka
	offerID := uuid.New().String()
	offerPayload := map[string]interface{}{
		"order_id":         asg.OrderID,
		"driver_id":        asg.DriverID,
		"offer_id":         offerID,
		"attempt_count":    asg.AttemptCount,
		"expires_at":       time.Now().UTC().Add(time.Duration(w.cfg.OfferTimeoutSeconds) * time.Second).Format(time.RFC3339),
		"lock_ttl_seconds": w.cfg.OfferTimeoutSeconds,
	}

	err = w.kafkaClient.PublishEvent(ctx, "Offers_Created", correlationID, offerPayload)
	if err != nil {
		log.Printf("[Order: %s] Error publishing Offers_Created event: %v. Releasing status.", asg.OrderID, err)
		w.dbClient.UpdateOrderStatus(ctx, asg.OrderID, "accumulating")
		w.redisClient.ReleaseDriverLock(ctx, asg.DriverID)
		return
	}

	log.Printf("[Order: %s] Offer %s created for driver %s (attempt %d). Starting tracking.", asg.OrderID, offerID, asg.DriverID, asg.AttemptCount)

	// 6. Spawn timeout state machine
	w.startOfferTracking(ctx, asg.OrderID, asg.DriverID, asg.AttemptCount, correlationID)
}

func (w *Worker) startOfferTracking(ctx context.Context, orderID, driverID string, attemptCount int, correlationID string) {
	ch := make(chan string, 1)

	offerCtx := &OfferContext{
		OrderID:       orderID,
		DriverID:      driverID,
		AttemptCount:  attemptCount,
		ResponseChan:  ch,
		CorrelationID: correlationID,
	}

	w.activeOffersMu.Lock()
	w.activeOffers[orderID] = offerCtx
	w.activeOffersMu.Unlock()

	go func() {
		defer func() {
			w.activeOffersMu.Lock()
			delete(w.activeOffers, orderID)
			w.activeOffersMu.Unlock()
		}()

		timeoutDuration := time.Duration(w.cfg.OfferTimeoutSeconds) * time.Second
		select {
		case resp := <-ch:
			if resp == "accept" {
				w.resolveAccept(ctx, offerCtx)
			} else {
				w.resolveReject(ctx, offerCtx, "driver_rejected")
			}
		case <-time.After(timeoutDuration):
			log.Printf("[Order: %s] Driver %s offer TIMEOUT after %s", orderID, driverID, timeoutDuration)
			w.resolveReject(ctx, offerCtx, "offer_timeout")
		case <-ctx.Done():
			// Server shut down, release locks safely
			w.redisClient.ReleaseDriverLock(context.Background(), driverID)
		}
	}()
}

func (w *Worker) consumeDriverResponses(ctx context.Context) {
	defer w.wg.Done()
	log.Printf("Starting driver responses consumer on topic: Driver_Responses")
	reader := w.kafkaClient.NewReader("Driver_Responses", "dispatch-engine-group")
	defer reader.Close()

	for {
		select {
		case <-ctx.Done():
			return
		default:
			msg, err := reader.ReadMessage(ctx)
			if err != nil {
				if ctx.Err() != nil {
					return
				}
				log.Printf("Error reading driver response: %v", err)
				continue
			}

			var env kafka.DriverResponseEvent
			if err := json.Unmarshal(msg.Value, &env); err != nil {
				log.Printf("Error deserializing DriverResponseEnvelope: %v", err)
				continue
			}

			// Support both wrapped and unwrapped event structures
			var orderID, driverID, response string
			if env.Payload.OrderID != "" {
				orderID = env.Payload.OrderID
				driverID = env.Payload.DriverID
				response = env.Payload.Response
			} else {
				// Try parsing raw payload at root
				var rawPayload struct {
					OrderID  string `json:"order_id"`
					DriverID string `json:"driver_id"`
					Response string `json:"response"`
				}
				if json.Unmarshal(msg.Value, &rawPayload) == nil {
					orderID = rawPayload.OrderID
					driverID = rawPayload.DriverID
					response = rawPayload.Response
				}
			}

			if orderID == "" || driverID == "" {
				log.Printf("Driver response missing order_id/driver_id: %s", string(msg.Value))
				continue
			}

			w.activeOffersMu.RLock()
			offerCtx, exists := w.activeOffers[orderID]
			w.activeOffersMu.RUnlock()

			if exists && offerCtx.DriverID == driverID {
				log.Printf("[Order: %s] Received driver response: %s from driver %s", orderID, response, driverID)
				select {
				case offerCtx.ResponseChan <- response:
				default:
				}
			} else {
				log.Printf("Received response for order %s and driver %s but no active offer context exists", orderID, driverID)
			}
		}
	}
}

func (w *Worker) resolveAccept(ctx context.Context, o *OfferContext) {
	log.Printf("[Order: %s] Resolving ACCEPT for driver %s", o.OrderID, o.DriverID)

	// 1. Update database status to matched
	err := w.dbClient.UpdateOrderStatus(ctx, o.OrderID, "matched")
	if err != nil {
		log.Printf("[Order: %s] DB error matching order: %v", o.OrderID, err)
		w.redisClient.ReleaseDriverLock(ctx, o.DriverID)
		return
	}

	// 2. Update interaction log
	_ = w.dbClient.UpdateInteractionLogStatus(ctx, o.OrderID, o.AttemptCount, 1)

	// 3. Insert transition record
	_ = w.dbClient.InsertStateTransition(ctx, o.OrderID,
		sql.NullString{String: "offering", Valid: true},
		sql.NullString{String: "matched", Valid: true},
		sql.NullString{String: o.DriverID, Valid: true},
		sql.NullInt32{Int32: int32(o.AttemptCount), Valid: true},
		sql.NullString{String: "offer_accepted", Valid: true},
	)

	// 4. Publish Orders_Finalized event to Kafka
	payload := map[string]interface{}{
		"order_id":      o.OrderID,
		"driver_id":     o.DriverID,
		"attempt_count": o.AttemptCount,
		"matched_at":    time.Now().UTC().Format(time.RFC3339),
		"final_status":  "matched",
	}

	err = w.kafkaClient.PublishEvent(ctx, "Orders_Finalized", o.CorrelationID, payload)
	if err != nil {
		log.Printf("[Order: %s] Error publishing Orders_Finalized event: %v", o.OrderID, err)
	}

	// 5. Release driver lock (since driver status will become ACTIVE in the DB profiles,
	// and they will no longer show up as candidates)
	_ = w.redisClient.ReleaseDriverLock(ctx, o.DriverID)
	log.Printf("[Order: %s] MATCH finalized with driver %s successfully", o.OrderID, o.DriverID)
}

func (w *Worker) resolveReject(ctx context.Context, o *OfferContext, reason string) {
	log.Printf("[Order: %s] Resolving REJECT (Reason: %s) for driver %s", o.OrderID, reason, o.DriverID)

	// 1. Release lock on driver
	_ = w.redisClient.ReleaseDriverLock(ctx, o.DriverID)

	// 2. Add driver to rejected set in Redis so they are skipped in future matching cycles for this order
	_ = w.redisClient.AddToRejectedDrivers(ctx, o.OrderID, o.DriverID)

	// 3. Update interaction log: is_accepted = 0 (or already 0)
	_ = w.dbClient.UpdateInteractionLogStatus(ctx, o.OrderID, o.AttemptCount, 0)

	nextAttempt := o.AttemptCount + 1
	if nextAttempt > w.cfg.MaxAttempts {
		// FAIL order
		log.Printf("[Order: %s] Max match attempts reached (%d). Failing order.", o.OrderID, w.cfg.MaxAttempts)

		err := w.dbClient.UpdateOrderStatus(ctx, o.OrderID, "failed")
		if err != nil {
			log.Printf("[Order: %s] DB error failing order: %v", o.OrderID, err)
			return
		}

		_ = w.dbClient.InsertStateTransition(ctx, o.OrderID,
			sql.NullString{String: "offering", Valid: true},
			sql.NullString{String: "failed", Valid: true},
			sql.NullString{String: o.DriverID, Valid: true},
			sql.NullInt32{Int32: int32(o.AttemptCount), Valid: true},
			sql.NullString{String: fmt.Sprintf("failed_after_%s", reason), Valid: true},
		)

		failedPayload := map[string]interface{}{
			"order_id":      o.OrderID,
			"reason":        "no_drivers_available",
			"attempt_count": o.AttemptCount,
			"failed_at":     time.Now().UTC().Format(time.RFC3339),
		}

		err = w.kafkaClient.PublishEvent(ctx, "Orders_Failed", o.CorrelationID, failedPayload)
		if err != nil {
			log.Printf("[Order: %s] Error publishing Orders_Failed event: %v", o.OrderID, err)
		}
	} else {
		// RETRY matching: transition back to accumulating and emit event to trigger batch matching
		log.Printf("[Order: %s] Retrying order. Routing back to Orders_Requested (Next Attempt: %d)", o.OrderID, nextAttempt)

		err := w.dbClient.UpdateOrderStatus(ctx, o.OrderID, "accumulating")
		if err != nil {
			log.Printf("[Order: %s] DB error setting order status to accumulating: %v", o.OrderID, err)
			return
		}

		_ = w.dbClient.InsertStateTransition(ctx, o.OrderID,
			sql.NullString{String: "offering", Valid: true},
			sql.NullString{String: "accumulating", Valid: true},
			sql.NullString{String: o.DriverID, Valid: true},
			sql.NullInt32{Int32: int32(o.AttemptCount), Valid: true},
			sql.NullString{String: fmt.Sprintf("retry_after_%s", reason), Valid: true},
		)

		// Get order details from DB to publish retry payload
		details, err := w.dbClient.GetOrderDetails(ctx, o.OrderID)
		if err != nil {
			log.Printf("[Order: %s] Error loading order details for retry payload: %v", o.OrderID, err)
			return
		}

		retryPayload := map[string]interface{}{
			"order_id":               details.OrderID,
			"user_id":                details.UserID,
			"pickup_lat":             details.PickupLat,
			"pickup_lon":             details.PickupLon,
			"dropoff_lat":            details.DropoffLat,
			"dropoff_lon":            details.DropoffLon,
			"shipping_fee":           details.ShippingFee,
			"cod_amount":             details.CodAmount,
			"distance_km":            details.DistanceKm,
			"requested_vehicle_type": details.RequestedVehicleType,
			"service_type":           details.ServiceType,
			"is_raining":             details.IsRaining,
			"hour_of_day":            details.HourOfDay,
			"attempt_count":          nextAttempt,
		}

		err = w.kafkaClient.PublishEvent(ctx, "Orders_Requested", o.CorrelationID, retryPayload)
		if err != nil {
			log.Printf("[Order: %s] Error publishing retry to Orders_Requested: %v", o.OrderID, err)
		}
	}
}
