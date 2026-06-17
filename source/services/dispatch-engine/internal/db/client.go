package db

import (
	"context"
	"database/sql"
	"time"

	_ "github.com/lib/pq"
)

type DbClient struct {
	db *sql.DB
}

type OrderDetails struct {
	OrderID              string
	UserID               string
	PickupLat            float64
	PickupLon            float64
	DropoffLat           float64
	DropoffLon           float64
	ShippingFee          float64
	CodAmount            float64
	DistanceKm           float64
	RequestedVehicleType string
	ServiceType          string
	IsRaining            bool
	HourOfDay            int
}

func NewDbClient(connStr string) (*DbClient, error) {
	db, err := sql.Open("postgres", connStr)
	if err != nil {
		return nil, err
	}
	db.SetMaxOpenConns(10)
	db.SetMaxIdleConns(5)
	db.SetConnMaxLifetime(5 * time.Minute)
	return &DbClient{db: db}, nil
}

func (c *DbClient) Ping(ctx context.Context) error {
	return c.db.PingContext(ctx)
}

func (c *DbClient) Close() error {
	return c.db.Close()
}

func (c *DbClient) GetCurrentOrderStatus(ctx context.Context, orderID string) (string, error) {
	var status string
	err := c.db.QueryRowContext(ctx, "SELECT status FROM orders WHERE order_id = $1", orderID).Scan(&status)
	return status, err
}

func (c *DbClient) GetOrderDetails(ctx context.Context, orderID string) (*OrderDetails, error) {
	query := `
		SELECT 
			order_id, user_id, pickup_lat, pickup_lon, dropoff_lat, dropoff_lon,
			shipping_fee, cod_amount, COALESCE(distance_km, 0), requested_vehicle_type,
			service_type, is_raining, hour_of_day
		FROM orders 
		WHERE order_id = $1
	`
	var o OrderDetails
	err := c.db.QueryRowContext(ctx, query, orderID).Scan(
		&o.OrderID, &o.UserID, &o.PickupLat, &o.PickupLon, &o.DropoffLat, &o.DropoffLon,
		&o.ShippingFee, &o.CodAmount, &o.DistanceKm, &o.RequestedVehicleType,
		&o.ServiceType, &o.IsRaining, &o.HourOfDay,
	)
	if err != nil {
		return nil, err
	}
	return &o, nil
}

func (c *DbClient) UpdateOrderStatus(ctx context.Context, orderID, status string) error {
	query := `
		UPDATE orders 
		SET status = $1, status_updated_at = NOW(), updated_at = NOW() 
		WHERE order_id = $2
	`
	_, err := c.db.ExecContext(ctx, query, status, orderID)
	return err
}

func (c *DbClient) InsertInteractionLog(
	ctx context.Context,
	orderID, driverID string,
	attemptCount int,
	driverLat, driverLon, distance, fatigue float64,
	isAccepted int,
) error {
	query := `
		INSERT INTO interaction_logs (
			order_id, driver_id, attempt_count, driver_lat, driver_lon,
			driver_distance_to_pickup, driver_fatigue_index, is_accepted, offered_at
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
	`
	_, err := c.db.ExecContext(ctx, query,
		orderID, driverID, attemptCount, driverLat, driverLon,
		distance, fatigue, isAccepted,
	)
	return err
}

func (c *DbClient) UpdateInteractionLogStatus(ctx context.Context, orderID string, attemptCount int, isAccepted int) error {
	query := `
		UPDATE interaction_logs 
		SET is_accepted = $1 
		WHERE order_id = $2 AND attempt_count = $3
	`
	_, err := c.db.ExecContext(ctx, query, isAccepted, orderID, attemptCount)
	return err
}

func (c *DbClient) InsertStateTransition(
	ctx context.Context,
	orderID string,
	fromStatus, toStatus sql.NullString,
	driverID sql.NullString,
	attemptCount sql.NullInt32,
	reason sql.NullString,
) error {
	query := `
		INSERT INTO order_state_transitions (
			order_id, from_status, to_status, driver_id, attempt_count, reason, created_at
		) VALUES ($1, $2, $3, $4, $5, $6, NOW())
	`
	_, err := c.db.ExecContext(ctx, query,
		orderID, fromStatus, toStatus, driverID, attemptCount, reason,
	)
	return err
}
