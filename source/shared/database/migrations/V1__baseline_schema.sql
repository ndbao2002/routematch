-- Baseline schema for the microservices migration.
-- Mirrors the current operational and training schema.

CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE drivers (
    driver_id VARCHAR(20) PRIMARY KEY,
    joined_date TIMESTAMP,
    vehicle_type VARCHAR(20),
    max_load_kg INT
);

CREATE TABLE orders (
    order_id VARCHAR(20) PRIMARY KEY,
    user_id VARCHAR(20),
    created_at TIMESTAMP,
    status VARCHAR(32) NOT NULL DEFAULT 'accumulating',
    status_updated_at TIMESTAMP,
    pickup_lat DOUBLE PRECISION,
    pickup_lon DOUBLE PRECISION,
    dropoff_lat DOUBLE PRECISION,
    dropoff_lon DOUBLE PRECISION,
    shipping_fee DOUBLE PRECISION,
    cod_amount DOUBLE PRECISION,
    distance_km DOUBLE PRECISION,
    requested_vehicle_type VARCHAR(20),
    service_type VARCHAR(20),
    is_raining BOOLEAN,
    hour_of_day INT
);

CREATE TABLE interaction_logs (
    interaction_id SERIAL PRIMARY KEY,
    order_id VARCHAR(20) REFERENCES orders(order_id),
    driver_id VARCHAR(20) REFERENCES drivers(driver_id),
    attempt_count INT NOT NULL DEFAULT 1,
    driver_lat DOUBLE PRECISION,
    driver_lon DOUBLE PRECISION,
    driver_distance_to_pickup DOUBLE PRECISION,
    driver_fatigue_index FLOAT,
    is_accepted INT,
    offered_at TIMESTAMP
);

CREATE TABLE order_state_transitions (
    transition_id SERIAL PRIMARY KEY,
    order_id VARCHAR(20) REFERENCES orders(order_id),
    from_status VARCHAR(32),
    to_status VARCHAR(32) NOT NULL,
    driver_id VARCHAR(20) REFERENCES drivers(driver_id),
    attempt_count INT,
    reason VARCHAR(255),
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_orders_pickup ON orders USING gist (ST_MakePoint(pickup_lon, pickup_lat));
CREATE INDEX idx_drivers_vehicle ON drivers(vehicle_type);
CREATE INDEX idx_interactions_driver ON interaction_logs(driver_id);
CREATE INDEX idx_interactions_order_attempt ON interaction_logs(order_id, attempt_count);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_order_state_transitions_order ON order_state_transitions(order_id, created_at);
