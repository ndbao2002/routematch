-- Enable PostGIS for geospatial calculation (ST_Distance, ST_DWithin)
CREATE EXTENSION IF NOT EXISTS postgis;

-- =============================================
-- 1. DRIVERS TABLE (Dimension)
-- =============================================
CREATE TABLE drivers (
    driver_id VARCHAR(20) PRIMARY KEY,
    joined_date TIMESTAMP,
    
    -- Asset Type (Hard Constraint)
    -- 'bike'       : < 30kg (Fast, cheap, weather-sensitive)
    -- 'truck_500'  : < 500kg (Tricycle/Van)
    -- 'truck_1000' : < 1000kg (Truck)
    vehicle_type VARCHAR(20),
    max_load_kg INT
);

-- =============================================
-- 2. ORDERS TABLE (Fact Table)
-- =============================================
CREATE TABLE orders (
    order_id VARCHAR(20) PRIMARY KEY,
    user_id VARCHAR(20),
    created_at TIMESTAMP,
    
    -- Geospatial Data
    pickup_lat DOUBLE PRECISION,
    pickup_lon DOUBLE PRECISION,
    dropoff_lat DOUBLE PRECISION,
    dropoff_lon DOUBLE PRECISION,
    
    -- Economics
    shipping_fee DOUBLE PRECISION, -- Revenue (Platform + Driver)
    cod_amount DOUBLE PRECISION,   -- Cash Friction (Risk)
    
    -- Trip Meta
    distance_km DOUBLE PRECISION,
    
    -- Matching Constraints
    requested_vehicle_type VARCHAR(20), -- Must match driver.vehicle_type
    service_type VARCHAR(20),           -- 'standard', 'fast', 'prioritize'
    
    -- Context (Features)
    is_raining BOOLEAN,
    hour_of_day INT
);

-- =============================================
-- 3. INTERACTION LOGS (Training Labels)
-- Captures the "Offer -> Decision" moment
-- =============================================
CREATE TABLE interaction_logs (
    interaction_id SERIAL PRIMARY KEY,
    order_id VARCHAR(20) REFERENCES orders(order_id),
    driver_id VARCHAR(20) REFERENCES drivers(driver_id),
    
    -- State Context (What the driver saw)
    driver_lat DOUBLE PRECISION,
    driver_lon DOUBLE PRECISION,
    driver_distance_to_pickup DOUBLE PRECISION, 
    
    -- Driver State
    driver_fatigue_index FLOAT, -- 0.0 to 1.0
    
    -- Outcome
    is_accepted INT, -- 1 = Accepted, 0 = Rejected/Ignored
    
    offered_at TIMESTAMP
);

-- =============================================
-- 4. PERFORMANCE INDEXES
-- =============================================
-- Spatial Index for Phase 2 Retrieval
CREATE INDEX idx_orders_pickup ON orders USING gist (ST_MakePoint(pickup_lon, pickup_lat));

-- Feature Lookup Indexes
CREATE INDEX idx_drivers_vehicle ON drivers(vehicle_type);
CREATE INDEX idx_interactions_driver ON interaction_logs(driver_id);