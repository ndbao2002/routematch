## 🗓️ Phase 1: Advanced Data Engineering & Simulation

**Objective:** Create a realistic data environment that simulates "Business Friction" (Rain, Traffic, Fatigue).

### 1.1 Relational Schema Design (PostgreSQL)

- Design normalized tables: `drivers`, `orders`, `interaction_logs` (logs of offers).
- Implement **PostGIS** extension for geospatial queries.
- **Key Challenge:** Design the `interaction_logs` table to capture *Negative Samples* (rejected offers), not just completed bookings.

### 1.2 Complex Event Simulation

#### 1.2.1 Driver Generation
- Generate 2,000 drivers with attributes:
    - `driver_id`: UUIDv4
    - `vehicle_type`: Enum (bike, truck_500, truck_1000)
    - `max_load_kg`: Integer (based on vehicle type)
    - `joined_date`: Random date within last 2 years

#### 1.2.2 Order Generation
- Generate 200,000 orders with attributes:
    - `order_id`: UUIDv4
    - `user_id`: random integer
    - `created_at`: Random timestamp within 30 days since 2024-01-01
    - `pickup_lat`, `pickup_lon`: Random coordinates within service area (~30km radius of Ho Chi Minh City center)
    - `dropoff_lat`, `dropoff_lon`: from 0.5km to 30km away from pickup location
    - `distance_km`: Calculated geodesic distance between pickup and dropoff
    - `shipping_fee`: Based on distance and vehicle type
    - `cod_amount`: Random amount (500,000 VND mean for bikes, 2,000,000 VND mean for trucks)
    - `requested_vehicle_type`: Enum (bike, truck_500, truck_1000)
    - `service_type`: Enum (standard, fast, prioritize)
    - `is_raining`: Boolean (based at 5%, increase differently across time slots)
    - `hour_of_day`: Extracted from `created_at`

#### 1.2.3 Interaction Log Simulation
- For each order, randomly select at most 5 drivers who are not at busy state (each order require plenty of time to complete), simulate offers one by one.
- For each offer, calculate acceptance probability based on:
    - **Distance to Pickup:** Closer drivers have higher acceptance rates.
    - **COD Friction:** High `cod_amount` reduces acceptance rate linearly, differently by bike and truck.
    - **Rain Effect:** If `is_raining`, broadly reduce Bike Driver acceptance rate.
    - **Service Pressure:** Reduce acceptance rate for `prioritize` service types due to increased urgency.
    - **Fatigue:** Simulate by randomly generating fatigue index for each driver; if `fatigue > 0.75` and `distance_km > 15km`, drastically reduce acceptance rate.
- **Output:** Generate ~320k+ interaction rows for robust training.

### 1.3 Online Feature Store (Redis)

- Design Redis Key Schema for O(1) access:
    - `driver:geo:{vehicle_type}` -> `GEOADD` (Lat, Lon) -> This design help retrieving process to be faster.
    - `driver:{id}:profile` -> Hash Map (Vehicle Type, Max Load, Joined Date)
    - `driver:{id}:state` -> Hash Map (Status, MinutesActive, FatigueIndex, CancelRate, OrdersCompleted)