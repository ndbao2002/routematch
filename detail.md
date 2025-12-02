**Goal:** Build a high-throughput, low-latency Order Matching System using a Hybrid Architecture (Retrieval + Ranking) with a focus on Business Logic and Cold-Start handling.

## 🏗️ System Architecture Specification

The system follows a **Three-Stage Funnel** design pattern used in production logistics systems.

### 1. The Data Plane (Feature Store)

- **Offline Store (PostgreSQL):** Stores historical `Order`, `Driver`, and `Interaction` logs for training.
- **Online Store (Redis):** Stores real-time driver state (`location`, `status`, `session_features`) for sub-millisecond inference retrieval.

### 2. The Model Plane (The Funnel)

- **Stage 1: Candidate Retrieval (Geospatial + Heuristic):**
    - *Input:* 50,000+ Active Drivers.
    - *Technique:* PostGIS Geospatial Filtering (`ST_DWithin`) + Hard Business Filters (Vehicle Type).
    - *Output:* Top 200 Candidates.
- **Stage 2: Fine Ranking (ML Probability):**
    - *Input:* 200 Candidates.
    - *Technique:* **XGBoost / LightGBM** (Gradient Boosted Decision Trees).
    - *Output:* Probability of Acceptance ($P_{accept}$).
- **Stage 3: Re-Ranking (Agentic Logic):**
    - *Input:* Ranked List.
    - *Technique:* **Contextual Bandits (UCB)** for new drivers + Business Logic (e.g., "Ban drivers with >10% cancellation rate today").
    - *Output:* Final Dispatch Candidate.

### 3. The Serving Plane

- **API:** FastAPI (Python) running via Docker Compose.
- **Protocol:** RESTful API (JSON).

## 🗓️ Phase 1: Advanced Data Engineering & Simulation (Weeks 1-3)

**Objective:** Create a realistic data environment that simulates "Business Friction" (Rain, Traffic, Fatigue).

### 1.1 Relational Schema Design (PostgreSQL)

- Design normalized tables: `drivers`, `orders`, `assignments` (logs of offers).
- Implement **PostGIS** extension for geospatial queries.
- **Key Challenge:** Design the `assignments` table to capture *Negative Samples* (rejected offers), not just completed bookings.

### 1.2 Complex Event Simulation

- Develop a Python simulation script (`simpy` or custom loop) that generates data with **Causal Relationships**:
    - *Rain Effect:* If `weather=rain`, reduce Bike Driver acceptance rate by 40%.
    - *COD Friction:* If `cod_amount > 2,000,000 VND`, reduce acceptance rate linearly.
    - *Fatigue:* Track `minutes_online`; reduce acceptance probability as fatigue increases.
- **Output:** Generate 500k+ interaction rows for robust training.

### 1.3 Online Feature Store (Redis)

- Design Redis Key Schema for O(1) access:
    - `driver:{id}:geo` -> `GEOADD` (Lat, Lon)
    - `driver:{id}:static` -> Hash Map (Vehicle Type, Service Level)
    - `driver:{id}:dynamic` -> Hash Map (OrdersToday, CancelRate, CurrentFatigue)

## 🗓️ Phase 2: The Retrieval Engine (Weeks 4-6)

**Objective:** Efficiently filter 50k drivers down to 200 candidates in < 10ms.

### 2.1 Geospatial Indexing

- Implement `PostGIS` queries to find "Drivers within Radius R".
- Implement **Dynamic Radius Expansion**:
    - Logic: If < 5 drivers found in 1km, automatically expand to 3km, then 5km.

### 2.2 Feature Fetching Pipeline

- Build a Python `FeatureStore` client.
- Implement **Vectorized Fetching**: Fetch features for all 200 candidates in a single Redis Pipeline batch request (minimizing RTT).
- *Metric:* Measure "Time to Fetch Features" (Target: < 15ms).

## 🗓️ Phase 3: The Ranking Engine (Weeks 7-10)

**Objective:** Accurately predict `P(Accept)` using Tabular ML.

### 3.1 Feature Engineering (The "Mid-Level" Skill)

- Create **Interaction Features** (The most powerful features in matching):
    - `distance_to_pickup`: Geodesic distance between Driver and Order.
    - `price_per_km`: Unit economics of the order.
    - `driver_acceptance_rate_7d`: Rolling window aggregate.
    - `district_affinity`: Does this driver usually work in this District?

### 3.2 Model Training (XGBoost/LightGBM)

- Train a Binary Classifier.
- **Class Imbalance Handling:** Apply `scale_pos_weight` or Focal Loss if rejection rate is high.
- **Evaluation:**
    - Primary Metric: **AUC-ROC**.
    - Business Metric: **Recall@5** (Is the acceptor in the top 5 predictions?).

### 3.3 Model Calibration

- Apply **Isotonic Regression** or **Platt Scaling** to raw model outputs.
- *Why:* We need the score 0.7 to mathematically mean "70% probability" for the Expectation maximization formula.

### 3.4 Explainability

- Implement **SHAP (Shapley Additive Explanations)**.
- Generate plots showing top factors driving rejections (e.g., "High COD" or "Long Distance").

## 🗓️ Phase 4: The "Agentic" Logic & Cold Start (Weeks 11-13)

**Objective:** Solve the "New Driver Problem" and apply business constraints.

### 4.1 Contextual Bandit Layer (UCB)

- Implement the **Upper Confidence Bound (UCB)** formula.
- Logic:
    - Calculate `Exploration_Bonus = Alpha * sqrt(log(Total_Requests) / (Driver_Requests + 1))`.
    - `Final_Score = XGBoost_Score + Exploration_Bonus`.
- *Effect:* New drivers (low `Driver_Requests`) get a score boost, allowing the system to "test" them.

### 4.2 Business Rules Engine

- Implement a rigid filter *after* ranking:
    - **Fraud Check:** Filter if driver is moving > 80km/h.
    - **Service Level:** Filter if Order is "Premium" but Driver is "Standard".
    - **Hoarding:** Filter if Driver canceled an order in the last 5 minutes.

## 🗓️ Phase 5: Production Engineering (Weeks 14-16)

**Objective:** Deploy as a robust, observable microservice.

### 5.1 FastAPI Microservice

- Endpoint: `POST /v1/dispatch`.
- Payload: `Order Details`.
- Response: `{"driver_id": "D_123", "score": 0.85, "debug_info": {...}}`.
- Optimization: Use `async`/`await` for Redis/Postgres calls to handle concurrency.

### 5.2 Docker Compose Orchestration

- Containerize the Application.
- Define services: `db` (Postgres), `cache` (Redis), `api` (FastAPI), `simulator` (Python script generating load).

### 5.3 Load Testing & Latency Profiling

- Use **Locust** to simulate 50-100 orders per second.
- Profile the API: Breakdown latency into Retrieval time vs. Inference time vs. Network time.
- *Goal:* P99 Latency < 200ms.

### 5.4 Monitoring Dashboard (Streamlit)

- Build a "Control Tower" dashboard.
- **Visuals:** Real-time map of Orders vs. Drivers.
- **Metrics:** "Matching Efficiency" (Percentage of orders accepted by Top 1 candidate).