-- Forward-only operational additions for the microservices path.

ALTER TABLE orders
    ADD COLUMN IF NOT EXISTS last_event_id VARCHAR(64);

ALTER TABLE orders
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP NOT NULL DEFAULT NOW();

CREATE INDEX IF NOT EXISTS idx_orders_updated_at ON orders(updated_at);
