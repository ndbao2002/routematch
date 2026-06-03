# Flyway Migrations

These migrations are the versioned database contract for the microservices.

Order:
- `V1__baseline_schema.sql` establishes the current shared schema.
- `V2__operational_state_and_monitoring.sql` adds forward-only operational fields.

Keep root-level legacy bootstrap files unchanged; the microservice migration path should evolve here instead.
