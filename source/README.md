# Source Layout (Migration Draft)

This folder is reserved for the step-by-step microservices migration described in MIGRATION.md.
The existing Python monolith under src/ remains untouched until services are cut over.

- services/: New service codebases (Go and Python) aligned to the migration plan.
- shared/: Cross-service contracts, protobufs, and shared client libraries.
- shared/database/: Flyway versioned schema and operational migrations for the microservices.
- infrastructure/: Docker Compose stack for Postgres, Flyway, Redis, and Kafka.
