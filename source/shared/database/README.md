# Database Versioning

Flyway-managed schema migrations for the microservices are kept here.
This folder is intentionally separate from the legacy root-level app so the migration path can evolve without changing the old bootstrap flow.

- `migrations/`: ordered Flyway SQL migrations.
- `flyway.conf`: connection and migration settings for the microservice database lifecycle.
