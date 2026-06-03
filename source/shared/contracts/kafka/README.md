# Kafka Contracts

Versioned Kafka event schemas for the microservices migration.

Design rules:
- Use a common envelope for all events.
- Keep payload schemas versioned by topic and contract version.
- Add only forward-compatible fields in new versions.
- Prefer explicit topic-level schemas over one large shared schema.

Current version set:
- `v1/` for the initial migration contract.
