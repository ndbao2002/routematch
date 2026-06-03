# Microservices Infrastructure

This stack is the first infrastructure layout for the migration under source:
- PostgreSQL with PostGIS
- Flyway migrations
- Redis
- Kafka (Redpanda-compatible)
- Redpanda Console for topic visibility

## Start

Run from source/infrastructure:

docker compose up -d db redis kafka

docker compose run --rm flyway

docker compose run --rm kafka-init

## Optional full boot

docker compose up -d

## Endpoints

- Postgres: localhost:5433
- Redis: localhost:6379
- Kafka external listener: localhost:19092
- Redpanda admin API: localhost:9644
- Pandaproxy: localhost:18082
- Redpanda Console: http://localhost:8081

## Notes

- Flyway reads config from source/shared/database/flyway.conf.
- Flyway migrations are loaded from source/shared/database/migrations.
- Topic names align with source/shared/contracts/kafka/v1/topic-matrix.md.
