# Stage 1: Build
FROM golang:1.21-alpine AS builder

WORKDIR /workspace

# Copy dispatch-engine service
COPY services/dispatch-engine/ services/dispatch-engine/

WORKDIR /workspace/services/dispatch-engine

# Compile the static binary
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-w -s" -o dispatch-engine cmd/main.go

# Stage 2: Runtime
FROM alpine:latest

WORKDIR /app

COPY --from=builder /workspace/services/dispatch-engine/dispatch-engine .

ENV KAFKA_BROKERS=kafka:9092
ENV REDIS_ADDR=redis:6379
ENV DB_CONN_STR=postgres://postgres:postgres@db:5432/routematch?sslmode=disable
ENV OFFER_TIMEOUT_SECONDS=15
ENV MAX_ATTEMPTS=3

CMD ["./dispatch-engine"]
