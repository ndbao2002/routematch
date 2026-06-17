# Stage 1: Build
FROM golang:1.23-alpine AS builder


# h3-go requires CGO_ENABLED=1 and gcc/make (build-base)
RUN apk add --no-cache build-base git

WORKDIR /workspace

# Copy shared library first for module resolution
COPY shared/libs/go/ shared/libs/go/

# Copy driver-state service
COPY services/driver-state/ services/driver-state/

WORKDIR /workspace/services/driver-state

# Compile the binary
RUN CGO_ENABLED=1 GOOS=linux go build -o driver-state cmd/server/main.go

# Stage 2: Runtime
FROM alpine:latest

# Install dynamic link compatibility for Go binary
RUN apk add --no-cache libc6-compat

WORKDIR /app

COPY --from=builder /workspace/services/driver-state/driver-state .

ENV PORT=50051
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379

EXPOSE 50051

CMD ["./driver-state"]
