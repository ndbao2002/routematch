FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# Copy dependency declaration for uv to install
COPY pyproject.toml uv.lock* /app/

# Use uv to sync dependencies into the image
RUN uv sync --no-dev --no-install-project || true

# Copy package sources
COPY ./order_gateway /app/order_gateway

ENV REDIS_HOST=redis
ENV REDIS_PORT=6379
ENV REDIS_TTL=300

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8080"]
