# Use the official Astral image which has Python + uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1

# Install system dependencies (libgomp1 is needed for XGBoost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock* /app/

# Install dependencies
RUN uv sync --frozen --no-dev --no-install-project

# Copy app code
COPY . /app

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]