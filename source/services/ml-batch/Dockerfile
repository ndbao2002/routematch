FROM python:3.11-slim

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Astral UV for fast package installations
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the pyproject.toml configuration first
COPY services/ml-batch/pyproject.toml services/ml-batch/

# Install the Python dependencies directly to the system python environment
RUN uv pip install --system -r services/ml-batch/pyproject.toml

# Copy the shared python protobuf library
COPY shared/libs/python/ shared/libs/python/

# Copy the ml-batch service directory
COPY services/ml-batch/ services/ml-batch/

WORKDIR /workspace/services/ml-batch

# Setup python path to allow importing pb package and app package
ENV PYTHONPATH=/workspace/shared/libs/python:/workspace/shared/libs/python/pb:/workspace/services/ml-batch

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
