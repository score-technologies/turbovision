# ---- Builder Stage ----
FROM python:3.12-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Install build dependencies in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    python3-dev \
    libffi-dev \
    libssl-dev \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /wheels

# Copy project metadata and source (limits cache invalidation to relevant files)
COPY pyproject.toml uv.lock README.md ./
COPY scorevision ./scorevision

# Build wheels for dependencies
RUN pip install --upgrade pip wheel setuptools hatchling && \
    pip wheel --wheel-dir /wheels .

# ---- Runtime Stage ----
FROM python:3.12-slim-bookworm

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/root/.local/bin:$PATH"

# Install only runtime dependencies (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install prebuilt wheels (fast, no compilation)
COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links=/wheels scorevision && \
    rm -rf /wheels

# Copy application code (separate layer for better caching)
COPY . /app

# Default command (can be overridden in docker-compose)
CMD ["sv", "-v", "validate"]
