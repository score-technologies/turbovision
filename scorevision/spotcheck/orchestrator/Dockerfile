FROM python:3.12-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make python3-dev libffi-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /wheels
COPY pyproject.toml README.md ./
COPY scorevision ./scorevision
RUN pip install --upgrade pip wheel setuptools hatchling && \
    pip wheel --wheel-dir /wheels .

FROM python:3.12-slim-bookworm

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links=/wheels scorevision && \
    pip uninstall -y opencv-python && \
    pip install opencv-python-headless kubernetes && \
    rm -rf /wheels

ENTRYPOINT ["python", "-m", "scorevision.spotcheck.orchestrator.main"]
