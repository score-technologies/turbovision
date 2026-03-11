FROM python:3.12-slim-bookworm

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make python3-dev libffi-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip && \
    pip install httpx pydantic python-dotenv

COPY scorevision ./scorevision
RUN echo "" > scorevision/__init__.py && \
    echo "" > scorevision/validator/__init__.py && \
    echo "" > scorevision/validator/central/__init__.py

ENTRYPOINT ["python", "-m", "scorevision.spotcheck.runner.main"]
