FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./

RUN mkdir -p scorevision/miner scorevision/utils \
    && echo "" > scorevision/__init__.py \
    && echo "" > scorevision/miner/__init__.py \
    && echo "" > scorevision/utils/__init__.py

RUN pip install --no-cache-dir -e .

RUN pip install --no-cache-dir --no-deps \
    "fiber @ git+https://github.com/rayonlabs/fiber.git@a41ab890708757140a3cf4aae8e5af57a8b03159"

COPY scorevision/miner/private_track/ ./scorevision/miner/private_track/
COPY scorevision/utils/logging.py scorevision/utils/schemas.py ./scorevision/utils/

EXPOSE 8000

CMD ["uvicorn", "scorevision.miner.private_track.server:app", "--host", "0.0.0.0", "--port", "8000"]
