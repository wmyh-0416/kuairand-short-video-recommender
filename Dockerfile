FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-serving.txt /app/

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r /app/requirements-serving.txt

COPY . /app

RUN mkdir -p /app/artifacts /app/processed

EXPOSE 8000

CMD ["python", "scripts/11_run_serving.py", "--config", "configs/serving.yaml", "--processed-dir", "/app/processed", "--artifacts-dir", "/app/artifacts", "--host", "0.0.0.0", "--port", "8000"]

