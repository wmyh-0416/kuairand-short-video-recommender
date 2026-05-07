# Deployment

## Local Start

```bash
cd /home/ym3447/kuairand-short-video-recommender

python scripts/11_run_serving.py \
  --config configs/serving.yaml \
  --processed-dir ./processed \
  --artifacts-dir ./artifacts \
  --host 127.0.0.1 \
  --port 8000
```

## Docker Compose Start

```bash
cd /home/ym3447/kuairand-short-video-recommender
docker compose up --build
```

This starts:

- `recommender-api`
- `redis`

By default, Redis is injected through:

- `REDIS_URL=redis://redis:6379/0`
- `SERVING_USE_REDIS=true`

## Validation

Health:

```bash
curl http://localhost:8000/health
```

Recommend:

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "0",
    "top_k": 5,
    "request_id": "deploy-rec-1",
    "context": {"device": "ios", "hour": "21"}
  }'
```

Feedback:

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "0",
    "video_id": "7136",
    "watch_time": 12.5,
    "duration": 20.0,
    "click": 1,
    "like": 1
  }'
```

Realtime feedback smoke test:

```bash
python scripts/12_test_realtime_feedback.py \
  --base-url http://localhost:8000 \
  --user-id 0 \
  --top-k 5
```

## Environment Variables

Supported overrides:

- `REDIS_URL`
- `SERVING_USE_REDIS`
- `SERVING_HOST`
- `SERVING_PORT`
- `KUAIRAND_PROCESSED_DIR`
- `KUAIRAND_ARTIFACTS_DIR`
- `KUAIRAND_SERVING_CONFIG`

Priority:

- environment variable
- CLI argument
- YAML default

## Common Issues

### `faiss-cpu` install failure

- Prefer `python:3.10-slim` or `python:3.11-slim`
- Rebuild without cache: `docker compose build --no-cache`
- If the wheel still fails, pin a compatible `faiss-cpu` version in `requirements-serving.txt`

### Artifacts do not exist

- The container expects existing offline outputs under mounted `./artifacts`
- At minimum, prepare the FAISS index, recall assets, and serving models before starting online serving
- If only some model artifacts are missing, the API can still start in degraded mode and `/health` will expose `component_errors`

### Redis connection failure

- Check `docker compose ps`
- Check `docker compose logs redis`
- Verify `REDIS_URL=redis://redis:6379/0`
- If Redis is unavailable, the API falls back to in-memory cache and `redis_connected` becomes `false`

### Docker path mismatch

- The API container reads:
  - `/app/processed`
  - `/app/artifacts`
  - `/app/configs`
- Keep the compose volume mounts aligned with local directories

### Large model artifacts

- Do not copy large `artifacts/` into the image layer
- Mount them with compose volumes instead
- `.dockerignore` already excludes `artifacts/` and `processed/` from the build context

## Degraded Mode

The API can still start when some serving artifacts are missing:

- missing prerank model: keep recall score as coarse score
- missing rank model: keep prerank order
- missing rerank config: return rank output directly
- missing user embedding: try on-demand Two-Tower encoding
- cold-start user: fall back to popular recall or mean-user embedding recall

This is still a lightweight deployable simulation, not a production training-feedback loop.
