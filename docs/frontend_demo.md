# Frontend Demo

This repository includes a lightweight React + Vite frontend for the FastAPI recommendation service. It is intended as a small interactive demo for:

- requesting recommendations for a `user_id`
- submitting `Like / Long View / Skip` feedback
- observing recommendation changes after feedback
- checking health, metrics, and request latency in one page

It does **not** implement real video playback, authentication, or production frontend concerns.

## What the demo shows

The page is built around the existing serving API:

- `GET /health`
- `POST /recommend`
- `POST /feedback`
- `GET /metrics`

The interaction loop is:

1. enter `user_id`, `top_k`, `device`, and `hour`
2. click **Get Recommendations**
3. inspect returned items and latency
4. click **Like**, **Long View**, or **Skip**
5. the frontend submits `/feedback`
6. the page refreshes `/recommend`
7. if the previous item disappears, the page highlights that recent-viewed filtering was applied

## Backend dependency

Start the FastAPI backend first:

```bash
python scripts/11_run_serving.py \
  --config configs/serving.yaml \
  --processed-dir ./processed \
  --artifacts-dir ./artifacts \
  --host 127.0.0.1 \
  --port 8000
```

The backend now enables CORS for local Vite development by default:

- `http://localhost:5173`
- `http://127.0.0.1:5173`

These origins are configured in [configs/serving.yaml](../configs/serving.yaml).

## Frontend startup

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

## API base URL

The frontend reads the backend base URL from:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Examples:

```bash
cd frontend
VITE_API_BASE_URL=http://127.0.0.1:8000 npm run dev
```

or create `frontend/.env.local`:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```

## Feedback presets

Each recommendation card provides three actions:

- `Like`
  - `watch_time=18`
  - `duration=20`
  - `click=1`
  - `like=1`
- `Long View`
  - `watch_time=16`
  - `duration=20`
  - `click=1`
  - `like=0`
- `Skip`
  - `watch_time=1`
  - `duration=20`
  - `click=0`
  - `like=0`

These are lightweight simulation presets for the current `/feedback` endpoint.

## How to observe recommendation changes

The clearest sequence is:

1. request recommendations for a user
2. click `Like` or `Long View` on the top item
3. wait for auto-refresh
4. inspect whether the previous `video_id` disappears from the refreshed TopK
5. inspect the feedback panel for `cache_invalidated=true`
6. inspect the metrics panel for updated request and feedback counters

If the item disappears, the page shows a message indicating recent-viewed filtering applied.

## Common issues

### CORS error

Likely causes:

- backend is running with a different config and CORS is disabled
- frontend origin is not in `serving.cors.allow_origins`
- backend is not listening on the host/port expected by the browser

### Backend not running

If `/health` fails in the page:

- confirm `scripts/11_run_serving.py` is running
- verify `http://127.0.0.1:8000/health` in the browser or with `curl`

### Wrong API base URL

If the frontend loads but API calls fail:

- check `VITE_API_BASE_URL`
- restart `npm run dev` after changing the environment variable

### Recommendations do not change after feedback

Possible reasons:

- the same user has too few viable candidates after filtering
- popular fallback reintroduces similar head items
- the service cache was not invalidated because feedback failed
- the request is hitting a different backend instance than the one receiving feedback

### Metrics look flat

The metrics endpoint is in-process. If the backend restarts, counters reset.

## Optional Docker Compose frontend

`docker-compose.yml` includes an optional `frontend` service for local demo use. The API still needs mounted `processed/` and `artifacts/` data.

Example:

```bash
docker compose up --build recommender-api redis frontend
```

The browser-facing frontend service uses:

```bash
VITE_API_BASE_URL=http://localhost:8000
```

## Scope note

This frontend is intentionally lightweight. Its purpose is to demonstrate the existing recommendation pipeline and feedback loop, not to represent a production web client.
