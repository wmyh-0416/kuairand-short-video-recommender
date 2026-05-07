from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional
from urllib import error, request


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "serving" / "benchmark_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the online recommender service.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Serving API base URL.")
    parser.add_argument("--num-requests", type=int, default=50, help="Total number of recommend requests.")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrent workers.")
    parser.add_argument("--user-ids", default=None, help="Comma-separated user ids. Defaults to 0..99.")
    parser.add_argument("--top-k", type=int, default=5, help="TopK per recommend request.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Benchmark report output path.")
    return parser.parse_args()


def request_json(method: str, url: str, payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = request.Request(url=url, data=data, headers=headers, method=method.upper())
    with request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (float(p) / 100.0)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return float(ordered[lower])
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def run_single_request(base_url: str, user_id: str, top_k: int, idx: int) -> dict[str, Any]:
    payload = {
        "user_id": str(user_id),
        "top_k": int(top_k),
        "request_id": f"bench-{idx}",
        "context": {"device": "ios", "hour": str(10 + (idx % 10)), "scenario": "benchmark"},
    }
    started = time.perf_counter()
    try:
        response = request_json("POST", f"{base_url}/recommend", payload=payload)
        latency_ms = float((time.perf_counter() - started) * 1000.0)
        return {
            "ok": True,
            "latency_ms": latency_ms,
            "response_latency_ms": float(response.get("latency_ms", {}).get("total", 0.0)),
            "returned_items": len(response.get("items", [])),
            "degraded_mode": bool(response.get("degraded_mode", False)),
        }
    except Exception as exc:
        latency_ms = float((time.perf_counter() - started) * 1000.0)
        return {"ok": False, "latency_ms": latency_ms, "error": str(exc)}


def main() -> None:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.user_ids:
        user_ids = [part.strip() for part in str(args.user_ids).split(",") if part.strip()]
    else:
        user_ids = [str(i) for i in range(100)]

    health = request_json("GET", f"{base_url}/health")
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.concurrency))) as pool:
        futures = [
            pool.submit(
                run_single_request,
                base_url,
                random.choice(user_ids),
                int(args.top_k),
                idx,
            )
            for idx in range(int(args.num_requests))
        ]
        for future in as_completed(futures):
            results.append(future.result())
    wall_time_s = float(time.perf_counter() - started)

    metrics = {}
    try:
        metrics = request_json("GET", f"{base_url}/metrics")
    except (error.URLError, error.HTTPError, json.JSONDecodeError):
        metrics = {}

    success_results = [row for row in results if row.get("ok")]
    error_results = [row for row in results if not row.get("ok")]
    latencies = [float(row.get("latency_ms", 0.0)) for row in success_results]
    degraded_count = sum(1 for row in success_results if row.get("degraded_mode"))

    report = {
        "base_url": base_url,
        "health": health,
        "total_requests": int(args.num_requests),
        "concurrency": int(args.concurrency),
        "success_count": len(success_results),
        "error_count": len(error_results),
        "qps": float(len(success_results) / max(wall_time_s, 1e-9)),
        "mean_latency_ms": float(sum(latencies) / max(len(latencies), 1)) if latencies else 0.0,
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "p99_latency_ms": percentile(latencies, 99),
        "min_latency_ms": float(min(latencies)) if latencies else 0.0,
        "max_latency_ms": float(max(latencies)) if latencies else 0.0,
        "degraded_count": int(degraded_count),
        "cache_hit_rate_from_server_metrics": float(metrics.get("cache_hit_rate", 0.0)) if metrics else 0.0,
        "server_metrics_snapshot": metrics,
        "sample_errors": [row.get("error", "") for row in error_results[:5]],
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "qps": report["qps"], "p95_latency_ms": report["p95_latency_ms"], "success_count": report["success_count"], "error_count": report["error_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
