from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional
from urllib import error, request


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_PATH = PROJECT_ROOT / "artifacts" / "serving" / "realtime_feedback_test_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test realtime feedback -> state update -> cache invalidation flow.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Serving API base URL.")
    parser.add_argument("--user-id", default="0", help="User id used for the smoke test.")
    parser.add_argument("--top-k", type=int, default=5, help="TopK used for recommend requests.")
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH), help="Output JSON report path.")
    return parser.parse_args()


def request_json(method: str, url: str, payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = request.Request(url=url, data=data, headers=headers, method=method.upper())
    try:
        with request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with status={exc.code}: {body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc}") from exc


def main() -> None:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    report_path = Path(args.report_path).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    health = request_json("GET", f"{base_url}/health")
    metrics_before = request_json("GET", f"{base_url}/metrics")
    recommend_before = request_json(
        "POST",
        f"{base_url}/recommend",
        payload={
            "user_id": args.user_id,
            "top_k": int(args.top_k),
            "request_id": "realtime-feedback-before",
            "context": {"device": "ios", "hour": "21", "scenario": "realtime_feedback_test"},
        },
    )
    if not recommend_before.get("items"):
        raise RuntimeError("First /recommend returned no items; cannot continue realtime feedback test.")
    top_item = recommend_before["items"][0]
    feedback = request_json(
        "POST",
        f"{base_url}/feedback",
        payload={
            "user_id": args.user_id,
            "video_id": str(top_item["video_id"]),
            "watch_time": 16.0,
            "duration": 20.0,
            "click": 1,
            "like": 1,
        },
    )
    recommend_after = request_json(
        "POST",
        f"{base_url}/recommend",
        payload={
            "user_id": args.user_id,
            "top_k": int(args.top_k),
            "request_id": "realtime-feedback-after",
            "context": {"device": "ios", "hour": "21", "scenario": "realtime_feedback_test"},
        },
    )
    metrics_after = request_json("GET", f"{base_url}/metrics")

    before_videos = [str(item["video_id"]) for item in recommend_before.get("items", [])]
    after_videos = [str(item["video_id"]) for item in recommend_after.get("items", [])]
    feedback_video_id = str(top_item["video_id"])

    checks = {
        "health_ok": health.get("status") == "ok",
        "feedback_item_removed": feedback_video_id not in after_videos,
        "feedback_count_increased": int(metrics_after.get("feedback_count", 0)) > int(metrics_before.get("feedback_count", 0)),
        "cache_invalidation_increased": int(metrics_after.get("cache_invalidation_count", 0)) > int(metrics_before.get("cache_invalidation_count", 0)),
        "recent_viewed_contains_video": bool(feedback.get("recent_viewed_contains_video", False)),
    }
    passed = all(checks.values())
    report = {
        "passed": bool(passed),
        "base_url": base_url,
        "user_id": str(args.user_id),
        "feedback_video_id": feedback_video_id,
        "before_videos": before_videos,
        "after_videos": after_videos,
        "checks": checks,
        "health": health,
        "metrics_before": metrics_before,
        "feedback_response": feedback,
        "metrics_after": metrics_after,
        "recommend_before": recommend_before,
        "recommend_after": recommend_after,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": passed, "report_path": str(report_path), "checks": checks}, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
