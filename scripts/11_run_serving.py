from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.serving.app import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FastAPI online recommendation service.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "serving.yaml"),
        help="Path to serving YAML config.",
    )
    parser.add_argument("--processed-dir", default=None, help="Override processed data directory.")
    parser.add_argument("--artifacts-dir", default=None, help="Override artifacts output directory.")
    parser.add_argument("--host", default=None, help="Override serving.api.host.")
    parser.add_argument("--port", type=int, default=None, help="Override serving.api.port.")
    return parser.parse_args()


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - depends on runtime env.
        raise SystemExit("uvicorn is required to run the online serving API. Install uvicorn and fastapi first.") from exc

    args = parse_args()
    app = create_app(
        config_path=args.config,
        processed_dir=args.processed_dir,
        artifacts_dir=args.artifacts_dir,
    )
    serving_cfg = app.state.recommender.serving_cfg["serving"]["api"]
    host = os.environ.get("SERVING_HOST") or args.host or str(serving_cfg.get("host", "0.0.0.0"))
    port = int(os.environ.get("SERVING_PORT") or args.port or serving_cfg.get("port", 8000))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
