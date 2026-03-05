from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import api_server
from project_config import DEFAULT_CONFIG, load_config


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    pre_args, _ = pre.parse_known_args()
    cfg = load_config(pre_args.config)
    serving_cfg = cfg.get("serving", {})

    p = argparse.ArgumentParser(description="Run FastAPI prediction server.")
    p.add_argument("--config", type=Path, default=pre_args.config)
    p.add_argument("--host", type=str, default=str(serving_cfg.get("host", "0.0.0.0")))
    p.add_argument("--port", type=int, default=int(serving_cfg.get("port", 8000)))
    p.add_argument("--reload", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    app = api_server.create_app(args.config)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
