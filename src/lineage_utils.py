from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_data_split_id(train_path: Path, holdout_path: Path) -> str:
    payload = {
        "train_path": str(train_path.resolve()),
        "holdout_path": str(holdout_path.resolve()),
        "train_sha256": file_sha256(train_path),
        "holdout_sha256": file_sha256(holdout_path),
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def write_lineage(path: Path, lineage: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lineage, indent=2), encoding="utf-8")


def read_lineage(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
