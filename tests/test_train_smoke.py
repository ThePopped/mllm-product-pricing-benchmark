from __future__ import annotations

import json
import sys
from pathlib import Path

from src import train
from src.lineage_utils import read_lineage


def _parsed_record(i: int) -> dict:
    return {
        "price": float(500 + i * 35),
        "discount_amount": None,
        "currency": "GBP",
        "brand": f"Brand{i % 4}",
        "model": f"Model{i}",
        "colour_1": ["grey", "blue", "beige"][i % 3],
        "colour_2": None,
        "colour_3": None,
        "material_1": "fabric" if i % 2 == 0 else "leather",
        "material_2": None,
        "reclining": bool(i % 2),
        "pull_out_bed": False,
        "storage": bool((i + 1) % 2),
        "seat_number": 2 + (i % 4),
        "height": float(85 + (i % 5)),
        "width": float(180 + i),
        "depth": float(90 + (i % 7)),
        "product_count": 1,
        "additional_features": [],
        "evidence": [],
    }


def _write_jsonl(path: Path, start: int, n: int) -> None:
    lines = []
    for i in range(start, start + n):
        lines.append(json.dumps({"parsed_json": _parsed_record(i)}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_train_smoke_produces_model_meta_and_lineage(tmp_path, monkeypatch) -> None:
    train_data = tmp_path / "train.jsonl"
    holdout_data = tmp_path / "holdout.jsonl"
    model_out = tmp_path / "best_model.pkl"
    meta_out = tmp_path / "category_meta.json"
    lineage_out = tmp_path / "lineage.json"

    _write_jsonl(train_data, start=0, n=30)
    _write_jsonl(holdout_data, start=100, n=10)

    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").resolve().as_uri())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--data",
            str(train_data),
            "--holdout-data",
            str(holdout_data),
            "--model-out",
            str(model_out),
            "--meta-out",
            str(meta_out),
            "--lineage-out",
            str(lineage_out),
            "--n-iter",
            "1",
            "--price-cap",
            "10000",
        ],
    )

    train.main()

    assert model_out.exists()
    assert meta_out.exists()
    assert lineage_out.exists()
    lineage = read_lineage(lineage_out)
    for key in ("train_run_id", "data_split_id", "model_version"):
        assert key in lineage
        assert lineage[key]
