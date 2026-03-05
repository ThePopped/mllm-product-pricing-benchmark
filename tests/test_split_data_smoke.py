from __future__ import annotations

import json
import sys
from pathlib import Path

from src import split_data


def _parsed_record(i: int) -> dict:
    return {
        "price": float(400 + i * 25),
        "discount_amount": None,
        "currency": "GBP",
        "brand": f"Brand{i % 3}",
        "model": f"Model{i}",
        "colour_1": "grey",
        "colour_2": None,
        "colour_3": None,
        "material_1": "fabric",
        "material_2": None,
        "reclining": False,
        "pull_out_bed": False,
        "storage": False,
        "seat_number": 3,
        "height": 90.0,
        "width": 200.0,
        "depth": 95.0,
        "product_count": 1,
        "additional_features": [],
        "evidence": [],
    }


def _write_source_jsonl(path: Path, n: int = 20) -> None:
    lines = []
    for i in range(n):
        lines.append(json.dumps({"parsed_json": _parsed_record(i)}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_split_data_writes_train_and_holdout(tmp_path, monkeypatch) -> None:
    src_path = tmp_path / "source.jsonl"
    train_out = tmp_path / "train.jsonl"
    holdout_out = tmp_path / "holdout.jsonl"
    _write_source_jsonl(src_path, n=24)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "split_data.py",
            "--src",
            str(src_path),
            "--train-out",
            str(train_out),
            "--holdout-out",
            str(holdout_out),
            "--holdout-size",
            "0.2",
            "--seed",
            "42",
        ],
    )

    split_data.main()

    assert train_out.exists()
    assert holdout_out.exists()
    train_lines = [l for l in train_out.read_text(encoding="utf-8").splitlines() if l.strip()]
    holdout_lines = [l for l in holdout_out.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(train_lines) > 0
    assert len(holdout_lines) > 0
    assert len(train_lines) + len(holdout_lines) == 24
