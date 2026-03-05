from __future__ import annotations

import json
from pathlib import Path

import joblib
import pytest
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor

from src import api_server
from src.build_features import build_features, extract_category_meta

pytest.importorskip("httpx")
from fastapi.testclient import TestClient


def _parsed_record(i: int) -> dict:
    return {
        "price": float(700 + i * 30),
        "discount_amount": None,
        "currency": "GBP",
        "brand": ["BrandA", "BrandB", "BrandC"][i % 3],
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


def _write_train_jsonl(path: Path, n: int = 30) -> list[dict]:
    rows = [{"parsed_json": _parsed_record(i)} for i in range(n)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return [r["parsed_json"] for r in rows]


def test_api_server_schema_and_predict_smoke(tmp_path) -> None:
    train_jsonl = tmp_path / "train.jsonl"
    records = _write_train_jsonl(train_jsonl)

    train_df = build_features(records, keep_price=True)
    y = train_df.pop("price")
    category_meta = extract_category_meta(train_df)

    model = HistGradientBoostingRegressor(random_state=42, max_iter=10)
    model.fit(train_df, y)

    model_out = tmp_path / "best_model.pkl"
    meta_out = tmp_path / "category_meta.json"
    lineage_out = tmp_path / "lineage.json"
    cfg_path = tmp_path / "defaults.yaml"

    joblib.dump(model, model_out)
    meta_out.write_text(json.dumps(category_meta), encoding="utf-8")
    lineage_out.write_text(
        json.dumps(
            {
                "train_run_id": "train-123",
                "data_split_id": "split-123",
                "model_version": "model-123",
            }
        ),
        encoding="utf-8",
    )
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "train_jsonl": str(train_jsonl.resolve()),
                    "model_out": str(model_out.resolve()),
                    "meta_out": str(meta_out.resolve()),
                    "lineage_out": str(lineage_out.resolve()),
                }
            }
        ),
        encoding="utf-8",
    )

    app = api_server.create_app(cfg_path)
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        ready = client.get("/readyz")
        assert ready.status_code == 200
        assert ready.json()["ready"] is True

        schema = client.get("/v1/schema")
        assert schema.status_code == 200
        payload = schema.json()
        assert "width" in payload["numeric_fields"]
        assert "brand" in payload["categorical_fields_all"]
        assert "brand" in payload["categorical_fields_top"]
        assert payload["top_k_categories"] == 10
        assert len(payload["categorical_fields_top"]["brand"]) <= payload["top_k_categories"]

        ui = client.get("/")
        assert ui.status_code == 200

        pred = client.post(
            "/v1/predict",
            json={
                "numeric_field": "width",
                "numeric_value": 210.0,
                "categorical_field": "brand",
                "categorical_value": "BrandA",
            },
        )
        assert pred.status_code == 200
        body = pred.json()
        assert isinstance(body["predicted_price"], float)
        assert body["model_version"] == "model-123"

        bad_numeric = client.post(
            "/v1/predict",
            json={
                "numeric_field": "width",
                "numeric_value": 0.0,
                "categorical_field": "brand",
                "categorical_value": "BrandA",
            },
        )
        assert bad_numeric.status_code == 422

        bad_category = client.post(
            "/v1/predict",
            json={
                "numeric_field": "width",
                "numeric_value": 210.0,
                "categorical_field": "brand",
                "categorical_value": "NotInTraining",
            },
        )
        assert bad_category.status_code == 400
