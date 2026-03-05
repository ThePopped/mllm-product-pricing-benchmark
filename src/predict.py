from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import mlflow

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_features import build_features
from lineage_utils import read_lineage
from project_config import DEFAULT_CONFIG, cfg_path, load_config

DEFAULT_INPUT = ROOT / "data" / "processed" / "holdout.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "predictions.jsonl"
DEFAULT_MODEL = ROOT / "models" / "best_model.pkl"
DEFAULT_META = ROOT / "models" / "category_meta.json"
DEFAULT_LINEAGE = ROOT / "models" / "lineage.json"


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    pre_args, _ = pre.parse_known_args()
    cfg = load_config(pre_args.config)
    paths_cfg = cfg.get("paths", {})
    inference_cfg = cfg.get("inference", {})

    p = argparse.ArgumentParser(description="Batch inference entrypoint for sofa price predictions.")
    p.add_argument("--config", type=Path, default=pre_args.config)
    p.add_argument("--input", type=Path, default=cfg_path(ROOT, paths_cfg.get("inference_input_jsonl"), DEFAULT_INPUT))
    p.add_argument("--output", type=Path, default=cfg_path(ROOT, paths_cfg.get("inference_output_jsonl"), DEFAULT_OUTPUT))
    p.add_argument("--model", type=Path, default=cfg_path(ROOT, paths_cfg.get("model_out"), DEFAULT_MODEL))
    p.add_argument("--meta", type=Path, default=cfg_path(ROOT, paths_cfg.get("meta_out"), DEFAULT_META))
    p.add_argument("--lineage-file", type=Path, default=cfg_path(ROOT, paths_cfg.get("lineage_out"), DEFAULT_LINEAGE))
    p.add_argument("--experiment", type=str, default=str(inference_cfg.get("experiment", "sofa_price_regression")))
    p.add_argument("--run-name", type=str, default=str(inference_cfg.get("run_name", "batch_inference")))
    return p.parse_args()


def _parse_input_entry(entry: Any, idx: int) -> tuple[dict, dict[str, Any]]:
    if isinstance(entry, dict) and isinstance(entry.get("parsed_json"), dict):
        meta = {"input_index": idx}
        for key in ("id", "image_file", "image_path", "url"):
            if key in entry:
                meta[key] = entry[key]
        return entry["parsed_json"], meta
    if isinstance(entry, dict):
        meta = {"input_index": idx}
        for key in ("id", "image_file", "image_path", "url"):
            if key in entry:
                meta[key] = entry[key]
        return entry, meta
    raise TypeError(f"Unsupported input entry type at index {idx}: {type(entry)}")


def load_inputs(path: Path) -> tuple[list[dict], list[dict[str, Any]]]:
    records: list[dict] = []
    metas: list[dict[str, Any]] = []

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            items = payload.get("records")
            if not isinstance(items, list):
                raise ValueError("JSON object input must contain a list under key 'records'.")
        elif isinstance(payload, list):
            items = payload
        else:
            raise ValueError("JSON input must be a list or {'records': [...]} object.")
        for i, item in enumerate(items):
            rec, meta = _parse_input_entry(item, i)
            records.append(rec)
            metas.append(meta)
        return records, metas

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            rec, meta = _parse_input_entry(entry, i)
            records.append(rec)
            metas.append(meta)
    return records, metas


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    records, metas = load_inputs(args.input)
    if not records:
        raise ValueError(f"No records found in input file: {args.input}")

    model = joblib.load(args.model)
    category_meta = json.loads(args.meta.read_text(encoding="utf-8"))
    X = build_features(records, category_meta=category_meta, keep_price=False)
    preds = model.predict(X)

    lineage = read_lineage(args.lineage_file) if args.lineage_file.exists() else {}
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=args.run_name):
        mlflow.set_tag("stage", "inference")
        if lineage.get("train_run_id"):
            mlflow.set_tag("train_run_id", lineage["train_run_id"])
            mlflow.log_param("train_run_id", lineage["train_run_id"])
        if lineage.get("data_split_id"):
            mlflow.set_tag("data_split_id", lineage["data_split_id"])
            mlflow.log_param("data_split_id", lineage["data_split_id"])
        if lineage.get("model_version"):
            mlflow.set_tag("model_version", lineage["model_version"])
            mlflow.log_param("model_version", lineage["model_version"])

        mlflow.log_param("input_path", str(args.input))
        mlflow.log_param("output_path", str(args.output))
        mlflow.log_param("model_path", str(args.model))
        mlflow.log_metric("num_records", len(records))

        with args.output.open("w", encoding="utf-8") as f:
            for i, (pred, meta) in enumerate(zip(preds, metas)):
                row = {"predicted_price": float(pred), **meta}
                if "id" not in row:
                    row["id"] = f"row_{i}"
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        mlflow.log_artifact(str(args.output), artifact_path="inference")

    print(f"Predictions written → {args.output}")


if __name__ == "__main__":
    main()
