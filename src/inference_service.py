from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from build_features import build_features


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    category_meta: dict[str, list]
    lineage: dict[str, Any]


def load_bundle(model_path: Path, meta_path: Path, lineage_path: Path) -> ModelBundle:
    model = joblib.load(model_path)
    category_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    lineage = json.loads(lineage_path.read_text(encoding="utf-8")) if lineage_path.exists() else {}
    return ModelBundle(model=model, category_meta=category_meta, lineage=lineage)


def _align_columns_for_model(model: Any, features: pd.DataFrame) -> pd.DataFrame:
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        return features
    names = list(feature_names)
    missing = [name for name in names if name not in features.columns]
    if missing:
        raise ValueError(f"Missing required model input columns: {missing}")
    return features.loc[:, names]


def predict_records(bundle: ModelBundle, records: list[dict]) -> list[float]:
    if not records:
        raise ValueError("No records provided for prediction.")
    features = build_features(records, category_meta=bundle.category_meta, keep_price=False)
    features = _align_columns_for_model(bundle.model, features)
    preds = bundle.model.predict(features)
    return [float(p) for p in preds]
