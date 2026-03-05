from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import sys
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_features import build_features
from inference_service import ModelBundle, load_bundle
from load_data import load_records
from project_config import DEFAULT_CONFIG, cfg_path, load_config

FRONTEND_INDEX = ROOT / "frontend" / "index.html"


class PredictRequest(BaseModel):
    numeric_field: str = Field(description="Numeric feature name to override.")
    numeric_value: float = Field(description="Numeric value to score with (must be > 0).")
    categorical_field: str = Field(description="Categorical feature name to override.")
    categorical_value: Any = Field(description="Categorical value observed in training data.")

    @model_validator(mode="after")
    def validate_positive_numeric(self) -> "PredictRequest":
        if self.numeric_value <= 0:
            raise ValueError("numeric_value must be greater than 0.")
        return self


class PredictResponse(BaseModel):
    predicted_price: float
    model_version: str | None = None
    train_run_id: str | None = None
    data_split_id: str | None = None


def _coerce_categorical_value(value: Any, allowed_values: list[Any]) -> Any:
    if value in allowed_values:
        return value

    if isinstance(value, str):
        lowered = value.strip().lower()

        # Common web-client behavior: booleans are submitted as strings.
        if any(isinstance(v, bool) for v in allowed_values):
            if lowered in ("true", "1", "yes"):
                candidate = True
                if candidate in allowed_values:
                    return candidate
            if lowered in ("false", "0", "no"):
                candidate = False
                if candidate in allowed_values:
                    return candidate

        # Optional numeric coercion for stringified numerics.
        if any(isinstance(v, (int, float)) and not isinstance(v, bool) for v in allowed_values):
            try:
                if "." in lowered:
                    candidate_num: Any = float(lowered)
                else:
                    candidate_num = int(lowered)
                if candidate_num in allowed_values:
                    return candidate_num
            except ValueError:
                pass

    return value


def _align_columns_for_model(model: Any, features: pd.DataFrame) -> pd.DataFrame:
    names = list(getattr(model, "feature_names_in_", []))
    if not names:
        return features
    missing = [name for name in names if name not in features.columns]
    if missing:
        raise ValueError(f"Missing required model input columns: {missing}")
    return features.loc[:, names]


def _is_numeric_series(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _build_baseline_features(bundle: ModelBundle, train_path: Path) -> pd.DataFrame:
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found for serving template: {train_path}")
    records = load_records(train_path)
    if not records:
        raise ValueError(f"No parsed_json records found in: {train_path}")

    features = build_features(records, category_meta=bundle.category_meta, keep_price=False)
    features = _align_columns_for_model(bundle.model, features)
    defaults: dict[str, Any] = {}

    for col in features.columns:
        s = features[col].dropna()
        if _is_numeric_series(features[col]):
            vals = pd.to_numeric(s, errors="coerce").dropna()
            if vals.empty:
                defaults[col] = 1.0
            else:
                positive_vals = vals[vals > 0]
                defaults[col] = float((positive_vals if not positive_vals.empty else vals).median())
        else:
            mode = s.mode(dropna=True)
            if not mode.empty:
                defaults[col] = mode.iloc[0]
            elif pd.api.types.is_categorical_dtype(features[col]) and len(features[col].cat.categories) > 0:
                defaults[col] = features[col].cat.categories[0]
            else:
                defaults[col] = None

    baseline = pd.DataFrame([defaults], columns=list(features.columns))
    for col in features.select_dtypes(include="category").columns:
        baseline[col] = pd.Categorical(baseline[col], categories=features[col].cat.categories)
    return baseline


def _build_top_categories(records: list[dict], all_categories: dict[str, list], top_k: int) -> dict[str, list]:
    if top_k <= 0:
        return {col: [] for col in all_categories}

    train_df = pd.json_normalize(records) if records else pd.DataFrame()
    top_categories: dict[str, list] = {}

    for col, allowed in all_categories.items():
        ranked: list[Any] = []
        if col in train_df.columns:
            observed_ranked = train_df[col].dropna().value_counts().index.tolist()
            ranked = [v for v in observed_ranked if v in allowed]

        top = ranked[:top_k]
        if len(top) < min(top_k, len(allowed)):
            # Backfill from allowed values to ensure deterministic options.
            top.extend([v for v in allowed if v not in top][: top_k - len(top)])
        top_categories[col] = top

    return top_categories


def _build_schema_payload(
    bundle: ModelBundle,
    baseline: pd.DataFrame,
    records: list[dict],
    top_k_categories: int,
) -> dict[str, Any]:
    numeric_fields = {
        col: {"min_exclusive": 0.0, "default": float(baseline.iloc[0][col])}
        for col in baseline.columns
        if _is_numeric_series(baseline[col])
    }
    categorical_fields_all = {
        col: values
        for col, values in bundle.category_meta.items()
        if col in baseline.columns and values
    }
    categorical_fields_top = _build_top_categories(
        records=records,
        all_categories=categorical_fields_all,
        top_k=top_k_categories,
    )
    return {
        "numeric_fields": numeric_fields,
        "categorical_fields_all": categorical_fields_all,
        "categorical_fields_top": categorical_fields_top,
        "top_k_categories": top_k_categories,
        "model_version": bundle.lineage.get("model_version"),
        "train_run_id": bundle.lineage.get("train_run_id"),
        "data_split_id": bundle.lineage.get("data_split_id"),
    }


def create_app(config_path: Path = DEFAULT_CONFIG) -> FastAPI:
    cfg = load_config(config_path)
    paths_cfg = cfg.get("paths", {})
    serving_cfg = cfg.get("serving", {})
    top_k_categories = int(serving_cfg.get("top_k_categories", 10))

    model_path = cfg_path(ROOT, paths_cfg.get("model_out"), ROOT / "models" / "best_model.pkl")
    meta_path = cfg_path(ROOT, paths_cfg.get("meta_out"), ROOT / "models" / "category_meta.json")
    lineage_path = cfg_path(ROOT, paths_cfg.get("lineage_out"), ROOT / "models" / "lineage.json")
    train_path = cfg_path(ROOT, paths_cfg.get("train_jsonl"), ROOT / "data" / "processed" / "train.jsonl")

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            records = load_records(train_path)
            if not records:
                raise ValueError(f"No parsed_json records found in: {train_path}")
            bundle = load_bundle(model_path=model_path, meta_path=meta_path, lineage_path=lineage_path)
            baseline = _build_baseline_features(bundle, train_path=train_path)
            _app.state.bundle = bundle
            _app.state.baseline = baseline
            _app.state.schema = _build_schema_payload(bundle, baseline, records, top_k_categories)
            _app.state.ready = True
            _app.state.ready_error = None
        except Exception as exc:  # pragma: no cover
            _app.state.bundle = None
            _app.state.baseline = None
            _app.state.schema = None
            _app.state.ready = False
            _app.state.ready_error = str(exc)
        yield

    app = FastAPI(
        title=str(serving_cfg.get("title", "Sofa Price Serving API")),
        version="1.0.0",
        lifespan=lifespan,
    )

    cors_origins = serving_cfg.get("cors_origins", [])
    if isinstance(cors_origins, list) and cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[str(origin) for origin in cors_origins],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.state.ready = False
    app.state.ready_error = None
    app.state.bundle = None
    app.state.baseline = None
    app.state.schema = None

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        if not FRONTEND_INDEX.exists():
            raise HTTPException(status_code=404, detail=f"Frontend not found: {FRONTEND_INDEX}")
        return FileResponse(FRONTEND_INDEX)

    @app.get("/readyz")
    def readyz() -> dict[str, Any]:
        if app.state.ready:
            return {"ready": True}
        return {"ready": False, "error": app.state.ready_error}

    @app.get("/v1/schema")
    def schema() -> dict[str, Any]:
        if not app.state.ready:
            raise HTTPException(status_code=503, detail=f"Model not ready: {app.state.ready_error}")
        return app.state.schema

    @app.post("/v1/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        if not app.state.ready:
            raise HTTPException(status_code=503, detail=f"Model not ready: {app.state.ready_error}")

        schema_payload = app.state.schema
        numeric_fields = schema_payload["numeric_fields"]
        categorical_fields = schema_payload["categorical_fields_all"]

        if req.numeric_field not in numeric_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown numeric_field '{req.numeric_field}'. Use /v1/schema.",
            )
        if req.categorical_field not in categorical_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown categorical_field '{req.categorical_field}'. Use /v1/schema.",
            )

        allowed_values = categorical_fields[req.categorical_field]
        coerced_categorical_value = _coerce_categorical_value(req.categorical_value, allowed_values)
        if coerced_categorical_value not in allowed_values:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid categorical_value '{req.categorical_value}' for field "
                    f"'{req.categorical_field}'. Use /v1/schema."
                ),
            )

        row = app.state.baseline.copy(deep=True)
        row.at[row.index[0], req.numeric_field] = float(req.numeric_value)
        row.at[row.index[0], req.categorical_field] = coerced_categorical_value
        preds = app.state.bundle.model.predict(row)

        return PredictResponse(
            predicted_price=float(preds[0]),
            model_version=app.state.bundle.lineage.get("model_version"),
            train_run_id=app.state.bundle.lineage.get("train_run_id"),
            data_split_id=app.state.bundle.lineage.get("data_split_id"),
        )

    return app


app = create_app()
