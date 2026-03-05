from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from scipy.stats import loguniform, randint
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_dataset import build_dataset
from build_features import extract_category_meta
from lineage_utils import compute_data_split_id, write_lineage

DEFAULT_DATA = ROOT / "data" / "processed" / "train.jsonl"
DEFAULT_HOLDOUT = ROOT / "data" / "processed" / "holdout.jsonl"
DEFAULT_MODEL_OUT = ROOT / "models" / "best_model.pkl"
DEFAULT_META_OUT = ROOT / "models" / "category_meta.json"
DEFAULT_LINEAGE_OUT = ROOT / "models" / "lineage.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train sofa price regression model.")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA,
                   help="Path to train.jsonl (default: data/processed/train.jsonl).")
    p.add_argument("--holdout-data", type=Path, default=DEFAULT_HOLDOUT,
                   help="Path to holdout.jsonl used to compute data_split_id lineage tag.")
    p.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_OUT)
    p.add_argument("--meta-out", type=Path, default=DEFAULT_META_OUT)
    p.add_argument("--lineage-out", type=Path, default=DEFAULT_LINEAGE_OUT)
    p.add_argument("--model-version", type=str, default=None,
                   help="Optional explicit model version label. Defaults to model-<train_run_id_prefix>.")
    p.add_argument("--n-iter", type=int, default=60,
                   help="Number of RandomizedSearchCV iterations.")
    p.add_argument("--price-cap", type=float, default=6000.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.model_out.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("Loading training data...")
    X, y = build_dataset(args.data, price_cap=args.price_cap)
    category_meta = extract_category_meta(X)
    print(f"Training set: {len(X)} records, {X.shape[1]} features")

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # ------------------------------------------------------------------
    # Baseline (default hyperparams, CV score only — not logged to MLflow)
    # ------------------------------------------------------------------
    print("Fitting baseline...")
    baseline_scores = cross_val_score(
        HistGradientBoostingRegressor(random_state=42),
        X, y, cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    baseline_rmse = float(-baseline_scores.mean())
    print(f"Baseline CV RMSE: {baseline_rmse:.2f}")

    # ------------------------------------------------------------------
    # Hyperparameter search + MLflow run
    # ------------------------------------------------------------------
    param_distributions = {
        "learning_rate": loguniform(1e-3, 2e-1),
        "max_iter": randint(200, 1500),
        "max_depth": [None, 3, 5, 8, 12],
        "max_leaf_nodes": randint(15, 255),
        "min_samples_leaf": randint(10, 200),
        "l2_regularization": loguniform(1e-6, 1e2),
    }

    search = RandomizedSearchCV(
        estimator=HistGradientBoostingRegressor(random_state=42),
        param_distributions=param_distributions,
        n_iter=args.n_iter,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        random_state=42,
        verbose=2,
    )

    mlflow.set_experiment("sofa_price_regression")
    data_split_id = compute_data_split_id(args.data, args.holdout_data)

    with mlflow.start_run(run_name="train_search"):
        train_run_id = mlflow.active_run().info.run_id
        model_version = args.model_version or f"model-{train_run_id[:8]}"
        mlflow.set_tag("stage", "training")
        mlflow.set_tag("train_run_id", train_run_id)
        mlflow.set_tag("data_split_id", data_split_id)
        mlflow.set_tag("model_version", model_version)

        print("Running hyperparameter search...")
        search.fit(X, y)
        best = search.best_estimator_
        cv_rmse = float(-search.best_score_)

        print(f"Baseline CV RMSE : {baseline_rmse:.2f}")
        print(f"Best CV RMSE     : {cv_rmse:.2f}")
        print(f"Best params      : {search.best_params_}")

        mlflow.log_param("n_iter", args.n_iter)
        mlflow.log_param("price_cap", args.price_cap)
        mlflow.log_param("train_size", len(X))
        mlflow.log_param("train_data_path", str(args.data))
        mlflow.log_param("holdout_data_path", str(args.holdout_data))
        mlflow.log_param("train_run_id", train_run_id)
        mlflow.log_param("data_split_id", data_split_id)
        mlflow.log_param("model_version", model_version)
        mlflow.log_params(best.get_params())
        mlflow.log_metric("baseline_cv_rmse", baseline_rmse)
        mlflow.log_metric("cv_rmse", cv_rmse)
        mlflow.sklearn.log_model(best, "model")

    # ------------------------------------------------------------------
    # Persist model and category metadata
    # ------------------------------------------------------------------
    joblib.dump(best, args.model_out)
    print(f"Model saved          → {args.model_out}")

    with open(args.meta_out, "w", encoding="utf-8") as f:
        json.dump(category_meta, f, indent=2)
    print(f"Category metadata    → {args.meta_out}")
    lineage = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "train_run_id": train_run_id,
        "data_split_id": data_split_id,
        "model_version": model_version,
        "train_data_path": str(args.data),
        "holdout_data_path": str(args.holdout_data),
        "model_path": str(args.model_out),
        "meta_path": str(args.meta_out),
    }
    write_lineage(args.lineage_out, lineage)
    print(f"Lineage metadata     → {args.lineage_out}")
    print("\nRun evaluate.py on the hold-out set to get final test metrics.")


if __name__ == "__main__":
    main()
