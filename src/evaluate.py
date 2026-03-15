from __future__ import annotations

"""Evaluate the trained model against the held-out test set.

Run after train.py:
    python src/evaluate.py

Loads the saved model and category metadata, applies them to the hold-out
set, and reports final metrics. Plots are saved to models/.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from scipy import stats
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_dataset import build_dataset
from lineage_utils import compute_data_split_id, read_lineage
from project_config import DEFAULT_CONFIG, cfg_path, load_config

DEFAULT_HOLDOUT = ROOT / "data" / "processed" / "holdout.jsonl"
DEFAULT_TRAIN = ROOT / "data" / "processed" / "train.jsonl"
DEFAULT_MODEL = ROOT / "models" / "best_model.pkl"
DEFAULT_META = ROOT / "models" / "category_meta.json"
DEFAULT_LINEAGE = ROOT / "models" / "lineage.json"
PLOTS_DIR = ROOT / "models"


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    pre_args, _ = pre.parse_known_args()
    cfg = load_config(pre_args.config)
    paths_cfg = cfg.get("paths", {})
    eval_cfg = cfg.get("evaluate", {})

    p = argparse.ArgumentParser(description="Evaluate trained model on the hold-out set.")
    p.add_argument("--config", type=Path, default=pre_args.config)
    p.add_argument("--holdout", type=Path, default=cfg_path(ROOT, paths_cfg.get("holdout_jsonl"), DEFAULT_HOLDOUT))
    p.add_argument("--train-data", type=Path, default=cfg_path(ROOT, paths_cfg.get("train_jsonl"), DEFAULT_TRAIN),
                   help="Path to train split used to compute/verify data_split_id lineage tag.")
    p.add_argument("--model", type=Path, default=cfg_path(ROOT, paths_cfg.get("model_out"), DEFAULT_MODEL))
    p.add_argument("--meta", type=Path, default=cfg_path(ROOT, paths_cfg.get("meta_out"), DEFAULT_META))
    p.add_argument("--lineage-file", type=Path, default=cfg_path(ROOT, paths_cfg.get("lineage_out"), DEFAULT_LINEAGE))
    p.add_argument("--plots-dir", type=Path, default=cfg_path(ROOT, paths_cfg.get("eval_plots_dir"), PLOTS_DIR))
    p.add_argument("--train-run-id", type=str, default=None)
    p.add_argument("--data-split-id", type=str, default=None)
    p.add_argument("--model-version", type=str, default=None)
    p.add_argument("--price-cap", type=float, default=float(eval_cfg.get("price_cap", 6000.0)))
    p.add_argument("--experiment", type=str, default=str(eval_cfg.get("experiment", "sofa_price_regression")))
    return p.parse_args()


def save_plot(fig: plt.Figure, name: str) -> Path:
    path = PLOTS_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def resolve_lineage(args: argparse.Namespace) -> dict[str, Any]:
    lineage = {}
    if args.lineage_file.exists():
        lineage = read_lineage(args.lineage_file)

    train_run_id = args.train_run_id or lineage.get("train_run_id")
    model_version = args.model_version or lineage.get("model_version")
    data_split_id = args.data_split_id or lineage.get("data_split_id")
    if data_split_id is None:
        data_split_id = compute_data_split_id(args.train_data, args.holdout)

    return {
        "train_run_id": train_run_id,
        "model_version": model_version,
        "data_split_id": data_split_id,
    }


def main() -> None:
    global PLOTS_DIR
    args = parse_args()
    PLOTS_DIR = args.plots_dir
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    lineage = resolve_lineage(args)

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name="holdout_evaluation"):
        mlflow.set_tag("stage", "evaluation")
        if lineage["train_run_id"] is not None:
            mlflow.set_tag("train_run_id", lineage["train_run_id"])
            mlflow.set_tag("mlflow.parentRunId", lineage["train_run_id"])
            mlflow.log_param("train_run_id", lineage["train_run_id"])
        if lineage["data_split_id"] is not None:
            mlflow.set_tag("data_split_id", lineage["data_split_id"])
            mlflow.log_param("data_split_id", lineage["data_split_id"])
        if lineage["model_version"] is not None:
            mlflow.set_tag("model_version", lineage["model_version"])
            mlflow.log_param("model_version", lineage["model_version"])

        # ------------------------------------------------------------------
        # Load model + category metadata
        # ------------------------------------------------------------------
        model = joblib.load(args.model)
        with open(args.meta, "r", encoding="utf-8") as f:
            category_meta = json.load(f)
        print(f"Loaded model         ← {args.model}")

        # ------------------------------------------------------------------
        # Load hold-out set with the training category levels applied
        # ------------------------------------------------------------------
        print("Loading hold-out data...")
        X, y = build_dataset(args.holdout, price_cap=args.price_cap, category_meta=category_meta)
        print(f"Hold-out set: {len(X)} records")

        # ------------------------------------------------------------------
        # Metrics
        # ------------------------------------------------------------------
        y_pred = model.predict(X)
        rmse = mean_squared_error(y, y_pred) ** 0.5
        r2 = model.score(X, y)

        print(f"\n{'='*40}")
        print(f"Hold-out RMSE : {rmse:.2f}")
        print(f"Hold-out R²   : {r2:.4f}")
        print(f"{'='*40}\n")

        mlflow.log_param("price_cap", args.price_cap)
        mlflow.log_param("holdout_size", len(X))
        mlflow.log_param("model_path", str(args.model))
        mlflow.log_param("meta_path", str(args.meta))
        mlflow.log_metric("holdout_rmse", float(rmse))
        mlflow.log_metric("holdout_r2", float(r2))

        # ------------------------------------------------------------------
        # Plots
        # ------------------------------------------------------------------
        # Predicted vs Actual
        fig, ax = plt.subplots()
        ax.scatter(y, y_pred, alpha=0.6)
        mn, mx = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], color="red", linewidth=1)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Predicted vs Actual (hold-out)")
        pred_actual_path = save_plot(fig, "holdout_predicted_vs_actual.png")

        # Residuals vs Predicted
        residuals = y - y_pred
        fig, ax = plt.subplots()
        ax.scatter(y_pred, residuals, alpha=0.6)
        ax.axhline(0, color="red", linewidth=1)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual (Actual − Predicted)")
        ax.set_title("Residuals vs Predicted (hold-out)")
        residuals_path = save_plot(fig, "holdout_residuals.png")

        # Residual distribution
        fig, ax = plt.subplots()
        ax.hist(residuals, bins=30, alpha=0.75, edgecolor="black")
        ax.axvline(0, color="red", linewidth=1, linestyle="--")
        ax.set_xlabel("Residual (Actual − Predicted)")
        ax.set_ylabel("Count")
        ax.set_title("Residual Distribution (hold-out)")
        residual_dist_path = save_plot(fig, "holdout_residual_distribution.png")

        # Residual Q-Q plot (normal reference)
        fig, ax = plt.subplots()
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title("Residual Q-Q Plot (hold-out)")
        qq_path = save_plot(fig, "holdout_residuals_qq.png")

        # Permutation feature importance
        print("Computing permutation feature importance...")
        perm = permutation_importance(
            model, X, y,
            n_repeats=10, random_state=1001,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        top_k = 20
        idx = np.argsort(perm.importances_mean)[::-1][:top_k]
        labels = X.columns[idx]
        fig, ax = plt.subplots()
        ax.bar(range(len(idx)), perm.importances_mean[idx],
               yerr=perm.importances_std[idx], capsize=3)
        ax.set_xticks(range(len(idx)))
        ax.set_xticklabels(labels, rotation=60, ha="right")
        ax.set_ylabel("Importance (Δ score when permuted)")
        ax.set_title(f"Permutation Feature Importance — top {top_k} (hold-out)")
        fig.tight_layout()
        importance_path = save_plot(fig, "holdout_feature_importance.png")

        mlflow.log_artifact(str(pred_actual_path), artifact_path="evaluation_plots")
        mlflow.log_artifact(str(residuals_path), artifact_path="evaluation_plots")
        mlflow.log_artifact(str(residual_dist_path), artifact_path="evaluation_plots")
        mlflow.log_artifact(str(qq_path), artifact_path="evaluation_plots")
        mlflow.log_artifact(str(importance_path), artifact_path="evaluation_plots")

        print(f"Plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
