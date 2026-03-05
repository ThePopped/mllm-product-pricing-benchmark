from __future__ import annotations

from pathlib import Path

import pandas as pd

from load_data import load_records
from build_features import build_features


def build_dataset(
    jsonl_path: str | Path,
    price_cap: float = 6000.0,
    category_meta: dict[str, list] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load raw JSONL data and return a ready-to-train (X, y) pair.

    Args:
        jsonl_path: Path to a JSONL file of ExtractionRecords
            (e.g. train.jsonl or holdout.jsonl).
        price_cap: Rows with price above this value are dropped as outliers.
        category_meta: Optional category metadata saved during training.
            Pass this when loading the hold-out set so categorical columns
            have exactly the same levels the model was trained on.

    Returns:
        X: Feature DataFrame (no `price` column).
        y: Price series.
    """
    records = load_records(jsonl_path)
    df = build_features(records, category_meta=category_meta, keep_price=True)

    df = df[df["price"] < price_cap].reset_index(drop=True)

    y = df.pop("price")
    X = df
    return X, y
