from __future__ import annotations

import pandas as pd

# Columns present in parsed_json that carry no predictive signal for price.
_COLS_TO_DROP = ["evidence", "additional_features", "model", "discount_amount"]


def build_features(
    data: dict | list[dict],
    category_meta: dict[str, list] | None = None,
    keep_price: bool = False,
) -> pd.DataFrame:
    """Transform one or more parsed-JSON records into a feature DataFrame.

    Accepts a single dict (inference) or a list of dicts (training) —
    pd.json_normalize handles both transparently, ensuring identical
    transformations in both paths.

    Args:
        data: A single sofa feature dict or a list of them.
        category_meta: Optional mapping of column name → list of known
            categories, as saved by `extract_category_meta` during training.
            When provided, categorical columns are cast with exactly those
            categories so they match what the model expects at inference time.
            When omitted, categories are inferred from the data (training only).
        keep_price: If True, the `price` column is retained (for training).
            If False (default), it is dropped so the DataFrame is ready for
            `model.predict()` (for inference).

    Returns:
        A DataFrame with feature columns, optionally including `price`.
    """
    df = pd.json_normalize(data)
    extra_drops = [] if keep_price else ["price"]
    df = _drop_cols(df, extra=extra_drops)
    df = _cast_categoricals(df, category_meta)
    return df


def extract_category_meta(df: pd.DataFrame) -> dict[str, list]:
    """Capture the category levels from a fitted training DataFrame.

    Call this after `build_features(..., keep_price=True)` and before saving
    the model. Pass the result to `build_features` at inference time so
    categorical columns have the exact same levels the model was trained on.

    Returns:
        Mapping of column name → sorted list of category values.
    """
    return {
        col: sorted(df[col].cat.categories.tolist())
        for col in df.columns
        if hasattr(df[col], "cat")
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _drop_cols(df: pd.DataFrame, extra: list[str]) -> pd.DataFrame:
    to_drop = [c for c in _COLS_TO_DROP + extra if c in df.columns]
    return df.drop(columns=to_drop)


def _cast_categoricals(
    df: pd.DataFrame,
    category_meta: dict[str, list] | None,
) -> pd.DataFrame:
    non_numeric = df.select_dtypes(exclude="number").columns.tolist()
    if category_meta is not None:
        for col in non_numeric:
            known = category_meta.get(col)
            if known is not None:
                df[col] = pd.Categorical(df[col], categories=known)
            else:
                df[col] = df[col].astype("category")
    else:
        df[non_numeric] = df[non_numeric].astype("category")
    return df
