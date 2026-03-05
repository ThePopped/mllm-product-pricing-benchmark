from __future__ import annotations

import pandas as pd

from src.build_features import build_features, extract_category_meta


def _record(price: float, brand: str, colour: str) -> dict:
    return {
        "price": price,
        "discount_amount": None,
        "currency": "GBP",
        "brand": brand,
        "model": "Sofa Model",
        "colour_1": colour,
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


def test_build_features_drops_expected_columns() -> None:
    df = build_features([_record(999.0, "A", "grey")], keep_price=True)
    for col in ("evidence", "additional_features", "model", "discount_amount"):
        assert col not in df.columns


def test_build_features_keep_price_flag() -> None:
    with_price = build_features([_record(500.0, "A", "blue")], keep_price=True)
    without_price = build_features([_record(500.0, "A", "blue")], keep_price=False)
    assert "price" in with_price.columns
    assert "price" not in without_price.columns


def test_build_features_single_dict_equals_list_of_one() -> None:
    rec = _record(750.0, "B", "green")
    a = build_features(rec, keep_price=True)
    b = build_features([rec], keep_price=True)
    pd.testing.assert_frame_equal(a, b)


def test_build_features_applies_category_meta() -> None:
    data = [_record(1000.0, "BrandA", "grey"), _record(1100.0, "BrandB", "beige")]
    category_meta = {"brand": ["BrandA", "BrandB", "BrandC"]}
    df = build_features(data, category_meta=category_meta, keep_price=True)
    assert str(df["brand"].dtype) == "category"
    assert list(df["brand"].cat.categories) == ["BrandA", "BrandB", "BrandC"]


def test_extract_category_meta_roundtrip() -> None:
    train_df = build_features(
        [_record(1200.0, "A", "charcoal"), _record(1300.0, "B", "charcoal")],
        keep_price=True,
    )
    meta = extract_category_meta(train_df)
    transformed = build_features(
        [_record(1250.0, "A", "charcoal")],
        category_meta=meta,
        keep_price=True,
    )
    assert "brand" in meta
    assert list(transformed["brand"].cat.categories) == list(meta["brand"])
