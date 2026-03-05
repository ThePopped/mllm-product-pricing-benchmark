from __future__ import annotations

"""One-time script to split the raw extraction JSONL into train and hold-out sets.

Run this once before training:
    python src/split_data.py

Produces:
    data/processed/train.jsonl
    data/processed/holdout.jsonl

The split is stratified on price quantile so both files have a similar price
distribution. The hold-out set must not be touched until final evaluation.
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = ROOT / "data" / "processed" / "MLLM_extracted_features.jsonl"
DEFAULT_TRAIN = ROOT / "data" / "processed" / "train.jsonl"
DEFAULT_HOLDOUT = ROOT / "data" / "processed" / "holdout.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split extraction JSONL into train / hold-out.")
    p.add_argument("--src", type=Path, default=DEFAULT_SRC)
    p.add_argument("--train-out", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--holdout-out", type=Path, default=DEFAULT_HOLDOUT)
    p.add_argument("--holdout-size", type=float, default=0.15,
                   help="Fraction of records reserved for hold-out (default: 0.15).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.train_out.exists() or args.holdout_out.exists():
        raise FileExistsError(
            f"Split files already exist ({args.train_out}, {args.holdout_out}). "
            "Delete them manually if you want to re-split."
        )

    # Load only lines with a valid price for stratification; keep all lines in
    # the final split regardless so no records are silently discarded.
    lines: list[str] = []
    prices: list[float] = []
    with args.src.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            parsed = entry.get("parsed_json") or {}
            price = parsed.get("price")
            lines.append(line)
            prices.append(float(price) if price is not None else float("nan"))

    n = len(lines)
    rng = np.random.default_rng(args.seed)

    # Stratify by price quantile (5 bins) for records that have a valid price;
    # records without a price are assigned to a separate stratum.
    prices_arr = np.array(prices)
    valid_mask = np.isfinite(prices_arr)
    strata = np.full(n, -1, dtype=int)  # -1 = no price
    if valid_mask.any():
        quantiles = np.nanpercentile(prices_arr[valid_mask], [20, 40, 60, 80])
        strata[valid_mask] = np.digitize(prices_arr[valid_mask], quantiles)

    holdout_indices: set[int] = set()
    for stratum in np.unique(strata):
        stratum_idx = np.where(strata == stratum)[0]
        n_holdout = max(1, round(len(stratum_idx) * args.holdout_size))
        chosen = rng.choice(stratum_idx, size=n_holdout, replace=False)
        holdout_indices.update(chosen.tolist())

    train_lines = [l for i, l in enumerate(lines) if i not in holdout_indices]
    holdout_lines = [l for i, l in enumerate(lines) if i in holdout_indices]

    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    args.train_out.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    args.holdout_out.write_text("\n".join(holdout_lines) + "\n", encoding="utf-8")

    print(f"Total records : {n}")
    print(f"Train         : {len(train_lines)}  → {args.train_out}")
    print(f"Hold-out      : {len(holdout_lines)}  → {args.holdout_out}")


if __name__ == "__main__":
    main()
