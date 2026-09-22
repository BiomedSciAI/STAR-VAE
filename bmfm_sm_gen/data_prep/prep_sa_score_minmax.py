#!/usr/bin/env python
"""
prep_sa_score_minmax.py
=======================
Add a y_normalized_minmax column by mapping the observed minimum and maximum
SA scores in the input CSV to 0 and 1, respectively.

Also saves a scaler_params.json alongside the output CSV so that
all downstream scripts can back-convert predicted values to raw SA
score units without re-loading the dataset.

Input:  $PUBCHEM_10K_SA (default: data/pubchem/data_pubchem_random_10k_selfies.csv)
Output: $DATA_DIR/pubchem/sa_score_minmax.csv
        $DATA_DIR/pubchem/sa_score_scaler.json

Obtain the input from the source described in the paper's Data section.
"""

import argparse
import json
import logging
import os
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

INPUT_CSV = os.environ.get(
    "PUBCHEM_10K_SA",
    "data/pubchem/data_pubchem_random_10k_selfies.csv"
)
OUTPUT_DIR = os.path.join(os.environ.get("DATA_DIR", "data"), "pubchem")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize an SA-score CSV and create deterministic train/validation splits."
    )
    parser.add_argument("--input", default=INPUT_CSV, help="Input CSV containing sascore.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1")
    log.info("=== SA-score minmax normalization ===")
    log.info(f"Input : {args.input}")

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"SA-score dataset not found: {args.input}")

    df = pd.read_csv(args.input)
    log.info(f"Loaded {len(df)} rows.  Columns: {df.columns.tolist()}")

    if "sascore" not in df.columns:
        raise ValueError(
            f"Expected column 'sascore' not found.  Available: {df.columns.tolist()}"
        )

    # Drop rows with missing scores
    before = len(df)
    df = df.dropna(subset=["sascore"]).reset_index(drop=True)
    log.info(f"Dropped {before - len(df)} rows with missing sascore.")

    # Raw stats
    raw_min = float(df["sascore"].min())
    raw_max = float(df["sascore"].max())
    log.info(f"Raw sascore: min={raw_min:.4f}  max={raw_max:.4f}  "
             f"mean={df['sascore'].mean():.4f}  std={df['sascore'].std():.4f}")

    # Fit MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    df["y_normalized_minmax"] = scaler.fit_transform(df[["sascore"]]).flatten()

    log.info(
        f"y_normalized_minmax: min={df['y_normalized_minmax'].min():.4f}  "
        f"max={df['y_normalized_minmax'].max():.4f}"
    )

    # Save CSV
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_out = out_dir / "sa_score_minmax.csv"
    df.to_csv(csv_out, index=False)
    log.info(f"Saved CSV → {csv_out}  ({len(df)} rows)")

    val_df = df.sample(frac=args.val_fraction, random_state=args.seed)
    train_df = df.drop(index=val_df.index)
    train_out = out_dir / "sa_score_minmax_train.csv"
    val_out = out_dir / "sa_score_minmax_val.csv"
    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)
    log.info(
        "Saved deterministic split (seed=%d): train=%d, validation=%d",
        args.seed,
        len(train_df),
        len(val_df),
    )

    # Save scaler params so downstream scripts can back-convert without
    # reloading the dataset.
    scaler_params = {
        "label": "sascore",
        "method": "minmax",
        "data_min": raw_min,
        "data_max": raw_max,
        "feature_range": [0, 1],
        "description": (
            "y_normalized_minmax = (sascore - data_min) / (data_max - data_min). "
            "Invert: sascore = y_normalized_minmax * (data_max - data_min) + data_min."
        ),
    }
    scaler_out = out_dir / "sa_score_scaler.json"
    scaler_out.write_text(json.dumps(scaler_params, indent=2))
    log.info(f"Saved scaler params → {scaler_out}")

    log.info("=== prep_sa_score_minmax.py complete ===")


if __name__ == "__main__":
    main()
