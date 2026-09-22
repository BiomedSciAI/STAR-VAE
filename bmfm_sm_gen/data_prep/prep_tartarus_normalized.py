#!/usr/bin/env python
"""
prep_tartarus_normalized.py
===========================
Prepare [0,1]-normalized Tartarus docking CSVs for normalized CVAE training.

Normalization scheme (0 = best binder):
  - Clip rows with score >= 0  (remove positive-score outliers)
  - data_min = score.min()  (most negative = best binder)
  - data_max = 0.0          (anchored at zero)
  - y_normalized_minmax = (score - data_min) / (data_max - data_min)
    → 0 = best binder, 1 = weakest binder near zero

Outputs per target:
  paper_results/exp17_improving_conditioning/data/tartarus_{target}_normalized.csv
  paper_results/exp17_improving_conditioning/data/tartarus_{target}_normalized_scaler.json

Usage:
    python bmfm_sm_gen/data_prep/prep_tartarus_normalized.py
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Tartarus dataset: https://github.com/aspuru-guzik-group/tartarus
INPUT_CSV = os.environ.get("TARTARUS_DATA", "data/tartarus/docking_processed.csv")
OUTPUT_DIR = os.path.join(os.environ.get("OUTPUT_DIR", "outputs"), "tartarus_normalized")

TARGETS = ["6y2f", "1syh", "4lde"]
N_BUCKETS = 10


def normalize_target(df: pd.DataFrame, target: str, out_dir: Path) -> None:
    score_col = f"{target}_score"
    log.info(f"\n{'='*60}")
    log.info(f"Target: {target}  |  score column: {score_col}")

    if score_col not in df.columns:
        raise ValueError(f"Column '{score_col}' not found. Available: {df.columns.tolist()}")

    raw = df[score_col].dropna()
    log.info(f"  Raw rows (non-NaN)  : {len(raw):,}")
    log.info(f"  Raw score range     : [{raw.min():.4f}, {raw.max():.4f}]")
    log.info(f"  Raw positive count  : {(raw >= 0).sum():,} "
             f"({100.0*(raw >= 0).mean():.1f}%)")

    # 1. Keep only negative scores (clip positive outliers)
    mask = df[score_col] < 0
    sub = df[mask].copy().reset_index(drop=True)
    log.info(f"  After clip (score<0): {len(sub):,} rows "
             f"(dropped {len(df) - len(sub):,})")

    # 2. Minmax: data_min = actual min, data_max = 0.0
    data_min = float(sub[score_col].min())
    data_max = 0.0
    span = data_max - data_min  # always positive since data_min < 0

    sub["y_normalized_minmax"] = (sub[score_col] - data_min) / span

    y = sub["y_normalized_minmax"]
    log.info(f"  y_normalized range  : [{y.min():.4f}, {y.max():.4f}]  "
             f"mean={y.mean():.4f}  std={y.std():.4f}")

    # 3. Bucket distribution
    bins = np.linspace(0.0, 1.0, N_BUCKETS + 1)
    counts, _ = np.histogram(y, bins=bins)
    log.info(f"  Bucket distribution ({N_BUCKETS} equal-width buckets [0,1]):")
    for i, c in enumerate(counts):
        lo, hi = bins[i], bins[i + 1]
        log.info(f"    [{lo:.2f}, {hi:.2f}): {c:>7,}  ({100*c/len(sub):.1f}%)")

    # 4. Save CSV
    out_csv = out_dir / f"tartarus_{target}_normalized.csv"
    sub.to_csv(out_csv, index=False)
    log.info(f"  Saved → {out_csv}  ({len(sub):,} rows)")

    # 5. Save scaler params
    scaler_params = {
        "target": target,
        "score_column": score_col,
        "method": "minmax_clipped",
        "clip_threshold": 0.0,
        "data_min": data_min,
        "data_max": data_max,
        "n_rows_after_clip": int(len(sub)),
        "y_min": float(y.min()),
        "y_max": float(y.max()),
        "description": (
            "y_normalized_minmax = (score - data_min) / (data_max - data_min). "
            "data_min = min(negative scores), data_max = 0.0. "
            "0 = best binder (most negative kcal/mol), 1 = weakest binder."
        ),
    }
    scaler_out = out_dir / f"tartarus_{target}_normalized_scaler.json"
    scaler_out.write_text(json.dumps(scaler_params, indent=2))
    log.info(f"  Saved scaler → {scaler_out}")


def main() -> None:
    log.info("=== prep_tartarus_normalized.py ===")
    log.info(f"Input  : {INPUT_CSV}")
    log.info(f"Output : {OUTPUT_DIR}")

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    log.info(f"Loaded {len(df):,} rows.  Columns: {df.columns.tolist()}")

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    for target in TARGETS:
        normalize_target(df, target, out_dir)

    log.info("\n=== prep_tartarus_normalized.py complete ===")


if __name__ == "__main__":
    main()
