#!/usr/bin/env python
"""
run_regressor.py
================
STATUS : NEW, Launch script for Experiment 15 (Controllability Evaluation)
PURPOSE: Train an XGBoost regressor on Morgan fingerprints for any
         SMILES+label CSV dataset. Used by exp15 to build independent
         property predictors for B3DB and Tartarus (1SYH / 4LDE / 6Y2F).

         Also reusable by exp14 and any future fingerprint-based regression.

Usage:
    python bmfm_sm_gen/launch/run_regressor.py \\
        --dataset PATH \\
        --smiles-col smiles \\
        --label-col  y_normalized_minmax \\
        --output-dir paper_results/exp15_controllability/xgboost/b3db

Outputs (in --output-dir):
    xgboost_model.json        XGBoost booster checkpoint
    scaler_params.json        raw label min/max for back-conversion
    metrics.json              held-out MAE, R², Spearman, Pearson
    test_predictions.csv      smiles, y_true, y_pred (test split)
    train_predictions.csv     smiles, y_true, y_pred (train split)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Morgan fingerprint generator (radius=2, 2048-bit), same as exp14
_FP_GEN = GetMorganGenerator(radius=2, fpSize=2048)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def smiles_to_fps(smiles_list: list[str]) -> tuple[np.ndarray, list[int]]:
    """Convert a list of SMILES to stacked Morgan FP matrix.

    Returns:
        fps        float32 array shape (n_valid, 2048)
        valid_idx  indices into smiles_list where conversion succeeded
    """
    fps, valid_idx = [], []
    for i, smi in enumerate(smiles_list):
        if not isinstance(smi, str):
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = _FP_GEN.GetFingerprintAsNumPy(mol).astype(np.float32)
        fps.append(fp)
        valid_idx.append(i)
    if not fps:
        raise ValueError("No valid SMILES produced fingerprints. Check --smiles-col.")
    return np.stack(fps, axis=0), valid_idx


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    sp   = float(spearmanr(y_true, y_pred).statistic)
    pe   = float(pearsonr(y_true, y_pred).statistic)
    return {"rmse": rmse, "mae": mae, "r2": r2, "spearman": sp, "pearson": pe}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train XGBoost regressor on Morgan fingerprints."
    )
    parser.add_argument("--dataset",      required=True,
                        help="Path to input CSV file.")
    parser.add_argument("--smiles-col",   default="smiles",
                        help="Name of the SMILES column (default: smiles).")
    parser.add_argument("--label-col",    required=True,
                        help="Name of the label column to predict.")
    parser.add_argument("--output-dir",   required=True,
                        help="Directory to save model and metrics.")
    parser.add_argument("--test-size",    type=float, default=0.2,
                        help="Fraction of data held out for testing (default: 0.2).")
    parser.add_argument("--random-state", type=int,   default=42,
                        help="Random seed for train/test split (default: 42).")
    parser.add_argument("--n-estimators", type=int,   default=500,
                        help="XGBoost n_estimators (default: 500).")
    parser.add_argument("--max-depth",    type=int,   default=6,
                        help="XGBoost max_depth (default: 6).")
    parser.add_argument("--learning-rate",type=float, default=0.05,
                        help="XGBoost learning_rate (default: 0.05).")
    args = parser.parse_args()

    # ---- Load data ----------------------------------------------------------
    log.info(f"Loading dataset from {args.dataset} ...")
    df = pd.read_csv(args.dataset)
    log.info(f"  Rows loaded: {len(df)}")

    if args.smiles_col not in df.columns:
        raise ValueError(
            f"SMILES column '{args.smiles_col}' not found. "
            f"Available: {df.columns.tolist()}"
        )
    if args.label_col not in df.columns:
        raise ValueError(
            f"Label column '{args.label_col}' not found. "
            f"Available: {df.columns.tolist()}"
        )

    df = df.dropna(subset=[args.smiles_col, args.label_col]).reset_index(drop=True)
    log.info(f"  Rows after dropping NaN: {len(df)}")

    raw_label_min = float(df[args.label_col].min())
    raw_label_max = float(df[args.label_col].max())
    log.info(
        f"  Label '{args.label_col}': min={raw_label_min:.4f}  "
        f"max={raw_label_max:.4f}  mean={df[args.label_col].mean():.4f}"
    )

    # ---- Fingerprints -------------------------------------------------------
    log.info("Computing Morgan fingerprints (radius=2, 2048-bit) ...")
    X, valid_idx = smiles_to_fps(df[args.smiles_col].tolist())
    y = df[args.label_col].iloc[valid_idx].values
    smiles_valid = df[args.smiles_col].iloc[valid_idx].values
    log.info(f"  Valid fingerprints: {len(X)}/{len(df)}")

    # ---- Split --------------------------------------------------------------
    X_tr, X_te, y_tr, y_te, smi_tr, smi_te = train_test_split(
        X, y, smiles_valid,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    log.info(f"  Train: {len(X_tr)}   Test: {len(X_te)}")

    # ---- Train XGBoost ------------------------------------------------------
    try:
        import xgboost as xgb
    except ImportError:
        log.error("xgboost not installed.  Run: pip install xgboost")
        sys.exit(1)

    log.info(
        f"Training XGBoost (n_estimators={args.n_estimators}, "
        f"max_depth={args.max_depth}, lr={args.learning_rate}) ..."
    )

    # Use GPU if available, fall back to CPU hist silently
    import subprocess
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, check=True)
        tree_method = "hist"
        device = "cuda"
    except Exception:
        tree_method = "hist"
        device = "cpu"

    model = xgb.XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method=tree_method,
        device=device,
        random_state=args.random_state,
        verbosity=0,
        early_stopping_rounds=30,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        verbose=False,
    )
    log.info(f"  Best iteration: {model.best_iteration}")

    # ---- Evaluate -----------------------------------------------------------
    y_pred_tr = model.predict(X_tr)
    y_pred_te = model.predict(X_te)

    train_metrics = compute_metrics(y_tr, y_pred_tr)
    test_metrics  = compute_metrics(y_te, y_pred_te)

    log.info("Train metrics: " + "  ".join(f"{k}={v:.4f}" for k, v in train_metrics.items()))
    log.info("Test  metrics: " + "  ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()))

    # ---- Save ---------------------------------------------------------------
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Model checkpoint
    model_path = out / "xgboost_model.json"
    model.save_model(str(model_path))
    log.info(f"Saved model → {model_path}")

    # Scaler params (for back-conversion in downstream scripts)
    scaler_params = {
        "label_col": args.label_col,
        "data_min":  raw_label_min,
        "data_max":  raw_label_max,
        "description": (
            f"raw_value = y_pred * (data_max - data_min) + data_min "
            f"[only valid if label_col is already y_normalized_minmax]"
        ),
    }
    (out / "scaler_params.json").write_text(json.dumps(scaler_params, indent=2))

    # Metrics
    full_metrics = {
        "train": train_metrics,
        "test":  test_metrics,
        "n_train": int(len(X_tr)),
        "n_test":  int(len(X_te)),
        "best_iteration": int(model.best_iteration),
        "label_col": args.label_col,
        "dataset": args.dataset,
    }
    metrics_path = out / "metrics.json"
    metrics_path.write_text(json.dumps(full_metrics, indent=2))
    log.info(f"Saved metrics → {metrics_path}")

    # Predictions CSVs
    pd.DataFrame({
        "smiles": smi_te, "y_true": y_te, "y_pred": y_pred_te
    }).to_csv(out / "test_predictions.csv", index=False)

    pd.DataFrame({
        "smiles": smi_tr, "y_true": y_tr, "y_pred": y_pred_tr
    }).to_csv(out / "train_predictions.csv", index=False)

    log.info(f"Saved test predictions  → {out / 'test_predictions.csv'}")
    log.info(f"Saved train predictions → {out / 'train_predictions.csv'}")
    log.info("=== run_regressor.py complete ===")


if __name__ == "__main__":
    main()
