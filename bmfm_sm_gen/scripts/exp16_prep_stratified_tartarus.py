#!/usr/bin/env python
"""
exp16_prep_stratified_tartarus.py
==================================
STATUS : NEW, Tier B, Step 5.1 (Stratified Dataset Construction)
PURPOSE: Build balanced training datasets for the stratified fine-tuning experiment.
         Fixes two root causes of score attenuation:
           F1: Label sparsity  → equal representation per bucket (1K or 2K per bucket)
           F2: Normalisation squeeze → percentile-rank normalisation replaces minmax

         Unique canonical molecules are split before sampling. Training buckets may
         be oversampled with score jitter; validation buckets are never augmented.

         Per-target design:
           6Y2F  score range [-9.4, -1.7]  - 3K target (1K/bucket)
           1SYH  score range [-10, +255.6] - filter score < 0, 6K target (2K/bucket)
           4LDE  score range [-10, +18.6]  - filter score < 0, 6K target (2K/bucket)

Outputs:
    paper_results/exp16_tartarus_oracle/data/tartarus_{target}_stratified_{N}k.csv
    Columns: smiles, selfies, {target}_score, y_normalized_percentile
    (selfies column required by ds_mol.py which loads modality="selfies")
    Also writes a summary JSON: tartarus_stratified_summary.json

Usage:
    python bmfm_sm_gen/scripts/exp16_prep_stratified_tartarus.py
    python bmfm_sm_gen/scripts/exp16_prep_stratified_tartarus.py --target 6y2f
    python bmfm_sm_gen/scripts/exp16_prep_stratified_tartarus.py --target all
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import percentileofscore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_IN   = REPO_ROOT / "paper_results/exp15_controllability/data"
DATA_OUT  = REPO_ROOT / "paper_results/exp16_tartarus_oracle/data"

# Bucket boundaries (raw kcal/mol, shared across all targets)
BOUNDS = {
    "high": (-12.0, -8.0),
    "mid":  (-8.0,  -4.0),
    "low":  (-4.0,   0.0),
}

SCORE_COL = {"1syh": "1syh_score", "4lde": "4lde_score", "6y2f": "6y2f_score"}

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Per-bucket stratified sampler
# ---------------------------------------------------------------------------

def stratified_subsample_mid(
    df: pd.DataFrame,
    score_col: str,
    lo: float,
    hi: float,
    n_total: int,
    n_bins: int = 8,
) -> pd.DataFrame:
    """
    Sample n_total rows from df where score is in [lo, hi], spread across n_bins
    equal-width bins so the dense peak does not dominate.
    """
    mask = (df[score_col] >= lo) & (df[score_col] < hi)
    sub  = df[mask].copy()
    if len(sub) == 0:
        return sub

    edges      = np.linspace(lo, hi, n_bins + 1)
    per_bin    = max(1, n_total // n_bins)
    frames     = []

    for b_lo, b_hi in zip(edges[:-1], edges[1:]):
        bin_mask = (sub[score_col] >= b_lo) & (sub[score_col] < b_hi)
        bin_df   = sub[bin_mask]
        if len(bin_df) == 0:
            continue
        k = min(per_bin, len(bin_df))
        frames.append(bin_df.sample(n=k, random_state=int(RNG.integers(1e6)), replace=False))

    sampled = pd.concat(frames, ignore_index=False) if frames else sub.iloc[0:0]
    n_missing = min(n_total, len(sub)) - len(sampled)
    if n_missing > 0:
        remaining = sub.drop(index=sampled.index)
        sampled = pd.concat([
            sampled,
            remaining.sample(
                n=n_missing,
                random_state=int(RNG.integers(1e6)),
                replace=False,
            ),
        ])
    return sampled.sample(frac=1, random_state=int(RNG.integers(1e6))).reset_index(drop=True)


def oversample_with_jitter(
    df: pd.DataFrame,
    score_col: str,
    n_target: int,
    jitter_sigma: float = 0.05,
) -> pd.DataFrame:
    """
    Oversample df with replacement to reach n_target rows.
    Add Gaussian jitter (sigma=jitter_sigma kcal/mol) to the score of each
    oversampled copy so that duplicate molecules receive distinct conditioning labels.
    The original rows are kept as-is (no jitter) and the oversampled copies receive jitter.
    """
    n_real    = len(df)
    n_extra   = n_target - n_real
    if n_real == 0:
        return df.copy()
    if n_extra <= 0:
        return df.sample(n=n_target, random_state=int(RNG.integers(1e6)), replace=False)

    extras = df.sample(n=n_extra, random_state=int(RNG.integers(1e6)), replace=True).copy()
    noise  = RNG.normal(0, jitter_sigma, size=n_extra)
    extras[score_col] = (extras[score_col] + noise).clip(
        df[score_col].min() - 0.5, df[score_col].max() + 0.5
    )
    return pd.concat([df, extras], ignore_index=True)


def compute_percentile_rank(scores: np.ndarray) -> np.ndarray:
    """
    Compute the percentile rank of each score within the array, returning values in [0, 1].
    Uses scipy percentileofscore for correctness with ties.
    """
    return np.array([
        percentileofscore(scores, s, kind="rank") / 100.0
        for s in scores
    ])


def canonicalize_and_deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return one row per valid canonical molecule and input-cleaning counts."""
    required = {"smiles", "selfies"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    canonical = []
    for smiles in df["smiles"]:
        mol = Chem.MolFromSmiles(str(smiles))
        canonical.append(Chem.MolToSmiles(mol, canonical=True) if mol is not None else None)

    out = df.copy()
    out["_canonical_smiles"] = canonical
    n_invalid = int(out["_canonical_smiles"].isna().sum())
    out = out.dropna(subset=["_canonical_smiles"])
    n_duplicates = int(out.duplicated("_canonical_smiles").sum())
    out = out.drop_duplicates("_canonical_smiles", keep="first").reset_index(drop=True)
    return out, {
        "n_input_rows": int(len(df)),
        "n_invalid_smiles_dropped": n_invalid,
        "n_duplicate_canonical_dropped": n_duplicates,
        "n_unique_canonical": int(len(out)),
    }


def split_unique_by_bucket(
    df: pd.DataFrame,
    score_col: str,
    val_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split unique molecules within score buckets before any augmentation."""
    train_parts = []
    val_parts = []
    for bucket_index, (_name, (lo, hi)) in enumerate(BOUNDS.items()):
        bucket = df[(df[score_col] >= lo) & (df[score_col] < hi)]
        n_val = int(round(len(bucket) * val_fraction))
        if len(bucket) > 1:
            n_val = min(max(n_val, 1), len(bucket) - 1)
        else:
            n_val = 0
        val = bucket.sample(n=n_val, random_state=seed + bucket_index, replace=False)
        train_parts.append(bucket.drop(index=val.index))
        val_parts.append(val)
    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(val_parts, ignore_index=True),
    )


# ---------------------------------------------------------------------------
# Balanced train/validation construction
# ---------------------------------------------------------------------------

TARGET_PER_BUCKET = {"6y2f": 1000, "1syh": 2000, "4lde": 2000}
JITTER_SIGMA = {"6y2f": 0.05, "1syh": 0.05, "4lde": 0.1}


def _sample_bucket(df: pd.DataFrame, score_col: str, name: str, n: int) -> pd.DataFrame:
    lo, hi = BOUNDS[name]
    bucket = df[(df[score_col] >= lo) & (df[score_col] < hi)]
    if name == "mid":
        return stratified_subsample_mid(bucket, score_col, lo, hi, n, n_bins=8)
    return bucket.sample(
        n=min(n, len(bucket)),
        random_state=int(RNG.integers(1e6)),
        replace=False,
    )


def build_target_splits(
    df_raw: pd.DataFrame,
    target: str,
    val_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build disjoint balanced splits, augmenting only the training split."""
    global RNG
    RNG = np.random.default_rng(seed)
    score_col = SCORE_COL[target]
    if score_col not in df_raw:
        raise ValueError(f"Input is missing score column: {score_col}")

    filtered = df_raw[df_raw[score_col] < 0].copy()
    unique, cleaning = canonicalize_and_deduplicate(filtered)
    train_pool, val_pool = split_unique_by_bucket(unique, score_col, val_fraction, seed)
    per_bucket = TARGET_PER_BUCKET[target]
    train_parts = []
    val_parts = []
    bucket_counts = {}

    for name in BOUNDS:
        nominal_train_goal = int(round(per_bucket * (1 - val_fraction)))
        val_goal = per_bucket - nominal_train_goal
        val = _sample_bucket(val_pool, score_col, name, val_goal)
        train_goal = per_bucket - len(val)
        train_real = _sample_bucket(train_pool, score_col, name, train_goal)
        train = oversample_with_jitter(
            train_real,
            score_col,
            train_goal,
            jitter_sigma=JITTER_SIGMA[target],
        )
        train_parts.append(train)
        val_parts.append(val)
        bucket_counts[name] = {
            "train_pool_unique": int(len(train_pool[
                (train_pool[score_col] >= BOUNDS[name][0])
                & (train_pool[score_col] < BOUNDS[name][1])
            ])),
            "validation_pool_unique": int(len(val_pool[
                (val_pool[score_col] >= BOUNDS[name][0])
                & (val_pool[score_col] < BOUNDS[name][1])
            ])),
            "train_target_rows": int(train_goal),
            "validation_target_rows": int(val_goal),
            "train_unique_before_oversampling": int(len(train_real)),
            "train_oversampled_rows": int(len(train) - len(train_real)),
            "train_rows": int(len(train)),
            "validation_augmented_rows": 0,
            "validation_rows": int(len(val)),
        }
        log.info(
            "  %s: train=%d rows (%d unique before oversampling), validation=%d rows",
            name,
            len(train),
            len(train_real),
            len(val),
        )

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    combined = pd.concat([train_df, val_df], ignore_index=True)
    combined["y_normalized_percentile"] = compute_percentile_rank(combined[score_col].to_numpy())
    train_df = combined.iloc[:len(train_df)].copy()
    val_df = combined.iloc[len(train_df):].copy()

    train_canonical = set(train_df["_canonical_smiles"])
    val_canonical = set(val_df["_canonical_smiles"])
    overlap = train_canonical & val_canonical
    assert len(val_canonical) == len(val_df), "Validation contains duplicate canonical molecules"
    assert not overlap, f"Canonical train/validation overlap detected: {len(overlap)}"

    columns = ["smiles", "selfies", score_col, "y_normalized_percentile"]
    details = {
        **cleaning,
        "n_filtered_nonnegative": int(len(df_raw) - len(filtered)),
        "target_rows": int(per_bucket * len(BOUNDS)),
        "n_train_rows": int(len(train_df)),
        "n_train_unique_canonical": int(len(train_canonical)),
        "n_validation_rows": int(len(val_df)),
        "n_validation_unique_canonical": int(len(val_canonical)),
        "n_combined_rows": int(len(combined)),
        "canonical_overlap_count": int(len(overlap)),
        "bucket_counts": bucket_counts,
    }
    return train_df[columns], val_df[columns], details


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TARGET_SIZES = {"6y2f": 3, "1syh": 6, "4lde": 6}   # in thousands


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="all",
                   choices=list(SCORE_COL.keys()) + ["all"],
                   help="Target to process (default: all).")
    p.add_argument("--input-dir", type=Path, default=DATA_IN,
                   help="Directory containing tartarus_<target>_minmax.csv files.")
    p.add_argument("--output-dir", type=Path, default=DATA_OUT,
                   help="Directory for stratified train/validation CSV files.")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args    = parse_args()
    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1")
    targets = list(SCORE_COL.keys()) if args.target == "all" else [args.target]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}

    for target in targets:
        log.info(f"\n{'='*60}\nBuilding stratified dataset for: {target}\n{'='*60}")
        sc  = SCORE_COL[target]
        input_path = args.input_dir / f"tartarus_{target}_minmax.csv"
        if not input_path.is_file():
            raise FileNotFoundError(f"Tartarus input not found: {input_path}")
        raw = pd.read_csv(input_path)

        train_df, val_df, details = build_target_splits(
            raw, target, args.val_fraction, args.seed
        )
        df_out = pd.concat([train_df, val_df], ignore_index=True)

        n_k   = TARGET_SIZES[target]
        fname = f"tartarus_{target}_stratified_{n_k}k.csv"
        out_p = args.output_dir / fname
        df_out.to_csv(out_p, index=False)
        log.info(f"Saved → {out_p}  ({len(df_out):,} rows)")
        train_df.to_csv(args.output_dir / f"tartarus_{target}_stratified_{n_k}k_train.csv", index=False)
        val_df.to_csv(args.output_dir / f"tartarus_{target}_stratified_{n_k}k_val.csv", index=False)
        log.info(
            "Split seed=%d: train=%d, validation=%d, canonical overlap=%d",
            args.seed,
            len(train_df),
            len(val_df),
            details["canonical_overlap_count"],
        )

        summary[target] = {
            "output_file": fname,
            "train_file": f"tartarus_{target}_stratified_{n_k}k_train.csv",
            "validation_file": f"tartarus_{target}_stratified_{n_k}k_val.csv",
            **details,
            "y_norm_min": float(df_out["y_normalized_percentile"].min()),
            "y_norm_max": float(df_out["y_normalized_percentile"].max()),
            "y_norm_mean": float(df_out["y_normalized_percentile"].mean()),
        }

    (args.output_dir / "tartarus_stratified_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    log.info("\n=== exp16_prep_stratified_tartarus.py complete ===")


if __name__ == "__main__":
    main()
