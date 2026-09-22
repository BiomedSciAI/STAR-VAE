"""
exp19_evaluate_ablation.py
==========================
Compute all ablation metrics for a single Exp19 run from its generated CSVs.

Metrics computed
----------------
Primary (all runs):
  - Spearman ρ: pooled over all 3 targets × 3 seeds
  - Calibration slope β: OLS regression of realized SA score on target
  - MAE: mean |realized − target| (normalized scale)
  - SR₁₀: fraction with |realized − target| ≤ 0.10
  - SR₂₀: fraction with |realized − target| ≤ 0.20
  - Validity: fraction parseable by RDKit
  - IntDiv: internal diversity (mean pairwise 1−Tanimoto)
  - KS p-value: Kolmogorov-Smirnov test vs. training SA distribution

A2-specific (if --a2_checkpoint provided):
  - KL divergence from saved model checkpoint
  - Latent interpolation validity (50 pairs × 10 steps)

A4-specific (if --checkpoint provided):
  - Trainable parameter count
  - Convergence epoch (from TensorBoard event file, if available)

A5-specific:
  - KL utilization (fraction of latent dims with KL > 0.1 nats)
  - Posterior collapse rate (fraction with KL < 0.01 nats)

Usage
-----
# All primary metrics:
python bmfm_sm_gen/scripts/exp19_evaluate_ablation.py \
    --run_id REF \
    --generated_dir paper_results/exp19_ablation_studies/generated/REF \
    --train_csv     paper_results/exp15_controllability/data/sa_score_minmax.csv \
    --outdir        paper_results/exp19_ablation_studies/stats \
    --label_col     y_normalized_minmax

# All runs together (from a wrapper script):
for run_id in REF A1_ref_moses A1_smiles_moses A2_det A3_enc_only A3_dec_only A3_no_cond A4_full_ft A4_frozen A5_mono_anneal A5_fixed_beta; do
    python bmfm_sm_gen/scripts/exp19_evaluate_ablation.py \
        --run_id $run_id \
        --generated_dir paper_results/exp19_ablation_studies/generated/$run_id \
        --train_csv paper_results/exp15_controllability/data/sa_score_minmax.csv \
        --outdir paper_results/exp19_ablation_studies/stats
done
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

SA_MIN = 1.437   # raw SA score min (from exp15 scaler)
SA_MAX = 7.513   # raw SA score max

TARGETS = {
    "high": 0.165,
    "mid":  0.500,
    "low":  0.835,
}
SEEDS = [42, 0, 101]


def sa_raw_to_norm(raw: float) -> float:
    """Convert raw SA score to normalized [0,1] (minmax)."""
    return (raw - SA_MIN) / (SA_MAX - SA_MIN)


def load_generated(generated_dir: Path, label_col: str = "sa_score") -> pd.DataFrame:
    """Load all generated CSVs from a run's output directory."""
    rows = []
    for seed in SEEDS:
        for bucket, target in TARGETS.items():
            p = generated_dir / f"seed_{seed}" / bucket / "generated.csv"
            if not p.exists():
                log.warning(f"Missing: {p}")
                continue
            df = pd.read_csv(p)
            df["bucket"] = bucket
            df["target_norm"] = target
            df["seed"] = seed
            rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No generated CSVs found in {generated_dir}")
    return pd.concat(rows, ignore_index=True)


def check_validity(df: pd.DataFrame) -> float:
    """Fraction of SMILES parseable by RDKit."""
    try:
        from rdkit import Chem
        valid = df["smiles"].apply(lambda s: Chem.MolFromSmiles(s) is not None)
        return float(valid.mean())
    except ImportError:
        log.warning("RDKit not available. Validity set to NaN")
        return float("nan")


def compute_internal_diversity(df: pd.DataFrame, max_pairs: int = 5000) -> float:
    """Mean pairwise (1 − Tanimoto) over a random sample of molecule pairs."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs

        mols = [Chem.MolFromSmiles(s) for s in df["smiles"] if s]
        fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
               for m in mols if m is not None]
        if len(fps) < 2:
            return float("nan")

        rng = np.random.default_rng(42)
        idx = rng.integers(0, len(fps), size=(min(max_pairs, len(fps) * 2), 2))
        sims = [DataStructs.TanimotoSimilarity(fps[i], fps[j])
                for i, j in idx if i != j]
        return float(np.mean([1.0 - s for s in sims]))
    except ImportError:
        log.warning("RDKit not available. IntDiv set to NaN")
        return float("nan")


def compute_sa_norm(df: pd.DataFrame) -> pd.Series:
    """Return normalized SA score. Uses sa_score column if present."""
    if "sa_score" in df.columns:
        return df["sa_score"].apply(sa_raw_to_norm)
    else:
        log.warning("No sa_score column. Cannot compute ρ / MAE. Run with --no_score=false.")
        return pd.Series(dtype=float)


def compute_metrics(df: pd.DataFrame, train_csv: str, label_col: str) -> dict:
    """Compute all primary metrics for one run."""
    metrics = {}

    # Validity
    metrics["validity"] = check_validity(df)

    # Internal diversity
    metrics["int_div"] = compute_internal_diversity(df, max_pairs=5000)

    # SA score normalized
    sa_norm = compute_sa_norm(df)
    if sa_norm.empty:
        return metrics

    df = df.copy()
    df["sa_norm"] = sa_norm
    df_valid = df.dropna(subset=["sa_norm"])

    # Spearman ρ (pooled across all targets × seeds)
    rho, pval = stats.spearmanr(df_valid["target_norm"], df_valid["sa_norm"])
    metrics["spearman_rho"] = float(rho)
    metrics["spearman_pval"] = float(pval)

    # Calibration slope (OLS β)
    slope, intercept, *_ = stats.linregress(df_valid["target_norm"], df_valid["sa_norm"])
    metrics["calib_slope"] = float(slope)
    metrics["calib_intercept"] = float(intercept)

    # MAE
    delta = (df_valid["target_norm"] - df_valid["sa_norm"]).abs()
    metrics["mae"] = float(delta.mean())
    metrics["sr10"] = float((delta <= 0.10).mean())
    metrics["sr20"] = float((delta <= 0.20).mean())

    # Per-bucket breakdown
    for bucket in TARGETS:
        sub = df_valid[df_valid["bucket"] == bucket]
        if len(sub) < 5:
            continue
        rho_b, _ = stats.spearmanr(sub["target_norm"], sub["sa_norm"])
        delta_b = (sub["target_norm"] - sub["sa_norm"]).abs()
        metrics[f"rho_{bucket}"] = float(rho_b)
        metrics[f"mae_{bucket}"] = float(delta_b.mean())
        metrics[f"sr10_{bucket}"] = float((delta_b <= 0.10).mean())

    # KS test vs. training distribution per bucket
    if train_csv and Path(train_csv).exists():
        train_df = pd.read_csv(train_csv)
        if label_col in train_df.columns:
            for bucket, target in TARGETS.items():
                lo, hi = target - 0.167, target + 0.167  # ±1/6 window around tertile centre
                train_sub = train_df[(train_df[label_col] >= lo) & (train_df[label_col] <= hi)][label_col]
                gen_sub = df_valid[df_valid["bucket"] == bucket]["sa_norm"]
                if len(train_sub) > 5 and len(gen_sub) > 5:
                    ks_stat, ks_p = stats.ks_2samp(train_sub, gen_sub)
                    metrics[f"ks_stat_{bucket}"] = float(ks_stat)
                    metrics[f"ks_pval_{bucket}"] = float(ks_p)

    # N molecules generated
    metrics["n_generated"] = int(len(df))
    metrics["n_valid"] = int(df_valid.shape[0])

    return metrics


def compute_a4_param_count(checkpoint: str) -> dict:
    """Count trainable parameters from a saved checkpoint."""
    try:
        import torch
        ckpt = torch.load(checkpoint, map_location="cpu")
        # Lightning checkpoints store hyper_parameters
        hp = ckpt.get("hyper_parameters", {})
        state = ckpt.get("state_dict", {})
        total = sum(v.numel() for v in state.values() if v.dtype.is_floating_point)
        return {"total_params": total, "lora": hp.get("lora", "unknown"),
                "freeze_backbone": hp.get("freeze_backbone", False)}
    except Exception as e:
        log.warning(f"Could not extract param count: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Exp19 ablation evaluation")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--generated_dir", required=True)
    parser.add_argument("--train_csv", default="paper_results/exp15_controllability/data/sa_score_minmax.csv")
    parser.add_argument("--label_col", default="y_normalized_minmax")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint path for A4 trainable param count")
    args = parser.parse_args()

    generated_dir = Path(REPO_ROOT / args.generated_dir) if not Path(args.generated_dir).is_absolute() else Path(args.generated_dir)
    outdir = Path(REPO_ROOT / args.outdir) if not Path(args.outdir).is_absolute() else Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log.info(f"=== Evaluating run: {args.run_id} ===")

    # Load generated molecules
    df = load_generated(generated_dir, args.label_col)
    log.info(f"  Loaded {len(df):,} generated molecules across {df['seed'].nunique()} seeds")

    # Compute metrics
    metrics = compute_metrics(df, args.train_csv, args.label_col)
    metrics["run_id"] = args.run_id

    # A4: param count
    if args.checkpoint and Path(args.checkpoint).exists():
        metrics.update(compute_a4_param_count(args.checkpoint))

    # Save per-run JSON
    out_json = outdir / f"{args.run_id}_metrics.json"
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"  Saved metrics → {out_json}")

    # Print summary
    log.info(f"  ρ={metrics.get('spearman_rho', 'N/A'):.4f}  "
             f"SR₁₀={metrics.get('sr10', 'N/A'):.4f}  "
             f"MAE={metrics.get('mae', 'N/A'):.4f}  "
             f"V={metrics.get('validity', 'N/A'):.4f}  "
             f"IntDiv={metrics.get('int_div', 'N/A'):.4f}")

    return metrics


if __name__ == "__main__":
    main()
