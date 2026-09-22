#!/usr/bin/env python
"""
exp16_generate_tartarus.py
==========================
STATUS : NEW, Experiment 16 (Tartarus Docking Controllability with QVina Oracle)
PURPOSE: Generate molecules from the three Tartarus CVAEs conditioned in the CORRECT
         label space (raw docking scores in [−12, 0]).

DESIGN:
  - Three targets: 1syh, 4lde, 6y2f
  - Three buckets per target, defined in raw docking-score space:
        high  (strong binders): y_raw ∈ [−12, −8]   →  label uniformly sampled from this range
        mid   (moderate):       y_raw ∈  [−8, −4]
        low   (weak binders):   y_raw ∈  [−4,  0]
  - For each bucket: 1000 molecules, each conditioned on an INDEPENDENTLY sampled y* drawn
    uniformly from the bucket's raw range.  This provides richer coverage of the conditioning
    space than a single fixed y* and removes any single-point artefact.
  - 3 seeds × 3 buckets × 3 targets = 27 generation jobs.

WHY RAW LABELS:
  The checkpoints in <see .env>
  were trained with label.range = [−12, 0] and NO min-max normalization.
  Labels were passed directly as raw clamped docking scores (see hparams.yaml).
  Exp15 erroneously passed minmax values {0.165, 0.500, 0.835} which all map to ~0
  in the model's learned space (worst binding).  This experiment fixes that.

Usage:
    python bmfm_sm_gen/scripts/exp16_generate_tartarus.py \\
        --target 6y2f --seed 0

    python bmfm_sm_gen/scripts/exp16_generate_tartarus.py \\
        --target all --seed 42

Outputs per (target, seed):
    paper_results/exp16_tartarus_oracle/generated/{target}/seed_{seed}/high.csv
    paper_results/exp16_tartarus_oracle/generated/{target}/seed_{seed}/mid.csv
    paper_results/exp16_tartarus_oracle/generated/{target}/seed_{seed}/low.csv

    Each CSV: smiles, conditioning_y_star, bucket, target, seed

When --save-latents is set, also saves:
    paper_results/exp16_tartarus_oracle/latents/{target}/seed_{seed}/{bucket}_latents.npy
    Shape: (n_valid_molecules, latent_dim): the VAE encoder mu vectors before sampling.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

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
OUT_ROOT      = REPO_ROOT / "paper_results/exp16_tartarus_oracle/generated"
LATENT_ROOT   = REPO_ROOT / "paper_results/exp16_tartarus_oracle/latents"

TASK_CKPT = {
    "1syh": (
        str(Path(os.environ.get("OUTPUT_DIR", "outputs")) /
            "vae_80m_conditional_1syh_score_film_lora_prior_film/checkpoints/"
            "vae-epochepoch=12-step=10081-val_loss=0.22.ckpt")
    ),
    "4lde": (
        str(Path(os.environ.get("OUTPUT_DIR", "outputs")) /
            "vae_80m_conditional_4lde_score_film_lora_prior_film/checkpoints/"
            "vae-epochepoch=12-step=10696-val_loss=0.23.ckpt")
    ),
    "6y2f": (
        str(Path(os.environ.get("OUTPUT_DIR", "outputs")) /
            "vae_80m_conditional_6y2f_score_film_lora_prior_film/checkpoints/"
            "vae-epochepoch=19-step=15842-val_loss=0.22.ckpt")
    ),
}

# Bucket definitions in raw docking-score space (training label space is [-12, 0])
BUCKETS = {
    "high": (-12.0, -8.0),   # strong binders (low = more negative = better)
    "mid":  (-8.0,  -4.0),   # moderate binders
    "low":  (-4.0,   0.0),   # weak binders
}

# Bucket definitions in percentile-rank label space [0, 1].
# Used when --label-space=percentile (stratified CVAE fine-tunes trained on
# y_normalized_percentile).  Boundaries match the stratified training CSVs:
#   high bucket ≈ [0.000, 0.333]  (strongest binders, lowest raw scores)
#   mid  bucket ≈ [0.333, 0.667]
#   low  bucket ≈ [0.667, 1.000]  (weakest binders)
BUCKETS_PERCENTILE = {
    "high": (0.000, 0.333),
    "mid":  (0.333, 0.667),
    "low":  (0.667, 1.000),
}

NUM_SAMPLES  = 1_000   # per bucket
TEMPERATURE  = 1.0
TOP_K        = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, device: torch.device):
    """Load CVAELightningModule from checkpoint."""
    sys.path.insert(0, str(REPO_ROOT))
    import copy, torch as _torch
    from bmfm_sm_gen.trainers.lm_cvae import CVAELightningModule

    log.info(f"Loading CVAE from {ckpt_path}")

    # The stratified fine-tune checkpoints were saved when lm_vae.py still stored
    # the post-increment value (max_len=72) in hparams instead of the user value
    # (71).  On replay, __init__ adds +1 again → 73 → pos_encoder.pe=[1,73,512]
    # which mismatches the state_dict's [1,72,512].  Fix: peek at the stored
    # hparams and, if max_len==72, override it back to 71 before construction.
    raw = _torch.load(ckpt_path, map_location="cpu")
    override_kwargs = {}
    stored_max_len = raw.get("hyper_parameters", {}).get("model_params", {}).get("max_len")
    if stored_max_len == 72:
        mp = copy.deepcopy(raw["hyper_parameters"]["model_params"])
        mp["max_len"] = 71
        override_kwargs["model_params"] = mp
        override_kwargs["pretrained_path"] = None  # suppress inner _load_pretrained

    lm = CVAELightningModule.load_from_checkpoint(
        ckpt_path,
        map_location=device,
        strict=False,
        **override_kwargs,
    )
    lm.to(device)
    lm.eval()
    return lm


def generate_bucket(lm, bucket_lo: float, bucket_hi: float,
                    num_samples: int, seed: int,
                    save_latents: bool = False,
                    ) -> tuple[list[str], list[float], np.ndarray | None]:
    """
    Generate num_samples molecules.  Each molecule is conditioned on an INDEPENDENTLY
    and UNIFORMLY sampled y* drawn from [bucket_lo, bucket_hi].

    When save_latents=True, also captures the encoder mu vectors (the latent mean
    before sampling) for every valid molecule.

    Returns (smiles_list, y_star_list, latents_or_None)
      - latents_or_None: np.ndarray of shape (n_valid, latent_dim) or None
    """
    rng = np.random.default_rng(seed)
    y_stars = rng.uniform(bucket_lo, bucket_hi, size=num_samples).tolist()

    sys.path.insert(0, str(REPO_ROOT))
    from bmfm_sm_gen.trainers.lm_cvae import generate_molecules_with_labels

    all_smiles:  list[str]   = []
    all_ystars:  list[float] = []
    all_latents: list        = []   # list of 1-D numpy arrays

    device = next(lm.parameters()).device
    chunk  = 100   # generate 100 at a time (each with its own y*)

    for start in range(0, num_samples, chunk):
        end      = min(start + chunk, num_samples)
        batch_ys = y_stars[start:end]
        batch_n  = len(batch_ys)

        labels = torch.tensor(batch_ys, dtype=torch.float32).unsqueeze(1).to(device)
        z      = torch.randn(batch_n, lm.model.latent_dim, device=device)

        with torch.no_grad():
            if save_latents:
                # Capture the FiLM-conditioned prior mu (p(z|y)).
                # get_conditional_prior(labels) returns (mu, logvar) for the
                # label-conditioned Gaussian prior. This is what the model uses
                # to steer generation and is the correct latent to analyse.
                try:
                    mu, _ = lm.model.get_conditional_prior(labels)
                except (AttributeError, AssertionError):
                    # Fallback: use the sampled z (isotropic prior, no label info)
                    mu = z
                latent_np = mu.cpu().numpy()   # (batch_n, latent_dim)

            smiles = generate_molecules_with_labels(
                lm.model, lm.tokenizer, z, labels,
                temperature=TEMPERATURE, top_k=TOP_K, lora=lm.lora,
            )

        # Filter empty/None; keep latents aligned with valid smiles
        for i, (smi, y) in enumerate(zip(smiles, batch_ys)):
            if smi and isinstance(smi, str) and smi.strip():
                all_smiles.append(smi.strip())
                all_ystars.append(y)
                if save_latents:
                    all_latents.append(latent_np[i])

        log.info(f"  {end}/{num_samples} generated ({len(all_smiles)} valid so far)")

    latents_out = np.vstack(all_latents) if (save_latents and all_latents) else None
    return all_smiles, all_ystars, latents_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True,
                   choices=list(TASK_CKPT.keys()) + ["all"],
                   help="Tartarus target or 'all'.")
    p.add_argument("--seed", type=int, required=True,
                   help="Random seed (use 0, 42, 101).")
    p.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    p.add_argument("--save-latents", action="store_true",
                   help="Save encoder mu vectors alongside generated molecules "
                        "for latent space analysis (Step 4.1).")
    p.add_argument("--checkpoint-override", default=None,
                   help="Override the default checkpoint path for this target. "
                        "Used for the stratified fine-tuned model (Tier B).")
    p.add_argument("--output-root", default=None,
                   help="Override the default output root directory. "
                        "Defaults to paper_results/exp16_tartarus_oracle/generated.")
    p.add_argument("--label-space", default="raw", choices=["raw", "percentile"],
                   help="Label space used by the model. Use 'percentile' for "
                        "stratified fine-tuned CVAEs trained on y_normalized_percentile. "
                        "Defaults to 'raw' (kcal/mol, [-12,0]) for baseline models.")
    return p.parse_args()


def run_one_target(target: str, seed: int, num_samples: int,
                   save_latents: bool = False,
                   checkpoint_override: str | None = None,
                   output_root: Path | None = None,
                   buckets_override: dict | None = None):
    log.info(f"\n{'='*60}\n Target: {target}  Seed: {seed}\n{'='*60}")

    ckpt   = checkpoint_override if checkpoint_override else TASK_CKPT[target]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    log.info(f"Checkpoint: {ckpt}")

    lm = load_model(ckpt, device)
    pl.seed_everything(seed)

    out_root = output_root if output_root else OUT_ROOT
    out_dir  = out_root / target / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if save_latents:
        lat_dir = LATENT_ROOT / target / f"seed_{seed}"
        lat_dir.mkdir(parents=True, exist_ok=True)

    active_buckets = buckets_override if buckets_override is not None else BUCKETS
    for bucket_name, (lo, hi) in active_buckets.items():
        log.info(f"\n--- Bucket: {bucket_name}  y* ∈ [{lo}, {hi}] ---")
        smiles, ystars, latents = generate_bucket(
            lm, lo, hi, num_samples, seed, save_latents=save_latents
        )

        df = pd.DataFrame({
            "smiles":              smiles,
            "conditioning_y_star": ystars,
            "bucket":              bucket_name,
            "target":              target,
            "seed":                seed,
        })
        out_csv = out_dir / f"{bucket_name}.csv"
        df.to_csv(out_csv, index=False)
        log.info(f"  Saved {len(df)} molecules → {out_csv}")

        if save_latents and latents is not None:
            lat_path = lat_dir / f"{bucket_name}_latents.npy"
            np.save(lat_path, latents)
            log.info(f"  Saved latents {latents.shape} → {lat_path}")


def main():
    args    = parse_args()
    targets = list(TASK_CKPT.keys()) if args.target == "all" else [args.target]
    out_root = Path(args.output_root) if args.output_root else None

    # Select bucket label ranges based on the model's training label space
    if args.label_space == "percentile":
        log.info("Using PERCENTILE bucket boundaries [0,0.333] / [0.333,0.667] / [0.667,1.0]")
        active_buckets = BUCKETS_PERCENTILE
    else:
        active_buckets = BUCKETS

    for t in targets:
        run_one_target(
            t, args.seed, args.num_samples,
            save_latents=args.save_latents,
            checkpoint_override=args.checkpoint_override,
            output_root=out_root,
            buckets_override=active_buckets,
        )
    log.info("\n=== exp16_generate_tartarus.py complete ===")


if __name__ == "__main__":
    main()
