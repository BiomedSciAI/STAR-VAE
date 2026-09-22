"""
exp19_generate_ablation.py
==========================
Generate molecules for a single Exp19 ablation run.

Loads the best (or last) checkpoint from a run's checkpoint directory,
then performs a conditioning sweep over 3 target tertiles (low/mid/high
SA score) × 3 random seeds × 1000 molecules per sweep, writing one CSV
per (tertile, seed) combination.

Usage
-----
# Single run:
python bmfm_sm_gen/scripts/exp19_generate_ablation.py \
    --run_id REF \
    --checkpoint paper_results/exp19_ablation_studies/checkpoints/REF/checkpoints/last.ckpt \
    --config   paper_results/exp19_ablation_studies/configs/model_vae_ablation_ref.yaml \
    --outdir   paper_results/exp19_ablation_studies/generated/REF

# Smoke test:
python bmfm_sm_gen/scripts/exp19_generate_ablation.py \
    --run_id REF ... --n_samples 10 --seeds 42

Outputs
-------
paper_results/exp19_ablation_studies/generated/<run_id>/
  seed_42/low/generated.csv     (SA target y*=0.835, hard to synthesize)
  seed_42/mid/generated.csv     (SA target y*=0.500)
  seed_42/high/generated.csv    (SA target y*=0.165, easy to synthesize)
  seed_{0,101}/...              (same for other seeds)
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Conditioning targets: tertile centres of the normalized SA score distribution
TARGETS = {
    "high": 0.165,   # easy to synthesize (low SA score)
    "mid":  0.500,
    "low":  0.835,   # hard to synthesize (high SA score)
}
SEEDS = [42, 0, 101]


def load_model(checkpoint: str, config: str, device: torch.device):
    """Load CVAELightningModule from checkpoint using its config.

    Handles two known checkpoint compatibility issues:

    1. Double-increment bug: the old CVAELightningModule saved model_params["max_len"]
       as the post-increment value (e.g. 72 when the config specified 71).
       load_from_checkpoint replays __init__ which adds +1 again, building max_len=73
       and a PE of shape [1,73,d] that mismatches the stored PE [1,72,d].
       We detect and correct this before loading.

    2. Missing tokenizer: checkpoints saved with Lightning ≥2.6 on a different machine
       may store tokenizer=None in hyper_parameters because the newer trainer saved it
       differently. This env's CVAELightningModule.__init__ requires tokenizer as a
       positional arg. We read the tokenizer class path from the YAML config and inject
       it as a keyword override into load_from_checkpoint.
    """
    import importlib
    import yaml as _yaml
    from bmfm_sm_gen.trainers.lm_cvae import CVAELightningModule

    log.info(f"Loading checkpoint: {checkpoint}")

    # --- detect double-increment bug in existing checkpoints ---
    ckpt_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hparams = ckpt_data.get("hyper_parameters", {})
    mp = hparams.get("model_params", {})

    # --- fix missing tokenizer: inject from YAML config if absent in hparams ---
    extra_kwargs = {}
    if hparams.get("tokenizer") is None and config and Path(config).exists():
        with open(config) as _f:
            cfg = _yaml.safe_load(_f)
        tokenizer_path = (
            cfg.get("data", {}).get("init_args", {}).get("tokenizer")
            or cfg.get("model", {}).get("init_args", {}).get("tokenizer")
        )
        vocab_file = (
            cfg.get("data", {}).get("init_args", {}).get("vocab_file")
            or cfg.get("model", {}).get("init_args", {}).get("vocab_file")
        )
        if tokenizer_path:
            module_path, cls_name = tokenizer_path.rsplit(".", 1)
            tokenizer_cls = getattr(importlib.import_module(module_path), cls_name)
            extra_kwargs["tokenizer"] = tokenizer_cls
            log.warning(
                f"[load_model] tokenizer missing in checkpoint hparams, "
                f"injecting from config: {tokenizer_path}"
            )
        stored_vocab = hparams.get("vocab_file")
        if vocab_file and (stored_vocab is None or not Path(stored_vocab).exists()):
            extra_kwargs["vocab_file"] = vocab_file
            log.warning(
                f"[load_model] vocab_file in hparams missing or not found ({stored_vocab!r}), "
                f"injecting from config: {vocab_file}"
            )

    # --- fix stale pretrained_path: null it out if the path doesn't exist on this machine ---
    # The weights are already merged into the state_dict; _load_pretrained is only called
    # during __init__ and is a no-op when pretrained_path is None.
    stored_pretrained = hparams.get("pretrained_path")
    if stored_pretrained and not Path(stored_pretrained).exists():
        log.warning(
            f"[load_model] pretrained_path in hparams not found on this machine "
            f"({stored_pretrained!r}). Setting to None (weights already in state_dict)."
        )
        ckpt_data["hyper_parameters"]["pretrained_path"] = None
        extra_kwargs["pretrained_path"] = None

    # --- fix stale val_csv path (saved from a different machine) ---
    stored_val_csv = hparams.get("val_csv")
    if stored_val_csv and not Path(stored_val_csv).exists():
        # Resolve relative to repo root
        val_csv_rel = Path(stored_val_csv).name  # filename only as fallback
        # Try to find it relative to REPO_ROOT
        candidate = REPO_ROOT / "paper_results/exp15_controllability/data" / val_csv_rel
        if candidate.exists():
            log.warning(
                f"[load_model] val_csv in hparams not found ({stored_val_csv!r}), "
                f"remapping to {candidate}"
            )
            ckpt_data["hyper_parameters"]["val_csv"] = str(candidate)
            extra_kwargs["val_csv"] = str(candidate)
        else:
            log.warning(
                f"[load_model] val_csv in hparams not found ({stored_val_csv!r}), "
                f"setting to None (not needed at generation time)."
            )
            ckpt_data["hyper_parameters"]["val_csv"] = None
            extra_kwargs["val_csv"] = None

    if mp:
        max_len_stored = mp.get("max_len")
        sd = ckpt_data.get("state_dict", {})
        pe_key = "model.encoder.pos_encoder.pe"
        if pe_key in sd and max_len_stored is not None:
            pe_shape = sd[pe_key].shape  # [1, actual_max_len, d_model]
            actual_max_len = pe_shape[1]  # e.g. 72
            # If stored max_len == actual_max_len the +1 will overshoot → fix
            if max_len_stored == actual_max_len:
                corrected = max_len_stored - 1
                log.warning(
                    f"[load_model] Detected double-increment in checkpoint hparams: "
                    f"max_len={max_len_stored} but PE shape implies pre-increment "
                    f"value should be {corrected}. Patching hparams before load."
                )
                ckpt_data["hyper_parameters"]["model_params"]["max_len"] = corrected
        torch.save(ckpt_data, checkpoint + ".patched.tmp")
        checkpoint_to_load = checkpoint + ".patched.tmp"
    else:
        checkpoint_to_load = checkpoint

    try:
        model = CVAELightningModule.load_from_checkpoint(
            checkpoint_to_load,
            map_location=device,
            strict=False,
            **extra_kwargs,
        )
    finally:
        # Remove temp file regardless of success/failure
        import os
        tmp = checkpoint + ".patched.tmp"
        if os.path.exists(tmp):
            os.remove(tmp)

    model.to(device)
    model.eval()
    return model


def generate_for_target(model, target: float, n_samples: int, seed: int, device) -> pd.DataFrame:
    """Generate one recorded row for each requested sampling attempt."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    generated = []

    with torch.no_grad():
        while len(generated) < n_samples:
            batch = min(128, n_samples - len(generated))
            try:
                # generate_molecules_with_target_label expects a scalar float
                # (it builds labels internally via torch.full)
                smiles_list = model.generate_molecules_with_target_label(
                    num_samples=batch,
                    target_label=float(target),
                    temperature=1.0,
                )
                smiles_list = list(smiles_list)
                if len(smiles_list) != batch:
                    log.warning(
                        "Generation returned %d results for %d attempts; "
                        "recording missing results as invalid",
                        len(smiles_list),
                        batch,
                    )
                generated.extend(smiles_list[:batch])
                generated.extend([None] * max(0, batch - len(smiles_list)))
            except Exception as e:
                log.warning(
                    "Generation batch of %d attempts failed; recording all as invalid: %s",
                    batch,
                    e,
                )
                generated.extend([None] * batch)

    valid = [isinstance(smiles, str) and bool(smiles) for smiles in generated]
    log.info(
        "  target=%.3f seed=%d: attempted=%d valid=%d invalid=%d",
        target,
        seed,
        n_samples,
        sum(valid),
        n_samples - sum(valid),
    )
    return pd.DataFrame(
        {
            "attempt_id": range(n_samples),
            "smiles": generated,
            "valid": valid,
            "target": target,
            "seed": seed,
        }
    )


def score_sa(df: pd.DataFrame) -> pd.DataFrame:
    """Add sa_score column using RDKit sascorer."""
    try:
        from rdkit import Chem
        from rdkit.Chem import RDConfig
        import sys, os
        sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
        import sascorer

        scores = []
        for smi in df["smiles"]:
            try:
                mol = Chem.MolFromSmiles(smi)
                scores.append(sascorer.calculateScore(mol) if mol else float("nan"))
            except Exception:
                scores.append(float("nan"))
        df = df.copy()
        df["sa_score"] = scores
    except ImportError:
        log.warning("sascorer not available. Skipping SA scoring (add manually)")
    return df


def main():
    parser = argparse.ArgumentParser(description="Exp19 ablation generation")
    parser.add_argument("--run_id", required=True, help="Run identifier (e.g. REF, A2_det)")
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    parser.add_argument("--config", required=True, help="Path to training YAML config (unused at inference; kept for documentation)")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--n_samples", type=int, default=1000, help="Molecules per (target, seed)")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS,
                        help="Random seeds (default: 42 0 101)")
    parser.add_argument("--targets", nargs="+", type=float, default=None,
                        help="Override target values (default: 0.165 0.500 0.835)")
    parser.add_argument("--no_score", action="store_true", help="Skip SA scoring step")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}  Run: {args.run_id}")

    outdir = Path(args.outdir)

    # Load model
    model = load_model(args.checkpoint, args.config, device)

    # Determine targets
    target_map = {}
    if args.targets:
        for i, t in enumerate(args.targets):
            target_map[f"target_{i}"] = t
    else:
        target_map = TARGETS

    # Generate
    for seed in args.seeds:
        for bucket, target in target_map.items():
            out_path = outdir / f"seed_{seed}" / bucket / "generated.csv"
            if out_path.exists():
                log.info(f"  Skipping {out_path} (already exists)")
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)

            df = generate_for_target(model, target, args.n_samples, seed, device)
            if not args.no_score:
                df = score_sa(df)

            df.to_csv(out_path, index=False)
            log.info(
                "  Saved %d attempted rows (%d valid) to %s",
                len(df),
                int(df["valid"].sum()),
                out_path,
            )

    log.info(f"=== {args.run_id} generation complete ===")


if __name__ == "__main__":
    main()
