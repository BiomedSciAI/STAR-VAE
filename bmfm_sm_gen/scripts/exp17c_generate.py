"""
exp17c_generate.py
==================
Exp17-C: Generate molecules using FiLM-only fine-tuned CVAE checkpoints.

Two modes:
  1. Standard conditioning only (alpha=1.0): baseline for Exp17-C
  2. CFG (alpha > 1.0): combined Exp17-C + CFG to test if stronger FiLM
     weights allow CFG to work for 1SYH/4LDE

Label convention: same as Exp17-B.
  y_model = 1 - y_sweep  (inverted, model was trained with y=0→best binder
  but the underlying v2 checkpoint and FiLM weights learned inverted mapping;
  the flip was verified in Exp17-C toy-validate step)

Sweep: y* ∈ {0.000, 0.001, ..., 1.000}, 3 samples per level = 3003 attempts.
Same protocol as Exp17-B for direct comparison.

Usage:
    python bmfm_sm_gen/scripts/exp17c_generate.py --target 6y2f --alpha 1.0
    python bmfm_sm_gen/scripts/exp17c_generate.py --target 1syh --alpha 8.0
    python bmfm_sm_gen/scripts/exp17c_generate.py --target all  --alpha 1.0
    python bmfm_sm_gen/scripts/exp17c_generate.py --target 6y2f --alpha 1.0 --smoke-test
"""

import argparse
import logging
import os
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT  = REPO_ROOT / "paper_results/exp17_improving_conditioning/C_film_only_finetune"
VOCAB     = os.environ.get("VOCAB_FILE", "checkpoints/vae_pubchem80m/vocab.json")

# ── Exp17-C trained checkpoints ───────────────────────────────────────────────
CHECKPOINTS = {
    "6y2f": str(EXP_ROOT / "6y2f" / "checkpoints" / "last.ckpt"),
    "1syh": str(EXP_ROOT / "1syh" / "checkpoints" / "last.ckpt"),
    "4lde": str(EXP_ROOT / "4lde" / "checkpoints" / "last.ckpt"),
}

# ── Sweep parameters ──────────────────────────────────────────────────────────
Y_STEP            = 0.001
SAMPLES_PER_LEVEL = 3

BUCKET_BOUNDS = {
    "high": (0.000, 0.333),
    "mid":  (0.333, 0.667),
    "low":  (0.667, 1.001),
}


def assign_bucket(y: float) -> str:
    for name, (lo, hi) in BUCKET_BOUNDS.items():
        if lo <= y < hi:
            return name
    return "low"


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(
    target: str,
    device: torch.device,
    checkpoint: str | None = None,
    base_checkpoint: str | None = None,
    vocab: str = VOCAB,
):
    """
    Load Exp17-C trained checkpoint.

    The Exp17-C training script saves the model state directly (FiLM weights
    only changed). The checkpoint is a full pl.LightningModule state dict
    saved by FiLMOnlyCVAEModule, but the model inside is a standard
    ConditionalTransformerVAE so we can load it via CVAELightningModule.

    Applies same max_len and prior shape fixes as Exp17-B.
    """
    from bmfm_sm_gen.trainers.lm_cvae import CVAELightningModule
    from bmfm_sm_gen.utils.trainer_utils import import_class

    ckpt_path = checkpoint or CHECKPOINTS[target]
    log.info(f"Loading Exp17-C checkpoint: {ckpt_path}")
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(
            f"Exp17-C checkpoint not found: {ckpt_path}\n"
            f"Run training first: python run_exp17c_train.py {target}"
        )

    raw_ckpt = torch.load(ckpt_path, map_location="cpu")

    # The FiLMOnlyCVAEModule saves its own hparams, so we rebuild using
    # the original model_params from the underlying CVAELightningModule.
    # The raw_ckpt may have a FiLMOnlyCVAEModule hyper_parameters dict.
    # We load the state_dict and patch manually into CVAELightningModule.
    # Strategy: load normalized-v2 as base, then overwrite conditioning weights.

    from bmfm_sm_gen.scripts.run_exp17c_train import load_normalized_v2_checkpoint

    EXPT_BASE = os.environ.get("OUTPUT_DIR", "outputs")
    BASE_CKPTS = {
        "6y2f": f"{EXPT_BASE}/cvae_6y2f_normalized_v2/checkpoints/last.ckpt",
        "1syh": f"{EXPT_BASE}/cvae_1syh_normalized_v2/checkpoints/last.ckpt",
        "4lde": f"{EXPT_BASE}/cvae_4lde_normalized_v2/checkpoints/last.ckpt",
    }

    # Load base architecture
    base_ckpt_path = base_checkpoint or BASE_CKPTS[target]
    _, model, tokenizer = load_normalized_v2_checkpoint(
        base_ckpt_path, device, vocab_path=vocab
    )

    # Extract model state_dict from Exp17-C checkpoint
    # Keys are stored as "model.XXX" inside the FiLMOnlyCVAEModule state_dict
    exp17c_sd = raw_ckpt["state_dict"]
    # Remap keys: "model.XXX" → "XXX"
    model_sd = {k[len("model."):]: v for k, v in exp17c_sd.items() if k.startswith("model.")}

    # Load only the conditioning pathway keys that Exp17-C trained
    CONDITIONING_PREFIXES = [
        "modulation.",
        "conditional_prior_mu.",
        "conditional_prior_logvar.",
        "latent_to_hidden.",
        "label_to_dmodel.",
    ]
    loaded_keys = []
    current_sd = model.state_dict()
    for key, val in model_sd.items():
        if any(key.startswith(pfx) for pfx in CONDITIONING_PREFIXES):
            if key in current_sd:
                current_sd[key] = val.to(device)
                loaded_keys.append(key)
    model.load_state_dict(current_sd, strict=False)
    log.info(f"  Loaded {len(loaded_keys)} Exp17-C conditioning weights into model")
    if not loaded_keys:
        log.warning("  No Exp17-C conditioning weights found! Check checkpoint format.")

    model.to(device)
    model.eval()
    return model, tokenizer


# ── CFG generation ─────────────────────────────────────────────────────────────

def generate_cfg_one(
    model,
    tokenizer,
    z:      torch.Tensor,
    y_cond: torch.Tensor,
    alpha:  float,
    device: torch.device,
) -> str | None:
    """
    Generate one molecule with Classifier-Free Guidance (same as Exp17-B).
    alpha=1.0 → standard conditioning (no CFG amplification).
    alpha>1.0 → amplified conditioning.
    """
    import selfies as sf
    from rdkit import Chem

    max_len    = model.max_len
    y_null     = torch.zeros_like(y_cond)
    pad_id     = tokenizer.pad_token_id
    eos_id     = tokenizer.eos_token_id
    bos_id     = tokenizer.bos_token_id
    sqrt_dmodel = math.sqrt(model.decoder.d_model)

    with torch.no_grad():
        zc_cond   = model.modulation(z, y_cond)
        h_cond    = model.latent_to_hidden(zc_cond)
        mem_cond  = h_cond.unsqueeze(1).repeat(1, max_len, 1).transpose(0, 1)

        zc_uncond  = model.modulation(z, y_null)
        h_uncond   = model.latent_to_hidden(zc_uncond)
        mem_uncond = h_uncond.unsqueeze(1).repeat(1, max_len, 1).transpose(0, 1)

    tokens = torch.full((1, 1), bos_id, dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_len - 1):
            tgt_emb  = model.decoder.embedding(tokens) * sqrt_dmodel
            tgt_emb  = model.decoder.pos_decoder(tgt_emb)
            tgt_emb  = tgt_emb.transpose(0, 1)
            tgt_mask = model.decoder.generate_square_subsequent_mask(
                tgt_emb.size(0)
            ).to(device)
            tgt_pad_mask = (tokens == pad_id)

            out_cond = model.decoder.transformer_decoder(
                tgt_emb, mem_cond,
                tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask,
            )
            logits_cond = model.decoder.output_layer(out_cond.transpose(0, 1)[:, -1, :])

            out_uncond = model.decoder.transformer_decoder(
                tgt_emb, mem_uncond,
                tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask,
            )
            logits_uncond = model.decoder.output_layer(out_uncond.transpose(0, 1)[:, -1, :])

            logits_cfg = logits_uncond + alpha * (logits_cond - logits_uncond)
            next_tok   = logits_cfg.argmax(dim=-1, keepdim=True)
            tokens     = torch.cat([tokens, next_tok], dim=1)

            if next_tok.item() == eos_id:
                break

    # Decode tokens → SMILES
    token_ids = tokens[0, 1:].tolist()
    selfies_tokens = []
    for tid in token_ids:
        if tid == eos_id:
            break
        if tid != pad_id:
            selfies_tokens.append(tokenizer.convert_ids_to_tokens(tid))

    selfies_str = "".join(selfies_tokens)
    try:
        smiles = sf.decoder(selfies_str)
        from rdkit import Chem
        if smiles and Chem.MolFromSmiles(smiles):
            return smiles
    except Exception:
        pass
    return None


# ── Systematic sweep ──────────────────────────────────────────────────────────

def generate_sweep(
    model,
    tokenizer,
    alpha:  float,
    seed:   int,
    device: torch.device,
    smoke_test: bool = False,
) -> pd.DataFrame:
    """Sweep y* from 0.000 to 1.000 in Y_STEP increments."""
    pl.seed_everything(seed)
    y_grid = np.round(np.arange(0.0, 1.0 + Y_STEP / 2, Y_STEP), 6)
    y_all  = np.repeat(y_grid, SAMPLES_PER_LEVEL)

    if smoke_test:
        y_all = y_all[:30]  # 10 levels × 3 samples
        log.info("  Smoke test: generating 30 molecules only")

    total = len(y_all)
    log.info(
        f"CFG sweep: alpha={alpha}  {len(y_grid)} y* levels × {SAMPLES_PER_LEVEL} = "
        f"{total} attempts"
    )

    rows = []
    for i, y_val in enumerate(y_all):
        # Flip: sweep y*=0 (best binder) → model receives 1.0
        y_model = round(1.0 - float(y_val), 6)
        y_t     = torch.tensor([[y_model]], dtype=torch.float32, device=device)

        with torch.no_grad():
            z = model.reparameterize(*model.get_conditional_prior(y_t))

        smiles = generate_cfg_one(model, tokenizer, z, y_t, alpha, device)
        if smiles:
            rows.append({
                "smiles":              smiles,
                "conditioning_y_star": round(float(y_val), 6),
                "y_model":             y_model,
                "bucket":              assign_bucket(float(y_val)),
                "alpha":               alpha,
            })

        if (i + 1) % 300 == 0 or (i + 1) == total:
            log.info(f"  {i+1}/{total} attempted, {len(rows)} valid so far")

    return pd.DataFrame(rows)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Exp17-C generation")
    p.add_argument("--target",     required=True, choices=["6y2f", "1syh", "4lde", "all"])
    p.add_argument("--alpha",      type=float, required=True,
                   help="CFG guidance scale (1.0=standard conditioning, >1=amplified)")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument(
        "--checkpoint",
        help="Exp17-C checkpoint path. For --target all, include '{target}' in the path.",
    )
    p.add_argument(
        "--base-checkpoint",
        help="Base CVAE checkpoint path. For --target all, include '{target}' in the path.",
    )
    p.add_argument("--vocab", default=VOCAB, help="Vocabulary JSON path")
    p.add_argument(
        "--output-dir",
        help="Output root; writes <target>/alpha_<scale>/sweep.csv. Default preserves the paper_results layout.",
    )
    p.add_argument("--smoke-test", action="store_true",
                   help="Generate only 30 molecules per target for testing")
    args = p.parse_args()

    for option in ("checkpoint", "base_checkpoint"):
        value = getattr(args, option)
        if args.target == "all" and value and "{target}" not in value:
            p.error(f"--{option.replace('_', '-')} must include '{{target}}' with --target all")

    targets = ["6y2f", "1syh", "4lde"] if args.target == "all" else [args.target]
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}  alpha={args.alpha}  seed={args.seed}")

    for target in targets:
        log.info(f"\n{'='*60}\n Target: {target}  Alpha: {args.alpha}\n{'='*60}")

        checkpoint = args.checkpoint.format(target=target) if args.checkpoint else None
        base_checkpoint = (
            args.base_checkpoint.format(target=target) if args.base_checkpoint else None
        )
        model, tokenizer = load_model(
            target,
            device,
            checkpoint=checkpoint,
            base_checkpoint=base_checkpoint,
            vocab=args.vocab,
        )

        if args.output_dir:
            out_dir = Path(args.output_dir) / target / f"alpha_{args.alpha:.1f}"
        else:
            out_dir = EXP_ROOT / "generated" / target / f"alpha_{args.alpha:.1f}"
        out_dir.mkdir(parents=True, exist_ok=True)

        df = generate_sweep(model, tokenizer, args.alpha, args.seed, device,
                            smoke_test=args.smoke_test)
        df["target"] = target
        df["seed"]   = args.seed

        out_csv = out_dir / "sweep.csv"
        df.to_csv(out_csv, index=False)

        for b in ["high", "mid", "low"]:
            sub = df[df["bucket"] == b]
            log.info(f"  {b:4s}: {len(sub):5d} valid")

        total_attempts = 30 if args.smoke_test else len(
            np.repeat(np.round(np.arange(0.0, 1.0 + Y_STEP / 2, Y_STEP), 6), SAMPLES_PER_LEVEL)
        )
        validity = 100 * len(df) / max(1, total_attempts)
        log.info(f"  Total: {len(df)} valid ({validity:.1f}%)  → {out_csv}")

    log.info("\n=== exp17c_generate.py complete ===")


if __name__ == "__main__":
    main()
