"""
exp17c_sa_score_generate.py
============================
Exp17-C extension: Generate molecules using FiLM-only fine-tuned CVAE for SA Score.

Same sweep protocol as Tartarus Exp17-C:
  - y* ∈ {0.000, 0.001, ..., 1.000} (1001 levels) × 3 samples = 3003 attempts
  - Alpha grid: [1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]
  - Alpha=1.0 → standard conditioning (no CFG); alpha>1.0 → CFG amplification

Label convention:
  y=0 → easiest to synthesize (sa_score→1.44), y=1 → hardest (sa_score→7.51)
  NO y-flip (unlike Tartarus where y_model = 1 - y_sweep).
  The SA Score model was trained directly with y_normalized_minmax as label.

Loading strategy (same as Tartarus exp17c_generate.py):
  1. Load SA Score base checkpoint (Exp15 sa_score_cvae_minmax/last.ckpt)
  2. Load Exp17-C trained checkpoint state_dict
  3. Overwrite conditioning pathway weights (modulation, prior, latent_to_hidden, label_to_dmodel)

Output: paper_results/exp17_improving_conditioning/C_film_only_finetune/
        generated/sa_score/alpha_{α}/sweep.csv

Usage:
    python bmfm_sm_gen/scripts/exp17c_sa_score_generate.py --alpha 1.0
    python bmfm_sm_gen/scripts/exp17c_sa_score_generate.py --alpha 8.0
    python bmfm_sm_gen/scripts/exp17c_sa_score_generate.py --alpha 1.0 --smoke-test
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

DATASET_NAME = "sa_score"

# ── Exp17-C trained checkpoint for SA Score ────────────────────────────────────
EXP17C_CHECKPOINT = str(EXP_ROOT / DATASET_NAME / "checkpoints" / "last.ckpt")

# ── Base SA Score checkpoint (Exp15) ──────────────────────────────────────────
BASE_CHECKPOINT = str(REPO_ROOT / "paper_results/exp15_controllability/sa_score_cvae_minmax/checkpoints/last.ckpt")

# ── Sweep parameters ──────────────────────────────────────────────────────────
Y_STEP            = 0.001
SAMPLES_PER_LEVEL = 3

BUCKET_BOUNDS = {
    "high": (0.000, 0.333),  # high = low SA score = easy to synthesize
    "mid":  (0.333, 0.667),
    "low":  (0.667, 1.001),  # low = high SA score = hard to synthesize
}


def assign_bucket(y: float) -> str:
    for name, (lo, hi) in BUCKET_BOUNDS.items():
        if lo <= y < hi:
            return name
    return "low"


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(
    device: torch.device,
    checkpoint: str = EXP17C_CHECKPOINT,
    base_checkpoint: str = BASE_CHECKPOINT,
    vocab: str = VOCAB,
):
    """
    Load Exp17-C SA Score trained checkpoint.

    Strategy:
    1. Load SA Score base checkpoint (Exp15) as architecture
    2. Extract conditioning weights from Exp17-C state_dict
    3. Overlay conditioning weights onto base model
    """
    from bmfm_sm_gen.trainers.lm_cvae import CVAELightningModule
    from bmfm_sm_gen.utils.trainer_utils import import_class

    if not Path(checkpoint).exists():
        raise FileNotFoundError(
            f"Exp17-C SA Score checkpoint not found: {checkpoint}\n"
            f"Run training first: python run_exp17c_sa_score_train.py"
        )

    log.info(f"Loading Exp17-C SA Score checkpoint: {checkpoint}")

    # Load base SA Score architecture
    raw_base = torch.load(base_checkpoint, map_location="cpu")
    hp_base = raw_base["hyper_parameters"]

    base_lm = CVAELightningModule.load_from_checkpoint(
        base_checkpoint,
        strict=False,
        map_location=device,
        model_params=hp_base["model_params"],
        vocab_file=vocab,
        val_csv=None,
        label_norm="minmax",
    )
    model = base_lm.model
    log.info(f"  Loaded base SA Score architecture")

    # Extract conditioning weights from Exp17-C checkpoint
    raw_exp17c = torch.load(checkpoint, map_location="cpu")
    exp17c_sd = raw_exp17c["state_dict"]
    # Keys stored as "model.XXX" inside FiLMOnlyCVAEModule
    model_sd = {k[len("model."):]: v for k, v in exp17c_sd.items() if k.startswith("model.")}

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
    log.info(f"  Loaded {len(loaded_keys)} Exp17-C conditioning weights")
    if not loaded_keys:
        log.warning("  No Exp17-C conditioning weights found! Check checkpoint format.")

    tokenizer_cls = import_class("bmfm_sm_gen.data.base_tokenizer.SELFIESTokenizer")
    tokenizer = tokenizer_cls(vocab=vocab)

    model.to(device)
    model.eval()
    return model, tokenizer


# ── CFG generation (identical to Tartarus version) ────────────────────────────

def generate_cfg_one(model, tokenizer, z, y_cond, alpha, device):
    import selfies as sf
    from rdkit import Chem

    max_len    = model.max_len
    y_null     = torch.zeros_like(y_cond)
    pad_id     = tokenizer.pad_token_id
    eos_id     = tokenizer.eos_token_id
    bos_id     = tokenizer.bos_token_id
    sqrt_dmodel = math.sqrt(model.decoder.d_model)

    with torch.no_grad():
        zc_cond    = model.modulation(z, y_cond)
        h_cond     = model.latent_to_hidden(zc_cond)
        mem_cond   = h_cond.unsqueeze(1).repeat(1, max_len, 1).transpose(0, 1)

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

def generate_sweep(model, tokenizer, alpha, seed, device, smoke_test=False):
    """Sweep y* from 0.000 to 1.000. No y-flip for SA Score."""
    pl.seed_everything(seed)
    y_grid = np.round(np.arange(0.0, 1.0 + Y_STEP / 2, Y_STEP), 6)
    y_all  = np.repeat(y_grid, SAMPLES_PER_LEVEL)

    if smoke_test:
        y_all = y_all[:30]
        log.info("  Smoke test: generating 30 molecules only")

    total = len(y_all)
    log.info(
        f"CFG sweep: alpha={alpha}  {len(y_grid)} y* levels × {SAMPLES_PER_LEVEL} = "
        f"{total} attempts (no y-flip for SA Score)"
    )

    rows = []
    for i, y_val in enumerate(y_all):
        # SA Score: no y-flip (y_model = y_sweep directly)
        y_model = round(float(y_val), 6)
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
                "dataset":             DATASET_NAME,
            })

        if (i + 1) % 300 == 0 or (i + 1) == total:
            log.info(f"  {i+1}/{total} attempted, {len(rows)} valid so far")

    return pd.DataFrame(rows)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Exp17-C SA Score generation")
    p.add_argument("--alpha",      type=float, required=True,
                   help="CFG guidance scale (1.0=standard, >1.0=amplified)")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--checkpoint", default=EXP17C_CHECKPOINT,
                   help="Exp17-C SA-score checkpoint path")
    p.add_argument("--base-checkpoint", default=BASE_CHECKPOINT,
                   help="Base SA-score CVAE checkpoint path")
    p.add_argument("--vocab", default=VOCAB, help="Vocabulary JSON path")
    p.add_argument("--output-dir", help="Directory for sweep.csv")
    p.add_argument("--smoke-test", action="store_true",
                   help="Generate only 30 molecules for testing")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}  alpha={args.alpha}  seed={args.seed}  dataset={DATASET_NAME}")

    model, tokenizer = load_model(
        device,
        checkpoint=args.checkpoint,
        base_checkpoint=args.base_checkpoint,
        vocab=args.vocab,
    )

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else EXP_ROOT / "generated" / DATASET_NAME / f"alpha_{args.alpha:.1f}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    df = generate_sweep(model, tokenizer, args.alpha, args.seed, device,
                        smoke_test=args.smoke_test)
    df["seed"] = args.seed

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
    log.info("\n=== exp17c_sa_score_generate.py complete ===")


if __name__ == "__main__":
    main()
