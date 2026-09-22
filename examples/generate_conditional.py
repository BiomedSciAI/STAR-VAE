#!/usr/bin/env python3
"""
generate_conditional.py
=======================
Generate property-conditioned molecules from a fine-tuned STAR-VAE CVAE.

Two generation modes
--------------------
1. **Label-conditioned prior sampling**: sample z from p(z|y) when the
   checkpoint enables a conditional prior, otherwise from N(0, I), then decode
   with the target label.
2. **Input-seeded sampling**: encode a SMILES string to µ, add small noise,
   condition on a target label.  Produces analogues near the input molecule
   that are nudged toward the desired property value.

Supported fine-tuned checkpoints (see CHECKPOINTS.md)
-----------------------------------------------------
  Checkpoint                          Target property    Label range
  ─────────────────────────────────── ────────────────   ──────────────────────
  cvae_sa_score/last.ckpt      SA score           0.0 (easy) → 1.0 (hard)
  cvae_1syh_stratified/last.ckpt      1SYH docking       0.0 (strong) to 1.0 (weak)
  cvae_4lde_stratified/last.ckpt      4LDE docking       0.0 (strong) to 1.0 (weak)
  cvae_6y2f_stratified/last.ckpt      6Y2F docking       0.0 (strong) to 1.0 (weak)

  Note: labels are normalized to [0, 1], but semantics are checkpoint-specific.
  SA-score checkpoints use min-max labels (lower is easier to synthesize).
  Stratified Tartarus checkpoints use percentile ranks of docking score, where
  0 is the strongest/more-negative end and 1 is the weakest/less-negative end.

Usage examples
--------------
# Generate 32 molecules with SA score label = 0.25 (easy to synthesise):
python examples/generate_conditional.py \\
    --checkpoint checkpoints/cvae_sa_score/last.ckpt \\
    --vocab     $VOCAB_FILE \\
    --target-label 0.25 \\
    --num-samples 32 \\
    --output    outputs/sa_score_0.25.csv

# Input-seeded: start from aspirin, target label = 0.25:
python examples/generate_conditional.py \\
    --checkpoint   checkpoints/cvae_sa_score/last.ckpt \\
    --vocab        $VOCAB_FILE \\
    --input-smiles "CC(=O)Oc1ccccc1C(=O)O" \\
    --target-label 0.25 \\
    --noise-scale  0.4 \\
    --num-samples  32

# Tartarus 1SYH docking - generate high-affinity molecules (label near 0.0):
python examples/generate_conditional.py \\
    --checkpoint   checkpoints/cvae_1syh_stratified/last.ckpt \\
    --vocab        $VOCAB_FILE \\
    --target-label 0.1 \\
    --num-samples  32

Requirements
------------
Install the package first:  pip install -e .   (or: uv pip install -e .)
"""

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import selfies as sf
import torch
from rdkit import Chem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_smiles(smiles: str, model, tokenizer, device: torch.device) -> torch.Tensor:
    """
    Encode a SMILES string to its mean latent vector µ.

    Returns
    -------
    mu : torch.Tensor, shape (1, latent_dim)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles!r}")
    canonical_smiles = Chem.MolToSmiles(mol)
    selfies_str = sf.encoder(canonical_smiles)

    enc = tokenizer(
        selfies_str,
        return_tensors="pt",
        max_length=model.max_len,
        truncation=True,
        padding="max_length",
    )
    input_ids = enc["input_ids"].to(device)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)   # (seq_len,) → (1, seq_len)

    model.eval()
    with torch.no_grad():
        mu, _ = model.model.encode(input_ids[:, :-1])
    return mu   # (1, latent_dim)


def decode_z_with_label(model, z: torch.Tensor, label: float,
                         temperature: float, top_k: int | None,
                         device: torch.device) -> list[str | None]:
    """
    Run autoregressive decoding of z conditioned on `label`.
    Returns a list of SMILES strings (None for invalid molecules).
    """
    from bmfm_sm_gen.trainers.lm_cvae import generate_molecules_with_labels

    batch_size = z.size(0)
    labels = torch.full((batch_size, 1), label, dtype=torch.float, device=device)

    model.model.eval()
    with torch.no_grad():
        smiles_list = generate_molecules_with_labels(
            model=model.model,
            tokenizer=model.tokenizer,
            z=z,
            labels=labels,
            temperature=temperature,
            top_k=top_k,
            lora=model.lora,
        )
    return smiles_list


def print_results(smiles_list: list, title: str = "Generated molecules"):
    valid = [s for s in smiles_list if s is not None]
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")
    print(f"  Requested : {len(smiles_list)}")
    print(f"  Valid     : {len(valid)}  ({100 * len(valid) / max(1, len(smiles_list)):.1f}%)")
    print(f"{'─' * 60}")
    for i, smi in enumerate(valid, 1):
        print(f"  {i:3d}.  {smi}")
    print()


def save_csv(smiles_list: list, path: str, target_label: float):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles", "target_label"])
        for smi in smiles_list:
            writer.writerow([smi, target_label])
    valid = sum(1 for s in smiles_list if s is not None)
    print(f"Saved {len(smiles_list)} attempts ({valid} valid) → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate property-conditioned molecules using a STAR-VAE CVAE.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint", "-c",
        default=os.environ.get("CKPT_SA_SCORE", ""),
        help="Path to the fine-tuned CVAE .ckpt file  (default: $CKPT_SA_SCORE)",
    )
    parser.add_argument(
        "--vocab", "-v",
        default=os.environ.get("VOCAB_FILE", "checkpoints/vae_pubchem80m/vocab.json"),
        help="Path to the vocabulary JSON  (default: $VOCAB_FILE)",
    )
    parser.add_argument(
        "--target-label", "-l",
        type=float,
        required=True,
        metavar="FLOAT",
        help="Target label in [0, 1]; normalization semantics depend on the checkpoint.",
    )
    parser.add_argument(
        "--input-smiles", "-i",
        default=None,
        metavar="SMILES",
        help="Optional SMILES to encode and use as the latent seed. "
             "If omitted, sample the checkpoint's conditional prior when enabled, "
             "otherwise N(0, I).",
    )
    parser.add_argument(
        "--noise-scale", "-n",
        type=float,
        default=0.3,
        help="Noise added to µ when --input-smiles is given  (default: 0.3)",
    )
    parser.add_argument(
        "--num-samples", "-k",
        type=int,
        default=32,
        help="Number of molecules to generate  (default: 32)",
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=1.0,
        help="Sampling temperature  (default: 1.0)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-K token filtering during decoding  (default: off)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed  (default: 42)",
    )
    parser.add_argument(
        "--prior-mode",
        choices=("auto", "standard", "conditional"),
        default="auto",
        help="Latent prior for non-seeded generation. 'auto' uses p(z|y) when enabled.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device: 'cuda' or 'cpu'  (default: auto-detect)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Optional path to save a CSV of generated SMILES.",
    )
    args = parser.parse_args()

    # ── Validation ────────────────────────────────────────────────────────────
    if not args.checkpoint:
        parser.error(
            "No checkpoint specified.  Pass --checkpoint or set $CKPT_SA_SCORE."
        )
    if not Path(args.checkpoint).exists():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    if not Path(args.vocab).exists():
        parser.error(f"Vocab file not found: {args.vocab}")
    if not (0.0 <= args.target_label <= 1.0):
        parser.error(f"--target-label must be in [0, 1], got {args.target_label}")
    if args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    if args.noise_scale < 0:
        parser.error("--noise-scale must be non-negative")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("--temperature must be a finite number greater than 0")
    if args.top_k is not None and args.top_k <= 0:
        parser.error("--top-k must be positive when provided")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    print("WARNING: PyTorch checkpoints may execute code; load only trusted files.")
    print(f"Loading checkpoint : {args.checkpoint}")
    from bmfm_sm_gen.trainers.lm_cvae import CVAELightningModule

    model = CVAELightningModule.load_from_checkpoint(
        args.checkpoint,
        vocab_file=args.vocab,
        map_location=device,
    )
    model.eval()
    model.to(device)

    print(
        f"Model loaded  │  latent_dim={model.model.latent_dim}"
        f"  max_len={model.max_len}"
        f"  lora={model.lora!r}"
        f"  device={device}"
    )
    print(f"Target label   : {args.target_label}")

    # ── Build latent vectors z ────────────────────────────────────────────────
    if args.input_smiles:
        print(f"\nEncoding input SMILES: {args.input_smiles!r}")
        mu = encode_smiles(args.input_smiles, model, model.tokenizer, device)
        print(f"µ norm = {mu.norm().item():.3f}  (latent_dim={mu.shape[-1]})")
        mu_expanded = mu.expand(args.num_samples, -1)
        noise = torch.randn_like(mu_expanded) * args.noise_scale
        z = mu_expanded + noise
        mode = f"input-seeded (noise_scale={args.noise_scale})"
    else:
        labels = torch.full(
            (args.num_samples, model.model.label_dim),
            args.target_label,
            dtype=torch.float,
            device=device,
        )
        use_conditional_prior = (
            model.model.use_conditional_prior and args.prior_mode != "standard"
        )
        if args.prior_mode == "conditional" and not model.model.use_conditional_prior:
            parser.error("--prior-mode conditional requires a conditional-prior checkpoint")
        if use_conditional_prior:
            prior_mu, prior_logvar = model.model.get_conditional_prior(labels)
            z = model.model.reparameterize(prior_mu, prior_logvar)
            mode = "label-conditioned prior (z ~ p(z|y))"
        else:
            z = torch.randn(args.num_samples, model.model.latent_dim, device=device)
            mode = "label-conditioned standard prior (z ~ N(0, I))"

    print(f"Generation mode : {mode}")
    print(f"Num samples     : {args.num_samples}")
    print(f"Temperature     : {args.temperature}")
    print(f"Top-K           : {args.top_k or 'off'}")

    # ── Generate ──────────────────────────────────────────────────────────────
    smiles_list = decode_z_with_label(
        model=model,
        z=z,
        label=args.target_label,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )

    # ── Print and optionally save ─────────────────────────────────────────────
    title = (
        f"Analogues of {args.input_smiles!r} | target_label={args.target_label}"
        if args.input_smiles
        else f"Conditional molecules | target_label={args.target_label}"
    )
    print_results(smiles_list, title)

    if args.output:
        save_csv(smiles_list, args.output, args.target_label)


if __name__ == "__main__":
    main()
