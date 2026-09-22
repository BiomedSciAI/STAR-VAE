#!/usr/bin/env python3
"""
generate_unconditional.py
=========================
Generate molecules from the pre-trained STAR-VAE (PubChem-80M backbone).

Two generation modes
--------------------
1. **Random sampling**: sample latent vectors z ~ N(0, I) and decode them.
2. **Input-seeded sampling**: encode a SMILES string to obtain its mean
   latent vector µ, add Gaussian noise scaled by `noise_scale`, and decode.
   This produces analogues near the input molecule in latent space.

Usage examples
--------------
# Random generation (32 molecules, no input SMILES needed):
python examples/generate_unconditional.py \\
    --checkpoint checkpoints/vae_pubchem80m/last.ckpt \\
    --vocab     $VOCAB_FILE \\
    --num-samples 32 \\
    --output    outputs/generated.csv

# Input-seeded generation (analogues of aspirin):
python examples/generate_unconditional.py \\
    --checkpoint checkpoints/vae_pubchem80m/last.ckpt \\
    --vocab     $VOCAB_FILE \\
    --input-smiles "CC(=O)Oc1ccccc1C(=O)O" \\
    --noise-scale 0.5 \\
    --num-samples 32 \\
    --output    outputs/aspirin_analogues.csv

# With temperature and top-k sampling:
python examples/generate_unconditional.py \\
    --checkpoint checkpoints/vae_pubchem80m/last.ckpt \\
    --vocab     $VOCAB_FILE \\
    --num-samples 32 \\
    --temperature 1.1 \\
    --top-k 20

Requirements
------------
Install the package first:  pip install -e .   (or: uv pip install -e .)
"""

import argparse
import math
import os
import sys
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_smiles(smiles: str, model, tokenizer, device: torch.device) -> torch.Tensor:
    """
    Encode a SMILES string to its mean latent vector µ.

    Parameters
    ----------
    smiles : str
        Input SMILES (will be converted to SELFIES internally).
    model : VAELightningModule
        Loaded checkpoint.
    tokenizer :
        The tokenizer stored inside the model.
    device :
        Target device.

    Returns
    -------
    mu : torch.Tensor, shape (1, latent_dim)
    """
    import selfies as sf
    from rdkit import Chem

    # Validate and canonicalize
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles!r}")
    canonical_smiles = Chem.MolToSmiles(mol)

    # Convert to SELFIES
    selfies_str = sf.encoder(canonical_smiles)

    # Tokenize and ensure shape (1, seq_len)
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
        mu, _ = model.model.encode(input_ids[:, :-1])   # strip last token position
    return mu   # (1, latent_dim)


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


def save_csv(smiles_list: list, path: str):
    import csv
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles"])
        for smi in smiles_list:
            writer.writerow([smi or ""])
    valid = sum(1 for s in smiles_list if s)
    print(f"Saved {len(smiles_list)} attempts ({valid} valid) → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate molecules by random-prior or seed-based STAR-VAE sampling.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint", "-c",
        default=os.environ.get("PRETRAINED_VAE_CKPT", ""),
        help="Path to the .ckpt file  (default: $PRETRAINED_VAE_CKPT)",
    )
    parser.add_argument(
        "--vocab", "-v",
        default=os.environ.get("VOCAB_FILE", "checkpoints/vae_pubchem80m/vocab.json"),
        help="Path to the vocabulary JSON  (default: $VOCAB_FILE)",
    )
    parser.add_argument(
        "--input-smiles", "-i",
        default=None,
        metavar="SMILES",
        help="Optional SMILES to encode and use as the latent seed. "
             "If omitted, z is sampled from N(0, I) (pure random generation).",
    )
    parser.add_argument(
        "--noise-scale", "-n",
        type=float,
        default=0.3,
        help="Gaussian noise scale added to µ when --input-smiles is given "
             "(0 = greedy reconstruction, 1 = full random walk).  Default: 0.3",
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
        help="Sampling temperature. Higher values give more diversity (default: 1.0)",
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
        help="Random seed for reproducibility  (default: 42)",
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
            "No checkpoint specified.  Pass --checkpoint or set $PRETRAINED_VAE_CKPT."
        )
    if not Path(args.checkpoint).exists():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    if not Path(args.vocab).exists():
        parser.error(f"Vocab file not found: {args.vocab}")
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
    print(f"Loading checkpoint: {args.checkpoint}")
    from bmfm_sm_gen.trainers.lm_vae import VAELightningModule
    from bmfm_sm_gen.trainers.lm_vae import _sample_from_logits

    model = VAELightningModule.load_from_checkpoint(
        args.checkpoint,
        vocab_file=args.vocab,
        map_location=device,
    )
    model.eval()
    model.to(device)

    print(
        f"Model loaded  │  latent_dim={model.model.latent_dim}"
        f"  max_len={model.max_len}"
        f"  device={device}"
    )

    # ── Build latent vectors z ────────────────────────────────────────────────
    if args.input_smiles:
        print(f"\nEncoding input SMILES: {args.input_smiles!r}")
        mu = encode_smiles(args.input_smiles, model, model.tokenizer, device)
        print(f"µ norm = {mu.norm().item():.3f}  (latent_dim={mu.shape[-1]})")

        # Broadcast µ to (num_samples, latent_dim) and add noise
        mu_expanded = mu.expand(args.num_samples, -1)
        noise = torch.randn_like(mu_expanded) * args.noise_scale
        z = mu_expanded + noise
        mode = f"input-seeded (noise_scale={args.noise_scale})"
    else:
        z = torch.randn(args.num_samples, model.model.latent_dim, device=device)
        mode = "random (z ~ N(0, I))"

    print(f"Generation mode : {mode}")
    print(f"Num samples     : {args.num_samples}")
    print(f"Temperature     : {args.temperature}")
    print(f"Top-K           : {args.top_k or 'off'}")

    # ── Decode ────────────────────────────────────────────────────────────────
    # generate_molecules() internally builds z from N(0,I). We instead
    # call the decoder loop directly with our pre-built z.
    import selfies as sf
    from rdkit import Chem

    generated_tokens = torch.full(
        (args.num_samples, 1), model.tokenizer.bos_token_id, dtype=torch.long, device=device
    )

    model.model.eval()
    with torch.no_grad():
        for _ in range(model.max_len - 1):
            tgt_emb = model.model.decoder.embedding(generated_tokens) * math.sqrt(
                model.model.decoder.d_model
            )
            tgt_emb = model.model.decoder.pos_decoder(tgt_emb)
            tgt_emb = tgt_emb.transpose(0, 1)
            tgt_mask = model.model.decoder.generate_square_subsequent_mask(
                tgt_emb.size(0)
            ).to(device)
            tgt_key_padding_mask = generated_tokens == model.tokenizer.pad_token_id

            z_hidden = model.model.latent_to_hidden(z)
            memory = z_hidden.unsqueeze(1).repeat(1, model.model.max_len, 1).transpose(0, 1)

            output = model.model.decoder.transformer_decoder(
                tgt_emb, memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )
            output = output.transpose(0, 1)
            logits = model.model.decoder.output_layer(output[:, -1, :])
            try:
                predicted = _sample_from_logits(logits, args.temperature, args.top_k)
            except (TypeError, ValueError) as exc:
                parser.error(str(exc))
            generated_tokens = torch.cat([generated_tokens, predicted], dim=1)
            if (predicted == model.tokenizer.eos_token_id).all():
                break

    # ── Decode tokens → SMILES ────────────────────────────────────────────────
    generated_tokens = generated_tokens[:, 1:]   # strip <sos>
    smiles_list = []
    for seq in generated_tokens:
        tokens = []
        for idx in seq.tolist():
            if idx == model.tokenizer.eos_token_id:
                break
            if idx != model.tokenizer.pad_token_id:
                tokens.append(model.tokenizer.convert_ids_to_tokens(idx))
        selfies_str = "".join(tokens)
        try:
            smi = sf.decoder(selfies_str)
            mol = Chem.MolFromSmiles(smi)
            smiles_list.append(smi if mol is not None else None)
        except Exception:
            smiles_list.append(None)

    # ── Print and optionally save ─────────────────────────────────────────────
    title = (
        f"Analogues of {args.input_smiles!r}"
        if args.input_smiles
        else "Random molecules (unconditional VAE)"
    )
    print_results(smiles_list, title)

    if args.output:
        save_csv(smiles_list, args.output)


if __name__ == "__main__":
    main()
