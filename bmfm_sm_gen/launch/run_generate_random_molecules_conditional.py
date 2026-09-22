import os
import pandas as pd
import numpy as np
from pathlib import Path
import torch

from bmfm_sm_gen.trainers.lm_cvae import CVAELightningModule
import pytorch_lightning as pl
import argparse
from bmfm_sm_gen.utils.inference_utils import sample_distribution_ranges


def main(checkpoint, vocab_file, num_samples, target_label, output_file, seed, training_set, temperature, top_k, num_top_bins, label_column, prior_mode="auto"):
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a finite number greater than 0")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive when provided")
    print(f"WARNING: loading trusted PyTorch checkpoint: {checkpoint}")
    pl.seed_everything(seed)
    
    lm = CVAELightningModule.load_from_checkpoint(
        checkpoint,
        vocab_file=vocab_file,
        map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    )
    
    print(f'training_set: {training_set}')
    
    if training_set is None:
        smiles = lm.generate_molecules_with_target_label(num_samples=num_samples, target_label=target_label, temperature=temperature, top_k=top_k, prior_mode=prior_mode)
    else:
        df_train = pd.read_csv(training_set)
        sample_dist = sample_distribution_ranges(df_train, label_column, num_samples, 0, 1, 10, num_top_bins)
        print(f'Sample distribution: {sample_dist}')
        smiles = []
        for sample_obj in sample_dist:
            sample_label = round(np.mean(sample_obj['range']), 5)
            sample_count = sample_obj['num']
            if sample_count > 0:
                smiles = smiles + lm.generate_molecules_with_target_label(num_samples=sample_count, target_label=sample_label, temperature=temperature, top_k=top_k, prior_mode=prior_mode)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"smiles": smiles}).to_csv(output_file, index=False)
    valid_count = sum(x is not None for x in smiles)
    print(f'{valid_count} valid SMILES written from {num_samples} attempts: {output_file}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate molecules conditioned on a target property label using a fine-tuned CVAE."
    )

    parser.add_argument(
        '--checkpoint', '--ckpt',
        default=os.environ.get('CKPT_SA_SCORE'),
        required=os.environ.get('CKPT_SA_SCORE') is None,
        type=str,
        help='Path to the CVAE model checkpoint (.ckpt).'
    )

    parser.add_argument(
        '--vocab_file', '--vocab-file', '--vocab',
        default=os.environ.get('VOCAB_FILE', 'checkpoints/vae_pubchem80m/vocab.json'),
        type=str,
        help='Path to the SELFIES vocabulary file (default: $VOCAB_FILE).'
    )

    parser.add_argument(
        '--num_samples', '--num-samples',
        default=10,
        type=int,
        help='Number of samples to generate.'
    )
    
    parser.add_argument(
        '--target_label', '--target-label',
        default=-1,
        type=float,
        help='Target label to condition for generation.'
    )
    
    parser.add_argument(
        '--training_set', '--training-set',
        default=None,
        type=str,
        help='Training set to sample new data following the distribution'
    )
    
    parser.add_argument(
        '--output_file', '--output-file',
        default=os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'cvae_generated.csv'),
        type=str,
        help='Path to save the generated SMILES CSV file.'
    )
    
    parser.add_argument(
        '--seed',
        default=101,
        type=int,
        help='Seed set for generation.'
    )
    
    parser.add_argument(
        '--temperature',
        default=1.0,
        type=float,
        help='Temperature for generation.'
    )
    
    parser.add_argument(
        '--top_k', '--top-k',
        default=None,
        type=int,
        help='Top K for generation.'
    )
    
    parser.add_argument(
        '--num_top_bins', '--num-top-bins',
        default=None,
        type=int,
        help='Number of top bins for generation.'
    )
    
    parser.add_argument(
        '--label_column', '--label-column',
        default="y_normalized_minmax",
        type=str,
        help='Label column name.'
    )

    parser.add_argument(
        '--prior_mode', '--prior-mode',
        choices=['auto', 'standard', 'conditional'],
        default='auto',
        help="Latent prior: auto uses p(z|y) when available; standard forces N(0,I)."
    )


    args = parser.parse_args()
    main(args.checkpoint, args.vocab_file, args.num_samples, args.target_label, args.output_file, args.seed, args.training_set, args.temperature, args.top_k, args.num_top_bins, args.label_column, args.prior_mode)
