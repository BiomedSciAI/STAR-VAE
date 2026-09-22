#!/usr/bin/env python3
"""
Script to sample molecules from a dataset (e.g., PubChem).

This script randomly samples molecules from a large dataset and saves them to a CSV file.
Useful for creating baseline comparisons or sampling from reference datasets.
"""

import pandas as pd
import argparse
import os
import sys
from pathlib import Path


def sample_molecules(
    data_path: str,
    num_samples: int,
    output_csv: str,
    smiles_col: str = 'smiles',
    random_seed: int = 42
) -> None:
    """
    Sample molecules from a dataset.
    
    Args:
        data_path: Path to the input CSV file
        num_samples: Number of molecules to sample
        output_csv: Path to save the sampled molecules
        smiles_col: Name of the SMILES column in the input file
        random_seed: Random seed for reproducibility
    """
    print("=" * 80)
    print("Molecule Sampling Script")
    print("=" * 80)
    print(f"Input data: {data_path}")
    print(f"Number of samples: {num_samples}")
    print(f"Output file: {output_csv}")
    print(f"SMILES column: {smiles_col}")
    print(f"Random seed: {random_seed}")
    print("=" * 80)
    
    # Check if input file exists
    if not Path(data_path).exists():
        print(f"Error: Input file does not exist: {data_path}")
        sys.exit(1)
    
    # Load the dataset
    print(f"\nLoading dataset from {data_path}...")
    try:
        df = pd.read_csv(data_path)
        print(f"✓ Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    # Check if SMILES column exists
    if smiles_col not in df.columns:
        print(f"Error: SMILES column '{smiles_col}' not found in dataset")
        print(f"Available columns: {df.columns.tolist()}")
        sys.exit(1)
    
    # Check if we have enough molecules
    if len(df) < num_samples:
        print(f"Warning: Dataset has only {len(df)} molecules, but {num_samples} samples requested")
        print(f"Sampling all {len(df)} molecules instead")
        num_samples = len(df)
    
    # Sample molecules
    print(f"\nSampling {num_samples} molecules...")
    sampled_df = df.sample(n=num_samples, random_state=random_seed)
    print(f"✓ Sampled {len(sampled_df)} molecules")
    
    # Save to output file
    print(f"\nSaving sampled molecules to {output_csv}...")
    try:
        # Create output directory if it doesn't exist
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        sampled_df.to_csv(output_csv, index=False)
        print(f"✓ Successfully saved to {output_csv}")
        print(f"  Rows: {len(sampled_df)}")
        print(f"  Columns: {len(sampled_df.columns)}")
    except Exception as e:
        print(f"Error saving output file: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("Sampling completed successfully!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Sample molecules from a dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sample 10000 molecules from PubChem
  python run_sample_molecules.py \\
    --num_samples 10000 \\
    --output_csv sampled_molecules.csv
  
  # Sample with custom data path and SMILES column
  python run_sample_molecules.py \\
    --num_samples 5000 \\
    --output_csv output.csv \\
    --data_path /path/to/data.csv \\
    --smiles_col Smiles
        """
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        required=True,
        help='Number of molecules to sample'
    )
    
    parser.add_argument(
        '--output_csv',
        type=str,
        required=True,
        help='Path to save the sampled molecules CSV file'
    )
    
    parser.add_argument(
        '--data_path',
        type=str,
        default=os.environ.get('PUBCHEM_80M_CSV', 'data/pubchem/data_pubchem_80m.csv'),
        help='Path to the input dataset CSV file (default: $PUBCHEM_80M_CSV or data/pubchem/data_pubchem_80m.csv)'
    )
    
    parser.add_argument(
        '--smiles_col',
        type=str,
        default='smiles',
        help='Name of the SMILES column in the input file (default: smiles)'
    )
    
    parser.add_argument(
        '--random_seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Run sampling
    sample_molecules(
        data_path=args.data_path,
        num_samples=args.num_samples,
        output_csv=args.output_csv,
        smiles_col=args.smiles_col,
        random_seed=args.random_seed
    )


if __name__ == "__main__":
    main()
