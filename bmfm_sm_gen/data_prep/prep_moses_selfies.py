#!/usr/bin/env python
"""
prep_moses_selfies.py
======================
Convert standard MOSES dataset splits (SMILES) to SELFIES:
  - train: 1,584,663 rows (entire train split)
  - val: 176,074 rows (entire test split)
Uses multiprocessing for extremely fast conversion.
"""

import argparse
import logging
import os
from pathlib import Path
import pandas as pd
import selfies as sf
from multiprocessing import Pool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Download MOSES dataset from: https://github.com/molecularsets/moses
# or via: pip install molsets && python -c "from moses.dataset import get_dataset; get_dataset('train')"
INPUT_CSV = os.environ.get("MOSES_RAW_CSV", "data/moses_raw/dataset_v1.csv")
OUTPUT_DIR = os.environ.get("MOSES_PROCESSED_DIR", "data/moses_processed")

def convert_smiles_to_selfies(smiles: str) -> str:
    try:
        # Convert smiles to selfies on-the-fly
        selfies_str = sf.encoder(smiles)
        return selfies_str
    except Exception:
        return None

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert MOSES train/test SMILES splits to SELFIES CSV files."
    )
    parser.add_argument("--input", default=INPUT_CSV, help="MOSES dataset_v1.csv path.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 16))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    log.info("Starting MOSES dataset SELFIES preprocessing (Full Train / Full Test for Val) ...")
    
    # Read the dataset
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"MOSES dataset not found at {args.input}")
        
    df = pd.read_csv(args.input)
    log.info(f"Loaded dataset containing {len(df)} rows.")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Filter splits
    df_train = df[df["SPLIT"] == "train"].copy()
    df_val = df[df["SPLIT"] == "test"].copy()
    
    log.info(f"Splits prepared - Train: {len(df_train)}, Val (Full Test): {len(df_val)}")
    
    # Parallel conversion
    num_workers = args.workers
    log.info(f"Converting SMILES to SELFIES using {num_workers} parallel workers ...")
    
    # Convert train split
    with Pool(num_workers) as pool:
        df_train["selfies"] = pool.map(convert_smiles_to_selfies, df_train["SMILES"])
    df_train = df_train.dropna(subset=["selfies"])
    log.info(f"Train conversion complete. Remaining valid rows: {len(df_train)}")
    
    # Convert val split
    with Pool(num_workers) as pool:
        df_val["selfies"] = pool.map(convert_smiles_to_selfies, df_val["SMILES"])
    df_val = df_val.dropna(subset=["selfies"])
    log.info(f"Validation conversion complete. Remaining valid rows: {len(df_val)}")
    
    # Save CSV files
    train_out_path = os.path.join(args.output_dir, "train_selfies.csv")
    val_out_path = os.path.join(args.output_dir, "val_selfies.csv")
    
    df_train[["selfies"]].to_csv(train_out_path, index=False)
    df_val[["selfies"]].to_csv(val_out_path, index=False)
    
    log.info(f"Saved processed train split → {train_out_path}")
    log.info(f"Saved processed validation split → {val_out_path}")
    log.info("Preprocessing successfully complete!")

if __name__ == "__main__":
    main()
