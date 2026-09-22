import argparse
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def normalize_labels(input_csv, smiles_col, label_col, output_csv, method='standardize'):
    """
    Normalize or standardize labels.
    
    Args:
        input_csv: Path to input CSV file
        smiles_col: Name of SMILES column
        label_col: Name of label column to normalize/standardize
        output_csv: Path to save processed CSV file
        method: 'normalize' for min-max scaling [0,1], 'standardize' for z-score (default)
    """
    print(f"Loading dataset from {input_csv}")
    df = pd.read_csv(input_csv)
    
    # Check required columns
    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column '{smiles_col}' not found in dataset")
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset")
    
    print(f"Dataset shape: {df.shape}")
    print(f"Original label statistics:")
    print(f"  Mean: {df[label_col].mean():.4f}")
    print(f"  Std: {df[label_col].std():.4f}")
    print(f"  Range: [{df[label_col].min():.4f}, {df[label_col].max():.4f}]")
    
    # Remove rows with missing values in label column
    original_len = len(df)
    df = df.dropna(subset=[label_col])
    if len(df) < original_len:
        print(f"Removed {original_len - len(df)} rows with missing labels")
    
    # Apply transformation based on method
    if method == 'normalize':
        # Min-max scaling to [0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        df['y_normalized_minmax'] = scaler.fit_transform(df[[label_col]])
        new_col = 'y_normalized_minmax'
        print(f"\nApplied min-max normalization:")
        print(f"  Range: [{df[new_col].min():.4f}, {df[new_col].max():.4f}]")
    elif method == 'standardize':
        # Z-score standardization (mean=0, std=1)
        scaler = StandardScaler()
        df['y_normalized_zscore'] = scaler.fit_transform(df[[label_col]])
        new_col = 'y_normalized_zscore'
        print(f"\nApplied z-score standardization:")
        print(f"  Mean: {df[new_col].mean():.4f}")
        print(f"  Std: {df[new_col].std():.4f}")
        print(f"  Range: [{df[new_col].min():.4f}, {df[new_col].max():.4f}]")
    else:
        raise ValueError(f"Unknown method: {method}. Use 'normalize' or 'standardize'")
    
    # Save to output file
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    print(f"\nProcessed dataset saved to {output_csv}")
    print(f"New column '{new_col}' added")
    
    return output_csv, new_col


def main():
    parser = argparse.ArgumentParser(
        description="Normalize labels using min-max scaling"
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input CSV file'
    )
    
    parser.add_argument(
        '--smiles-col',
        type=str,
        required=True,
        help='Name of SMILES column'
    )
    
    parser.add_argument(
        '--label-col',
        type=str,
        required=True,
        help='Name of label column to normalize'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save processed CSV file'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        choices=['normalize', 'standardize'],
        default='standardize',
        help='Transformation method: normalize (min-max [0,1]) or standardize (z-score) (default: standardize)'
    )
    
    args = parser.parse_args()
    
    normalize_labels(
        input_csv=args.input,
        smiles_col=args.smiles_col,
        label_col=args.label_col,
        output_csv=args.output,
        method=args.method
    )


if __name__ == "__main__":
    main()
