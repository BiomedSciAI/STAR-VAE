import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs


def smiles_to_morgan_fingerprint(smiles, radius=2, n_bits=2048):
    """
    Convert SMILES to Morgan fingerprint.
    
    Args:
        smiles: SMILES string
        radius: Radius for Morgan fingerprint (default: 2)
        n_bits: Number of bits in fingerprint (default: 2048)
    
    Returns:
        Morgan fingerprint as numpy array, or None if conversion fails
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        # Convert to numpy array
        arr = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception as e:
        return None


def generate_fingerprints(smiles_list, verbose=True):
    """
    Generate Morgan fingerprints for a list of SMILES.
    
    Args:
        smiles_list: List of SMILES strings
        verbose: Whether to print progress
    
    Returns:
        fingerprints: numpy array of fingerprints
        valid_indices: list of valid indices
    """
    fingerprints = []
    valid_indices = []
    failed_count = 0
    
    for idx, smiles in enumerate(smiles_list):
        if verbose and idx % 10000 == 0 and idx > 0:
            print(f"  Processed {idx}/{len(smiles_list)} molecules...")
        
        fp = smiles_to_morgan_fingerprint(smiles)
        if fp is not None:
            fingerprints.append(fp)
            valid_indices.append(idx)
        else:
            failed_count += 1
            if failed_count <= 5:  # Print first 5 failures
                print(f"  Warning: Failed to generate fingerprint for SMILES at index {idx}: {smiles}")
    
    if verbose:
        print(f"\nFingerprint generation summary:")
        print(f"  Successful: {len(fingerprints)}")
        print(f"  Failed: {failed_count}")
        print(f"  Total processed: {len(smiles_list)}")
    
    if len(fingerprints) == 0:
        raise ValueError(f"No valid fingerprints generated. All {len(smiles_list)} SMILES failed.")
    
    fingerprints = np.vstack(fingerprints)
    return fingerprints, valid_indices


def predict_with_regressor(
    input_csv,
    model_path,
    smiles_col,
    output_csv,
    prediction_col='predicted_y_normalized_zscore'
):
    """
    Make predictions using a trained XGBoost regressor.
    
    Args:
        input_csv: Path to input CSV file with SMILES
        model_path: Path to trained XGBoost model (JSON format)
        smiles_col: Name of SMILES column
        output_csv: Path to save output CSV with predictions
        prediction_col: Name of column for predictions
    """
    print(f"Loading dataset from {input_csv}")
    df = pd.read_csv(input_csv)
    
    # Check SMILES column
    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column '{smiles_col}' not found in dataset")
    
    print(f"Dataset shape: {df.shape}")
    
    # Generate fingerprints
    print("\nGenerating Morgan fingerprints (radius=2, fpSize=2048)...")
    fingerprints, valid_indices = generate_fingerprints(df[smiles_col].tolist())
    
    print(f"Successfully generated {len(fingerprints)} fingerprints")
    
    # Load model
    print(f"\nLoading model from {model_path}")
    model = xgb.Booster()
    model.load_model(model_path)
    
    # Create DMatrix for prediction
    dmatrix = xgb.DMatrix(fingerprints)
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(dmatrix)
    
    print(f"Predictions summary:")
    print(f"  Mean: {predictions.mean():.4f}")
    print(f"  Std: {predictions.std():.4f}")
    print(f"  Range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    
    # Create output dataframe with predictions
    # Initialize prediction column with NaN for all rows
    df[prediction_col] = np.nan
    
    # Assign predictions to valid indices
    df.loc[valid_indices, prediction_col] = predictions
    
    # Save to output file
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    print(f"\nPredictions saved to {output_csv}")
    print(f"  Total rows: {len(df)}")
    print(f"  Rows with predictions: {len(valid_indices)}")
    print(f"  Rows with failed fingerprints: {len(df) - len(valid_indices)}")
    
    return output_csv


def main():
    parser = argparse.ArgumentParser(
        description="Make predictions using trained XGBoost regressor on molecular fingerprints"
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input CSV file with SMILES'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained XGBoost model (JSON format)'
    )
    
    parser.add_argument(
        '--smiles-col',
        type=str,
        default='smiles',
        help='Name of SMILES column in dataset'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save output CSV with predictions'
    )
    
    parser.add_argument(
        '--prediction-col',
        type=str,
        default='predicted_y_normalized_zscore',
        help='Name of column for predictions (default: predicted_y_normalized_zscore)'
    )
    
    args = parser.parse_args()
    
    predict_with_regressor(
        input_csv=args.input,
        model_path=args.model,
        smiles_col=args.smiles_col,
        output_csv=args.output,
        prediction_col=args.prediction_col
    )


if __name__ == "__main__":
    main()
