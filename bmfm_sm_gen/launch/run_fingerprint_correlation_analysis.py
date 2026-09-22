import os
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
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
        print(f"Error converting SMILES {smiles}: {e}")
        return None


def calculate_tanimoto_distance(fp1, fp2):
    """
    Calculate Tanimoto distance between two fingerprints.
    Tanimoto distance = 1 - Tanimoto similarity
    
    Args:
        fp1: First fingerprint (numpy array)
        fp2: Second fingerprint (numpy array)
    
    Returns:
        Tanimoto distance (float)
    """
    # Calculate intersection and union
    intersection = np.sum(fp1 & fp2)
    union = np.sum(fp1 | fp2)
    
    # Avoid division by zero
    if union == 0:
        return 1.0
    
    # Tanimoto similarity
    similarity = intersection / union
    
    # Tanimoto distance
    distance = 1.0 - similarity
    
    return distance


def generate_fingerprints(smiles_list):
    """
    Generate Morgan fingerprints for a list of SMILES.
    
    Args:
        smiles_list: List of SMILES strings
    
    Returns:
        fingerprints: numpy array of fingerprints
        valid_indices: list of valid indices
    """
    fingerprints = []
    valid_indices = []
    failed_count = 0
    
    for idx, smiles in enumerate(tqdm(smiles_list, desc="Generating fingerprints")):
        fp = smiles_to_morgan_fingerprint(smiles)
        if fp is not None:
            fingerprints.append(fp)
            valid_indices.append(idx)
        else:
            failed_count += 1
            if failed_count <= 5:  # Print first 5 failures
                print(f"Failed to generate fingerprint for SMILES at index {idx}: {smiles}")
    
    print(f"\nFingerprint generation summary:")
    print(f"  Successful: {len(fingerprints)}")
    print(f"  Failed: {failed_count}")
    print(f"  Total processed: {len(smiles_list)}")
    
    if len(fingerprints) == 0:
        raise ValueError(f"No valid fingerprints generated. All {len(smiles_list)} SMILES failed.")
    
    if len(fingerprints) < 2:
        raise ValueError(f"Only {len(fingerprints)} valid fingerprint(s) generated. Need at least 2 samples for correlation analysis.")
    
    fingerprints = np.vstack(fingerprints)
    return fingerprints, valid_indices


def calculate_pairwise_distances(fingerprints, labels, chunk_size=10000):
    """
    Calculate pairwise Tanimoto distances for fingerprints and label distances.
    
    For large datasets, this uses chunked processing to avoid memory issues.
    
    Returns:
        dict with keys: 'tanimoto', 'label'
    """
    n_samples = len(fingerprints)
    
    # For small datasets, compute all pairs
    if n_samples <= chunk_size:
        print(f"Computing all pairwise distances for {n_samples} samples...")
        tanimoto_distances = []
        label_distances = []
        
        for i in tqdm(range(n_samples), desc="Computing pairwise distances"):
            for j in range(i + 1, n_samples):
                # Calculate Tanimoto distance
                tanimoto_dist = calculate_tanimoto_distance(fingerprints[i], fingerprints[j])
                tanimoto_distances.append(tanimoto_dist)
                
                # Calculate label distance
                label_dist = abs(labels[i] - labels[j])
                label_distances.append(label_dist)
        
        return {
            'tanimoto': np.array(tanimoto_distances),
            'label': np.array(label_distances)
        }
    
    # For large datasets, use chunked processing with sampling
    print(f"Large dataset detected ({n_samples} samples). Using chunked processing with sampling...")
    
    # Sample pairs for correlation (to avoid memory issues)
    max_pairs = min(int(np.sqrt(n_samples) * 1000), n_samples * (n_samples - 1) // 2)
    print(f"Sampling {max_pairs:,} pairs from {n_samples * (n_samples - 1) // 2:,} total pairs")
    
    # Generate random pair indices
    np.random.seed(42)
    all_pairs = n_samples * (n_samples - 1) // 2
    sampled_indices = np.random.choice(all_pairs, size=max_pairs, replace=False)
    sampled_indices = np.sort(sampled_indices)
    
    # Convert condensed indices to (i, j) pairs
    def condensed_to_square(k, n):
        """Convert condensed index to (i, j) pair."""
        i = int(n - 2 - np.floor(np.sqrt(-8*k + 4*n*(n-1) - 7) / 2.0 - 0.5))
        j = int(k + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2)
        return i, j
    
    tanimoto_distances = []
    label_distances = []
    
    print("Computing sampled pairwise distances...")
    for idx, k in enumerate(sampled_indices):
        if idx % 10000 == 0:
            print(f"  Processed {idx:,} / {max_pairs:,} pairs")
        
        i, j = condensed_to_square(k, n_samples)
        
        # Calculate Tanimoto distance
        tanimoto_dist = calculate_tanimoto_distance(fingerprints[i], fingerprints[j])
        tanimoto_distances.append(tanimoto_dist)
        
        # Calculate label distance
        label_dist = abs(labels[i] - labels[j])
        label_distances.append(label_dist)
    
    print(f"Completed computing {len(tanimoto_distances):,} pairwise distances")
    
    return {
        'tanimoto': np.array(tanimoto_distances),
        'label': np.array(label_distances)
    }


def calculate_correlation(fingerprint_distances, label_distances):
    """Calculate Pearson and Spearman correlation between distances."""
    # Remove any NaN or Inf values
    valid_mask = np.isfinite(fingerprint_distances) & np.isfinite(label_distances)
    fingerprint_distances = fingerprint_distances[valid_mask]
    label_distances = label_distances[valid_mask]
    
    if len(fingerprint_distances) < 2:
        return None, None, None, None
    
    # Calculate correlations
    pearson_corr, pearson_pval = pearsonr(fingerprint_distances, label_distances)
    spearman_corr, spearman_pval = spearmanr(fingerprint_distances, label_distances)
    
    return pearson_corr, pearson_pval, spearman_corr, spearman_pval


def save_pairwise_distances_csv(distance_dict, output_path):
    """Save pairwise distances to CSV file."""
    df = pd.DataFrame({
        'tanimoto_distance': distance_dict['tanimoto'],
        'label_distance': distance_dict['label']
    })
    df.to_csv(output_path, index=False)
    print(f"Pairwise distances saved to {output_path}")


def save_correlation_json(results, output_path):
    """Save correlation results to JSON file."""
    # Convert numpy types to Python types for JSON serialization
    json_results = {}
    for key, value in results.items():
        if isinstance(value, (np.integer, np.floating)):
            json_results[key] = float(value)
        else:
            json_results[key] = value
    
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Correlation results saved to {output_path}")


def create_correlation_plot(distance_dict, pearson_corr, spearman_corr, output_path):
    try:
        import altair as alt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Fingerprint correlation charts require the optional 'altair' package."
        ) from exc
    """
    Create correlation scatter plot with Altair and save as PDF.
    
    Args:
        distance_dict: Dict with keys 'tanimoto', 'label'
        pearson_corr: Pearson correlation coefficient
        spearman_corr: Spearman correlation coefficient
        output_path: Path to save PDF
    """
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Tanimoto Distance': distance_dict['tanimoto'],
        'Label Distance': distance_dict['label']
    })
    
    # Sample data if too large (Altair has a 5000 row limit by default)
    if len(df) > 5000:
        df = df.sample(n=5000, random_state=42)
    
    # Create scatter plot
    scatter = alt.Chart(df).mark_circle(size=15, opacity=0.4).encode(
        x=alt.X('Tanimoto Distance:Q', title='Tanimoto Distance'),
        y=alt.Y('Label Distance:Q', title='Label Distance'),
        tooltip=['Tanimoto Distance:Q', 'Label Distance:Q']
    ).properties(
        width=500,
        height=400,
        title='Fingerprint-based Correlation Analysis: Tanimoto Distance vs Label Distance'
    )
    
    # Add regression line
    regression = scatter.transform_regression(
        'Tanimoto Distance', 'Label Distance'
    ).mark_line(color='red', size=2)
    
    # Create text annotations for correlation coefficients
    x_max = float(df['Tanimoto Distance'].max())
    y_max = float(df['Label Distance'].max())
    text_data = pd.DataFrame({
        'x': [x_max * 0.05],
        'y': [y_max * 0.95],
        'text': [f'Pearson: {pearson_corr:.4f}\nSpearman: {spearman_corr:.4f}']
    })
    
    text = alt.Chart(text_data).mark_text(
        align='left',
        baseline='top',
        fontSize=12,
        fontWeight='bold',
        dx=5,
        dy=-5
    ).encode(
        x='x:Q',
        y='y:Q',
        text='text:N'
    )
    
    # Combine scatter, regression, and text
    combined_chart = (scatter + regression + text).configure_axis(
        labelFontSize=11,
        titleFontSize=13
    ).configure_title(
        fontSize=15,
        anchor='start'
    )
    
    # Save as PDF
    combined_chart.save(output_path)
    print(f"Correlation plot saved to {output_path}")


def run_fingerprint_correlation_analysis(
    dataset_path,
    smiles_col='smiles',
    label_col='label',
    output_file=None,
    max_samples=None
):
    """
    Run correlation analysis between fingerprint-based distances and label distances.
    
    Args:
        dataset_path: Path to CSV file with smiles and label columns
        smiles_col: Name of SMILES column in dataset
        label_col: Name of label column in dataset
        output_file: Path to save results (optional)
        max_samples: Maximum number of samples to process (optional)
    """
    print(f"Loading dataset from {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # Check required columns
    if smiles_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Dataset must contain '{smiles_col}' and '{label_col}' columns")
    
    # Limit samples if specified
    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
        print(f"Sampled {max_samples} rows from dataset")
    
    # Remove rows with missing values
    df = df[[smiles_col, label_col]].dropna()
    print(f"Processing {len(df)} samples")
    
    # Generate fingerprints
    print("Generating Morgan fingerprints (radius=2, fpSize=2048)...")
    fingerprints, valid_indices = generate_fingerprints(df[smiles_col].tolist())
    
    # Filter labels to match valid fingerprints
    labels = df[label_col].iloc[valid_indices].values
    valid_smiles = df[smiles_col].iloc[valid_indices].values
    
    print(f"Successfully generated {len(fingerprints)} fingerprints")
    print(f"Fingerprint shape: {fingerprints.shape}")
    
    # Calculate pairwise distances
    print("Calculating pairwise distances...")
    distance_dict = calculate_pairwise_distances(fingerprints, labels)
    
    print(f"Number of pairwise distances: {len(distance_dict['tanimoto'])}")
    print(f"Tanimoto distance range: [{distance_dict['tanimoto'].min():.4f}, {distance_dict['tanimoto'].max():.4f}]")
    print(f"Label distance range: [{distance_dict['label'].min():.4f}, {distance_dict['label'].max():.4f}]")
    
    # Calculate correlations
    print("Calculating correlations...")
    pearson_corr, pearson_pval, spearman_corr, spearman_pval = calculate_correlation(
        distance_dict['tanimoto'], distance_dict['label']
    )
    
    # Print results
    print("\n" + "="*60)
    print("FINGERPRINT CORRELATION ANALYSIS RESULTS")
    print("="*60)
    print(f"Dataset: {dataset_path}")
    print(f"Fingerprint: Morgan (radius=2, fpSize=2048)")
    print(f"Distance metric: Tanimoto")
    print(f"Number of samples: {len(fingerprints)}")
    print(f"Number of pairwise comparisons: {len(distance_dict['tanimoto'])}")
    print("-"*60)
    print(f"Pearson correlation:  {pearson_corr:.6f} (p={pearson_pval:.6e})")
    print(f"Spearman correlation: {spearman_corr:.6f} (p={spearman_pval:.6e})")
    print("="*60)
    
    # Save results
    results = {
        'dataset_path': dataset_path,
        'fingerprint_type': 'Morgan',
        'fingerprint_radius': 2,
        'fingerprint_size': 2048,
        'distance_metric': 'Tanimoto',
        'num_samples': len(fingerprints),
        'num_pairwise_comparisons': len(distance_dict['tanimoto']),
        'pearson_correlation': pearson_corr,
        'pearson_pvalue': pearson_pval,
        'spearman_correlation': spearman_corr,
        'spearman_pvalue': spearman_pval,
        'tanimoto_distance_mean': float(distance_dict['tanimoto'].mean()),
        'tanimoto_distance_std': float(distance_dict['tanimoto'].std()),
        'label_distance_mean': float(distance_dict['label'].mean()),
        'label_distance_std': float(distance_dict['label'].std()),
    }
    
    # Generate output file paths
    if output_file:
        output_base = Path(output_file).stem
        output_dir = Path(output_file).parent
        
        # Save pairwise distances to CSV
        distances_csv = output_dir / f"{output_base}_pairwise_distances.csv"
        save_pairwise_distances_csv(distance_dict, str(distances_csv))
        
        # Save correlation results to JSON
        correlation_json = output_dir / f"{output_base}_correlation.json"
        save_correlation_json(results, str(correlation_json))
        
        # Create and save correlation plot
        plot_pdf = output_dir / f"{output_base}_correlation_plot.pdf"
        create_correlation_plot(distance_dict, pearson_corr, spearman_corr, str(plot_pdf))
        
        # Also save summary CSV for backward compatibility
        results_df = pd.DataFrame([results])
        results_df.to_csv(output_file, index=False)
        print(f"\nSummary results saved to {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Calculate correlation between fingerprint-based distances and label distances"
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to CSV file with smiles and label columns'
    )
    
    parser.add_argument(
        '--smiles-col',
        type=str,
        default='smiles',
        help='Name of SMILES column in dataset'
    )
    
    parser.add_argument(
        '--label-col',
        type=str,
        default='label',
        help='Name of label column in dataset'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save results CSV file'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to process'
    )
    
    args = parser.parse_args()
    
    run_fingerprint_correlation_analysis(
        dataset_path=args.dataset,
        smiles_col=args.smiles_col,
        label_col=args.label_col,
        output_file=args.output,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()
