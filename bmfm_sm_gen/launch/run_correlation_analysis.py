import os
import argparse
import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import selfies as sf
import altair as alt

# Monkey patch prepare_validation_info BEFORE importing CVAELightningModule
from bmfm_sm_gen.utils import conditional_evaluation
_original_prepare_validation_info = conditional_evaluation.prepare_validation_info

def _patched_prepare_validation_info(val_csv_path, label_key="zscore", num_bins=10):
    """Patched version that handles None val_csv_path for inference-only use."""
    if val_csv_path is None:
        # Return dummy values for inference-only use
        dummy_df = pd.DataFrame()
        dummy_fps = np.array([])
        dummy_points = np.array([0.5] * num_bins)
        return dummy_df, dummy_fps, dummy_points, dummy_points
    return _original_prepare_validation_info(val_csv_path, label_key, num_bins)

# Apply the patch globally
conditional_evaluation.prepare_validation_info = _patched_prepare_validation_info

from bmfm_sm_gen.models.vae import TransformerVAE
from bmfm_sm_gen.trainers.lm_vae import VAELightningModule
from bmfm_sm_gen.trainers.lm_cvae import CVAELightningModule


def load_vae_model(ckpt_path, vocab_file):
    """Load pre-trained VAE model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lm = VAELightningModule.load_from_checkpoint(
        ckpt_path, vocab_file=vocab_file, map_location=torch.device(device),
        tokenizer="bmfm_sm_gen.data.base_tokenizer.SELFIESTokenizer"
    )
    model = TransformerVAE(**lm.model_params)
    checkpoint = torch.load(ckpt_path, map_location=torch.device(device))
    weights = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(weights)
    model.to(device)  # Move model to device
    model.eval()
    return model, lm, device


def load_cvae_model(ckpt_path, vocab_file, pretrained_path):
    """Load fine-tuned CVAE model (uses global monkey patch for validation)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model (prepare_validation_info is already patched globally)
    # Use strict=False to ignore unexpected keys like scorer_model weights
    lm = CVAELightningModule.load_from_checkpoint(
        ckpt_path,
        vocab_file=vocab_file,
        pretrained_path=pretrained_path,
        map_location=device,
        strict=False,  # Ignore unexpected keys in checkpoint
    )
    lm.model.eval()
    lm.model.to(device)
    return lm.model, lm, device


def smiles_to_token_indices(lm, smiles: str, max_len: int) -> torch.Tensor:
    """Convert SMILES to token indices."""
    try:
        selfies_str = sf.encoder(smiles)
        tokens = [lm.token_to_idx["<sos>"]]
        selfies_tokens = sf.split_selfies(selfies_str)
        for token in selfies_tokens:
            idx = lm.token_to_idx.get(token, lm.token_to_idx["<unk>"])
            tokens.append(idx)
        tokens.append(lm.token_to_idx["<eos>"])
        if len(tokens) < max_len:
            tokens += [lm.token_to_idx["<pad>"]] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]
        return torch.tensor(tokens, dtype=torch.long)
    except Exception as e:
        print(f"Error converting SMILES {smiles}: {e}")
        return None


def encode_smiles(model, lm, smiles_list, device, labels=None, use_labels=False):
    """
    Encode SMILES strings to latent embeddings.
    
    Args:
        model: VAE or CVAE model
        lm: Lightning module
        smiles_list: List of SMILES strings
        device: torch device
        labels: Optional list of labels (required if use_labels=True)
        use_labels: If True, use encode_with_labels for CVAE models
    
    Returns:
        embeddings: numpy array of embeddings
        valid_indices: list of valid indices
    """
    embeddings = []
    valid_indices = []
    failed_count = 0
    tokenize_failures = 0
    encode_failures = 0
    
    # Determine if we should use conditional encoding
    is_cvae = hasattr(model, 'lora') and hasattr(model, 'encode_with_labels')
    use_conditional = use_labels and is_cvae and labels is not None
    
    if use_conditional:
        print(f"Using conditional encoding (encode_with_labels) for CVAE model")
        if len(labels) != len(smiles_list):
            raise ValueError(f"Labels length ({len(labels)}) must match smiles_list length ({len(smiles_list)})")
    else:
        print(f"Using standard encoding (encode)")
    
    for idx, smiles in enumerate(tqdm(smiles_list, desc="Encoding SMILES")):
        try:
            # Convert SMILES to token indices
            tokens = smiles_to_token_indices(lm, smiles, model.max_len)
            if tokens is None:
                tokenize_failures += 1
                failed_count += 1
                if failed_count <= 5:  # Print first 5 failures
                    print(f"Failed to tokenize SMILES at index {idx}: {smiles}")
                continue
            
            # Prepare input (remove last token for encoding)
            tokens = tokens.unsqueeze(0)[:, :-1].to(device)
            
            # Encode to get mu and logvar
            with torch.no_grad():
                if use_conditional:
                    # Use encode_with_labels for CVAE
                    label_tensor = torch.tensor([[labels[idx]]], dtype=torch.float32).to(device)
                    result = model.encode_with_labels(tokens, label_tensor)
                else:
                    # Use standard encode
                    result = model.encode(tokens)
                
                if isinstance(result, tuple) and len(result) == 2:
                    mu, logvar = result
                else:
                    print(f"Unexpected encode output at index {idx}: {type(result)}")
                    encode_failures += 1
                    failed_count += 1
                    continue
            
            # Validate mu
            if mu is None or torch.isnan(mu).any() or torch.isinf(mu).any():
                print(f"Invalid mu at index {idx}: mu is None or contains NaN/Inf")
                encode_failures += 1
                failed_count += 1
                continue
            
            # Use mu as the embedding
            embeddings.append(mu.cpu().numpy())
            valid_indices.append(idx)
        except Exception as e:
            encode_failures += 1
            failed_count += 1
            if failed_count <= 5:  # Print first 5 failures
                print(f"Error encoding SMILES at index {idx} ({smiles}): {e}")
                import traceback
                traceback.print_exc()
            continue
    
    print(f"\nEncoding summary:")
    print(f"  Successful: {len(embeddings)}")
    print(f"  Tokenization failures: {tokenize_failures}")
    print(f"  Encoding failures: {encode_failures}")
    print(f"  Total processed: {len(smiles_list)}")
    
    if len(embeddings) == 0:
        raise ValueError(f"No valid embeddings generated. All {len(smiles_list)} SMILES failed to encode. Tokenization failures: {tokenize_failures}, Encoding failures: {encode_failures}")
    
    if len(embeddings) < 2:
        raise ValueError(f"Only {len(embeddings)} valid embedding(s) generated. Need at least 2 samples for correlation analysis.")
    
    embeddings = np.vstack(embeddings)
    return embeddings, valid_indices


def calculate_pairwise_distances(embeddings, labels, chunk_size=10000):
    """
    Calculate pairwise distances for embeddings and labels with memory-efficient chunking.
    Computes L1 (Manhattan), L2 (Euclidean), and Cosine distances for embeddings.
    
    For large datasets, this avoids creating the full pairwise distance matrix in memory.
    Instead, it processes in chunks and only keeps what's needed for correlation.
    
    Returns:
        dict with keys: 'l1', 'l2', 'cosine', 'label'
    """
    n_samples = len(embeddings)
    
    # For small datasets, use the fast scipy method
    if n_samples <= chunk_size:
        embedding_l1 = pdist(embeddings, metric='cityblock')  # Manhattan/L1
        embedding_l2 = pdist(embeddings, metric='euclidean')  # Euclidean/L2
        embedding_cosine = pdist(embeddings, metric='cosine')  # Cosine distance
        labels = np.array(labels).reshape(-1, 1)
        label_distances = pdist(labels, metric='cityblock')
        return {
            'l1': embedding_l1,
            'l2': embedding_l2,
            'cosine': embedding_cosine,
            'label': label_distances
        }
    
    # For large datasets, use chunked processing
    print(f"Large dataset detected ({n_samples} samples). Using chunked processing...")
    
    # Sample pairs for correlation (to avoid memory issues)
    # We'll sample sqrt(n_samples) * 1000 pairs randomly
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
    
    embedding_l1 = []
    embedding_l2 = []
    embedding_cosine = []
    label_distances = []
    
    print("Computing sampled pairwise distances...")
    for idx, k in enumerate(sampled_indices):
        if idx % 10000 == 0:
            print(f"  Processed {idx:,} / {max_pairs:,} pairs")
        
        i, j = condensed_to_square(k, n_samples)
        
        # Calculate embedding distances with different metrics
        diff = embeddings[i] - embeddings[j]
        
        # L1 (Manhattan) distance
        l1_dist = np.sum(np.abs(diff))
        embedding_l1.append(l1_dist)
        
        # L2 (Euclidean) distance
        l2_dist = np.linalg.norm(diff)
        embedding_l2.append(l2_dist)
        
        # Cosine distance (1 - cosine similarity)
        dot_product = np.dot(embeddings[i], embeddings[j])
        norm_i = np.linalg.norm(embeddings[i])
        norm_j = np.linalg.norm(embeddings[j])
        cosine_sim = dot_product / (norm_i * norm_j + 1e-8)
        cosine_dist = 1 - cosine_sim
        embedding_cosine.append(cosine_dist)
        
        # Calculate label distance
        label_dist = abs(labels[i] - labels[j])
        label_distances.append(label_dist)
    
    print(f"Completed computing {len(embedding_l1):,} pairwise distances")
    
    return {
        'l1': np.array(embedding_l1),
        'l2': np.array(embedding_l2),
        'cosine': np.array(embedding_cosine),
        'label': np.array(label_distances)
    }


def calculate_correlation(embedding_distances, label_distances):
    """Calculate Pearson and Spearman correlation between distances."""
    # Remove any NaN or Inf values
    valid_mask = np.isfinite(embedding_distances) & np.isfinite(label_distances)
    embedding_distances = embedding_distances[valid_mask]
    label_distances = label_distances[valid_mask]
    
    if len(embedding_distances) < 2:
        return None, None, None, None
    
    # Calculate correlations
    pearson_corr, pearson_pval = pearsonr(embedding_distances, label_distances)
    spearman_corr, spearman_pval = spearmanr(embedding_distances, label_distances)
    
    return pearson_corr, pearson_pval, spearman_corr, spearman_pval


def save_pairwise_distances_csv(distance_dict, output_path):
    """Save pairwise distances to CSV file with all distance metrics."""
    df = pd.DataFrame({
        'embedding_l1_distance': distance_dict['l1'],
        'embedding_l2_distance': distance_dict['l2'],
        'embedding_cosine_distance': distance_dict['cosine'],
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


def create_correlation_plot(distance_dict, correlations_dict, output_path):
    """
    Create multi-panel correlation scatter plots with Altair and save as PDF.
    
    Args:
        distance_dict: Dict with keys 'l1', 'l2', 'cosine', 'label'
        correlations_dict: Dict with correlation results for each metric
        output_path: Path to save PDF
    """
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'L1 Distance': distance_dict['l1'],
        'L2 Distance': distance_dict['l2'],
        'Cosine Distance': distance_dict['cosine'],
        'Label Distance': distance_dict['label']
    })
    
    # Sample data if too large (Altair has a 5000 row limit by default)
    if len(df) > 5000:
        df = df.sample(n=5000, random_state=42)
    
    # Create three subplots for L1, L2, and Cosine distances
    plots = []
    metrics = [
        ('L1 Distance', 'L1 (Manhattan)', 'l1'),
        ('L2 Distance', 'L2 (Euclidean)', 'l2'),
        ('Cosine Distance', 'Cosine', 'cosine')
    ]
    
    for col_name, title_name, metric_key in metrics:
        # Get correlation values for this metric
        corr_data = correlations_dict[metric_key]
        pearson_corr = corr_data['pearson_corr']
        spearman_corr = corr_data['spearman_corr']
        
        # Create scatter plot
        scatter = alt.Chart(df).mark_circle(size=15, opacity=0.4).encode(
            x=alt.X(f'{col_name}:Q', title=f'{title_name} Distance'),
            y=alt.Y('Label Distance:Q', title='Label Distance'),
            tooltip=[f'{col_name}:Q', 'Label Distance:Q']
        ).properties(
            width=300,
            height=300,
            title=f'{title_name} vs Label Distance'
        )
        
        # Add regression line
        regression = scatter.transform_regression(
            col_name, 'Label Distance'
        ).mark_line(color='red', size=2)
        
        # Create text annotations for correlation coefficients
        x_max = float(df[col_name].max())
        y_max = float(df['Label Distance'].max())
        text_data = pd.DataFrame({
            'x': [x_max * 0.05],
            'y': [y_max * 0.95],
            'text': [f'Pearson: {pearson_corr:.4f}\nSpearman: {spearman_corr:.4f}']
        })
        
        text = alt.Chart(text_data).mark_text(
            align='left',
            baseline='top',
            fontSize=11,
            fontWeight='bold',
            dx=5,
            dy=-5
        ).encode(
            x='x:Q',
            y='y:Q',
            text='text:N'
        )
        
        # Combine scatter, regression, and text
        plot = (scatter + regression + text)
        plots.append(plot)
    
    # Combine all three plots horizontally
    combined_chart = alt.hconcat(*plots).configure_axis(
        labelFontSize=10,
        titleFontSize=12
    ).configure_title(
        fontSize=14,
        anchor='start'
    ).properties(
        title='Correlation Analysis: Embedding Distances vs Label Distances'
    )
    
    # Save as PDF
    combined_chart.save(output_path)
    print(f"Multi-panel correlation plot saved to {output_path}")


def run_correlation_analysis(
    dataset_path,
    checkpoint_path,
    vocab_file,
    model_type='vae',
    pretrained_path=None,
    smiles_col='smiles',
    label_col='label',
    output_file=None,
    max_samples=None,
    use_label_encoding=None
):
    """
    Run correlation analysis between embedding distances and label distances.
    
    Args:
        dataset_path: Path to CSV file with 'smiles' and 'label' columns
        checkpoint_path: Path to model checkpoint
        vocab_file: Path to vocabulary JSON file
        model_type: 'vae' for pre-trained or 'cvae' for fine-tuned
        pretrained_path: Path to pre-trained checkpoint (required for cvae)
        smiles_col: Name of SMILES column in dataset
        label_col: Name of label column in dataset
        output_file: Path to save results (optional)
        max_samples: Maximum number of samples to process (optional)
        use_label_encoding: For CVAE models, controls encoding method:
            - 'encoder_only' or 'both': Use encode_with_labels (label-conditioned encoding)
            - 'decoder_only' or None: Use standard encode (no label conditioning)
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
    
    # Load model
    print(f"Loading {model_type.upper()} model from {checkpoint_path}")
    if model_type == 'vae':
        model, lm, device = load_vae_model(checkpoint_path, vocab_file)
    elif model_type == 'cvae':
        if pretrained_path is None:
            raise ValueError("pretrained_path is required for cvae model")
        model, lm, device = load_cvae_model(checkpoint_path, vocab_file, pretrained_path)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Determine if we should use label-conditioned encoding
    use_labels = False
    if model_type == 'cvae' and use_label_encoding in ['encoder_only', 'both']:
        use_labels = True
        print(f"Using label-conditioned encoding (use_label_encoding={use_label_encoding})")
    
    # Encode SMILES to embeddings
    print("Encoding SMILES to embeddings...")
    embeddings, valid_indices = encode_smiles(
        model, lm, df[smiles_col].tolist(), device,
        labels=df[label_col].tolist() if use_labels else None,
        use_labels=use_labels
    )
    
    # Filter labels to match valid embeddings
    labels = df[label_col].iloc[valid_indices].values
    valid_smiles = df[smiles_col].iloc[valid_indices].values
    
    print(f"Successfully encoded {len(embeddings)} molecules")
    print(f"Embedding shape: {embeddings.shape}")
    
    # Calculate pairwise distances (returns dict with l1, l2, cosine, label)
    print("Calculating pairwise distances...")
    distance_dict = calculate_pairwise_distances(embeddings, labels)
    
    print(f"Number of pairwise distances: {len(distance_dict['l1'])}")
    print(f"L1 distance range: [{distance_dict['l1'].min():.4f}, {distance_dict['l1'].max():.4f}]")
    print(f"L2 distance range: [{distance_dict['l2'].min():.4f}, {distance_dict['l2'].max():.4f}]")
    print(f"Cosine distance range: [{distance_dict['cosine'].min():.4f}, {distance_dict['cosine'].max():.4f}]")
    print(f"Label distance range: [{distance_dict['label'].min():.4f}, {distance_dict['label'].max():.4f}]")
    
    # Calculate correlations for each distance metric
    print("Calculating correlations...")
    correlations_dict = {}
    for metric in ['l1', 'l2', 'cosine']:
        pearson_corr, pearson_pval, spearman_corr, spearman_pval = calculate_correlation(
            distance_dict[metric], distance_dict['label']
        )
        correlations_dict[metric] = {
            'pearson_corr': pearson_corr,
            'pearson_pval': pearson_pval,
            'spearman_corr': spearman_corr,
            'spearman_pval': spearman_pval
        }
    
    # Print results
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS RESULTS")
    print("="*60)
    print(f"Dataset: {dataset_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Model Type: {model_type.upper()}")
    print(f"Number of samples: {len(embeddings)}")
    print(f"Number of pairwise comparisons: {len(distance_dict['l1'])}")
    print("-"*60)
    print("L1 (Manhattan) Distance:")
    print(f"  Pearson:  {correlations_dict['l1']['pearson_corr']:.6f} (p={correlations_dict['l1']['pearson_pval']:.6e})")
    print(f"  Spearman: {correlations_dict['l1']['spearman_corr']:.6f} (p={correlations_dict['l1']['spearman_pval']:.6e})")
    print("-"*60)
    print("L2 (Euclidean) Distance:")
    print(f"  Pearson:  {correlations_dict['l2']['pearson_corr']:.6f} (p={correlations_dict['l2']['pearson_pval']:.6e})")
    print(f"  Spearman: {correlations_dict['l2']['spearman_corr']:.6f} (p={correlations_dict['l2']['spearman_pval']:.6e})")
    print("-"*60)
    print("Cosine Distance:")
    print(f"  Pearson:  {correlations_dict['cosine']['pearson_corr']:.6f} (p={correlations_dict['cosine']['pearson_pval']:.6e})")
    print(f"  Spearman: {correlations_dict['cosine']['spearman_corr']:.6f} (p={correlations_dict['cosine']['spearman_pval']:.6e})")
    print("="*60)
    
    # Save results
    results = {
        'dataset_path': dataset_path,
        'checkpoint_path': checkpoint_path,
        'model_type': model_type,
        'num_samples': len(embeddings),
        'num_pairwise_comparisons': len(distance_dict['l1']),
        # L1 metrics
        'l1_pearson_correlation': correlations_dict['l1']['pearson_corr'],
        'l1_pearson_pvalue': correlations_dict['l1']['pearson_pval'],
        'l1_spearman_correlation': correlations_dict['l1']['spearman_corr'],
        'l1_spearman_pvalue': correlations_dict['l1']['spearman_pval'],
        'l1_distance_mean': float(distance_dict['l1'].mean()),
        'l1_distance_std': float(distance_dict['l1'].std()),
        # L2 metrics
        'l2_pearson_correlation': correlations_dict['l2']['pearson_corr'],
        'l2_pearson_pvalue': correlations_dict['l2']['pearson_pval'],
        'l2_spearman_correlation': correlations_dict['l2']['spearman_corr'],
        'l2_spearman_pvalue': correlations_dict['l2']['spearman_pval'],
        'l2_distance_mean': float(distance_dict['l2'].mean()),
        'l2_distance_std': float(distance_dict['l2'].std()),
        # Cosine metrics
        'cosine_pearson_correlation': correlations_dict['cosine']['pearson_corr'],
        'cosine_pearson_pvalue': correlations_dict['cosine']['pearson_pval'],
        'cosine_spearman_correlation': correlations_dict['cosine']['spearman_corr'],
        'cosine_spearman_pvalue': correlations_dict['cosine']['spearman_pval'],
        'cosine_distance_mean': float(distance_dict['cosine'].mean()),
        'cosine_distance_std': float(distance_dict['cosine'].std()),
        # Label metrics
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
        
        # Create and save multi-panel correlation plot
        plot_pdf = output_dir / f"{output_base}_correlation_plot.pdf"
        create_correlation_plot(distance_dict, correlations_dict, str(plot_pdf))
        
        # Also save summary CSV for backward compatibility
        results_df = pd.DataFrame([results])
        results_df.to_csv(output_file, index=False)
        print(f"\nSummary results saved to {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Calculate correlation between embedding distances and label distances"
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to CSV file with smiles and label columns'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    
    parser.add_argument(
        '--vocab',
        type=str,
        required=True,
        help='Path to vocabulary JSON file'
    )
    
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['vae', 'cvae'],
        default='vae',
        help='Model type: vae (pre-trained) or cvae (fine-tuned)'
    )
    
    parser.add_argument(
        '--pretrained',
        type=str,
        default=None,
        help='Path to pre-trained checkpoint (required for cvae)'
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
    
    parser.add_argument(
        '--use-label-encoding',
        type=str,
        choices=['encoder_only', 'decoder_only', 'both'],
        default=None,
        help='For CVAE models: encoder_only/both uses encode_with_labels (label-conditioned), decoder_only uses standard encode'
    )
    
    args = parser.parse_args()
    
    run_correlation_analysis(
        dataset_path=args.dataset,
        checkpoint_path=args.checkpoint,
        vocab_file=args.vocab,
        model_type=args.model_type,
        pretrained_path=args.pretrained,
        smiles_col=args.smiles_col,
        label_col=args.label_col,
        output_file=args.output,
        max_samples=args.max_samples,
        use_label_encoding=args.use_label_encoding
    )


if __name__ == "__main__":
    main()
