import functools
import logging
import math
import os
import random
from pathlib import Path

import click
import numpy as np
import pandas as pd
import selfies as sf
import torch
import torch.nn.functional as F
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw
from tqdm import tqdm

DATA_ROOT = Path(os.getenv("DATA_ROOT", "data_root")).resolve()

DEFAULT_CKPT_PATH = DATA_ROOT / "last.ckpt"
DEFAULT_VOCAB_PATH = DATA_ROOT / "vocab.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(ckpt_path, vocab_file):
    logging.warning(
        "Loading a PyTorch checkpoint can execute code. Only load trusted files: %s",
        ckpt_path,
    )
    from bmfm_sm_gen.trainers.lm_vae import VAELightningModule

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lm = VAELightningModule.load_from_checkpoint(
        ckpt_path, vocab_file=vocab_file, map_location=device,
        tokenizer="bmfm_sm_gen.data.base_tokenizer.SELFIESTokenizer"
    )
    model = lm.model
    lm.to(device)
    model.eval()
    logging.info(f"Model loaded on device: {device}")
    return model, lm


def smiles_to_token_indices(
    lm, smiles: str, token_to_idx: dict, max_len: int
) -> torch.Tensor:
    selfies_str = sf.encoder(smiles)
    tokens = [lm.token_to_idx["<sos>"]]
    selfies_tokens = sf.split_selfies(selfies_str)
    for token in selfies_tokens:
        idx = token_to_idx.get(token, token_to_idx["<unk>"])
        tokens.append(idx)
    tokens.append(token_to_idx["<eos>"])
    if len(tokens) < max_len:
        tokens += [lm.token_to_idx["<pad>"]] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    return torch.tensor(tokens, dtype=torch.long)


def decode_latent_vectors_batch(
    lm, model, z: torch.Tensor, temperature: float = 1.0, top_k: int | None = None
) -> list[str | None]:
    """
    Decode a batch of latent vectors into SMILES strings.
    
    Args:
        lm: Lightning module with tokenizer
        model: VAE model
        z: Latent vectors of shape (batch_size, latent_dim)
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
    
    Returns:
        List of SMILES strings (None for failed decodings)
    """
    device = z.device
    batch_size = z.size(0)
    
    with torch.no_grad():
        # Initialize generated_tokens with <sos> token for all samples
        generated_tokens = torch.full(
            (batch_size, 1), lm.sos_token, dtype=torch.long, device=device
        )
        
        # Track which sequences have finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(model.max_len - 1):
            tgt = generated_tokens  # (batch_size, seq_len)
            
            tgt_emb = model.decoder.embedding(tgt) * math.sqrt(model.decoder.d_model)
            tgt_emb = model.decoder.pos_decoder(tgt_emb)
            tgt_emb = tgt_emb.transpose(0, 1)  # (seq_len, batch_size, d_model)
            
            tgt_mask = model.decoder.generate_square_subsequent_mask(tgt_emb.size(0)).to(device)
            tgt_key_padding_mask = tgt == lm.padding_idx  # (batch_size, seq_len)
            
            z_hidden = model.latent_to_hidden(z)  # (batch_size, d_model)
            memory = z_hidden.unsqueeze(1).repeat(1, model.max_len, 1).transpose(0, 1)
            
            output_logits = model.decoder.transformer_decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )
            
            output_logits = output_logits.transpose(0, 1)  # (batch_size, seq_len, d_model)
            output_logits = model.decoder.output_layer(output_logits[:, -1, :])
            logits = output_logits / temperature
            
            if top_k is not None:
                values, indices = torch.topk(logits, top_k)
                logits = torch.full_like(logits, -float("Inf"))
                logits.scatter_(dim=-1, index=indices, src=values)
            
            probabilities = F.softmax(logits, dim=-1)
            predicted = torch.multinomial(probabilities, num_samples=1)
            
            generated_tokens = torch.cat((generated_tokens, predicted), dim=1)
            
            # Update finished status
            finished = finished | (predicted.squeeze(-1) == lm.eos_token)
            
            # Early stopping if all sequences finished
            if finished.all():
                break
        
        # Remove the initial <sos> token
        generated_tokens = generated_tokens[:, 1:]
        
        # Decode each sequence
        smiles_list = []
        for token_seq in generated_tokens:
            tokens = []
            for idx in token_seq:
                idx = idx.item()
                if idx == lm.eos_token:
                    break
                if idx != lm.padding_idx:
                    tokens.append(lm.idx_to_token.get(idx, ""))
            selfies_str = "".join(tokens)
            
            try:
                smiles = sf.decoder(selfies_str)
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    smiles_list.append(smiles)
                else:
                    smiles_list.append(None)
            except Exception:
                smiles_list.append(None)
        
        return smiles_list


def decode_latent_vector(
    lm, model, z: torch.Tensor, temperature: float = 1.0, top_k: int = None
) -> str:
    """
    Decode a single latent vector into a SMILES string.
    This is a wrapper around decode_latent_vectors_batch for backward compatibility.
    """
    result = decode_latent_vectors_batch(lm, model, z, temperature, top_k)
    return result[0] if result else None


def calculate_similarity(smiles1: str, smiles2: str) -> float:
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return 0.0
    fp1 = Chem.RDKFingerprint(mol1)
    fp2 = Chem.RDKFingerprint(mol2)
    similarity = DataStructs.FingerprintSimilarity(fp1, fp2)
    return similarity


def generate_from_multiple_targets_batch(
    model,
    lm,
    target_smiles_list: list[str],
    num_samples_per_target: int = 1,
    std_dev: float = 1.0,
    sampling_scheme: str = "around_mu",
    temperature: float = 1.0,
    top_k: int | None = None,
    include_errors: bool = False,
) -> list[tuple]:
    """
    Generate molecules from multiple target SMILES in batch.
    
    Args:
        target_smiles_list: List of target SMILES strings
        num_samples_per_target: Number of molecules to generate per target
        
    Returns:
        List of tuples: (target_smiles, generated_smiles, similarity)
    """
    device = next(model.parameters()).device
    results = []
    
    with torch.no_grad():
        # Encode all targets at once
        target_tokens_list = []
        valid_targets = []
        valid_target_indices = []
        attempts = [[None] * num_samples_per_target for _ in target_smiles_list]
        
        for target_idx, target_smiles in enumerate(target_smiles_list):
            try:
                tokens = smiles_to_token_indices(lm, target_smiles, lm.token_to_idx, model.max_len)
                target_tokens_list.append(tokens[:-1])  # Remove last token
                valid_targets.append(target_smiles)
                valid_target_indices.append(target_idx)
            except Exception as e:
                logging.warning(f"Failed to tokenize {target_smiles}: {e}")
                for attempt_idx in range(num_samples_per_target):
                    attempts[target_idx][attempt_idx] = (
                        target_smiles, None, None, f"tokenization failed: {e}"
                    )
                continue
        
        if not valid_targets:
            results = [attempt for target in attempts for attempt in target]
            return results if include_errors else []
        
        # Stack all target tokens into a batch
        target_tokens_batch = torch.stack(target_tokens_list).to(device)  # (N, L-1)
        
        # Encode all targets in one forward pass
        mu_batch, logvar_batch = model.encode(target_tokens_batch)  # (N, D)
        
        latent_dim = mu_batch.size(-1)
        std_batch = torch.exp(0.5 * logvar_batch)  # (N, D)
        
        # Generate latent vectors for ALL targets at once
        n_targets = len(valid_targets)
        
        if sampling_scheme == "around_mu":
            # For each target, generate num_samples_per_target latent vectors
            # Expand mu and std to match the number of samples per target
            mu_expanded = mu_batch.repeat_interleave(num_samples_per_target, dim=0)  # (N*num_samples, D)
            std_expanded = std_batch.repeat_interleave(num_samples_per_target, dim=0)  # (N*num_samples, D)
            
            eps = torch.randn(n_targets * num_samples_per_target, latent_dim, device=device)
            z_all = mu_expanded + eps * std_expanded * std_dev  # (N*num_samples, D)
        elif sampling_scheme == "prior":
            z_all = torch.randn(n_targets * num_samples_per_target, latent_dim, device=device) * std_dev
        else:
            raise ValueError(f"Unknown sampling_scheme: {sampling_scheme}")
        
        # Decode ALL molecules in one batch (parallel decoding!)
        smiles_all = decode_latent_vectors_batch(
            lm=lm,
            model=model,
            z=z_all,  # (N*num_samples, D)
            temperature=temperature,
            top_k=top_k,
        )
        
        # Map generated SMILES back to their target SMILES and calculate similarities
        for i, (target_idx, target_smiles) in enumerate(
            zip(valid_target_indices, valid_targets)
        ):
            start_idx = i * num_samples_per_target
            end_idx = start_idx + num_samples_per_target
            
            target_smiles_batch = smiles_all[start_idx:end_idx]
            
            for attempt_idx, smiles in enumerate(target_smiles_batch):
                if smiles:
                    sim = calculate_similarity(target_smiles, smiles)
                    attempts[target_idx][attempt_idx] = (
                        target_smiles, smiles, sim, None
                    )
                else:
                    attempts[target_idx][attempt_idx] = (
                        target_smiles, None, None, "decoding produced an invalid molecule"
                    )
    
    results = [attempt for target in attempts for attempt in target]
    if include_errors:
        return results
    return [(target, smiles, similarity) for target, smiles, similarity, _ in results if smiles]


def generate_similar_molecules(
    model,
    lm,
    target_smiles: str,
    num_samples: int,
    std_dev: float = 1.0,
    sampling_scheme: str = "around_mu",  # 'around_mu' | 'prior'
    temperature: float = 1.0,
    top_k: int | None = None,
    batch_size: int = 256,
) -> list[tuple[str, float]]:
    device = next(model.parameters()).device
    results: list[tuple[str, float]] = []

    with torch.no_grad():
        target_tokens = (
            smiles_to_token_indices(lm, target_smiles, lm.token_to_idx, model.max_len)
            .unsqueeze(0)[:, :-1]
            .to(device)
        )  # shape (1, L-1)
        mu, logvar = model.encode(target_tokens)  # shapes (1, D)

        latent_dim = mu.size(-1)
        std = torch.exp(0.5 * logvar)  # (1, D)

        n_batches = math.ceil(num_samples / batch_size)
        iterator = range(n_batches)
        iterator = tqdm(iterator, total=n_batches, desc="Generating", unit="batch")

        for b in iterator:
            this_bs = min(batch_size, num_samples - b * batch_size)

            if sampling_scheme == "around_mu":
                eps = torch.randn(this_bs, latent_dim, device=device)
                z = mu + eps * std * std_dev  # (B, D)
            elif sampling_scheme == "prior":
                z = torch.randn(this_bs, latent_dim, device=device) * std_dev
            else:
                raise ValueError(f"Unknown sampling_scheme: {sampling_scheme}")

            # Batch decode all latent vectors at once
            smiles_batch = decode_latent_vectors_batch(
                lm=lm,
                model=model,
                z=z,  # (batch_size, latent_dim)
                temperature=temperature,
                top_k=top_k,
            )
            
            # Calculate similarities for valid SMILES
            for smiles in smiles_batch:
                if smiles:
                    sim = calculate_similarity(target_smiles, smiles)
                    results.append((smiles, sim))

    return results


def generate_unconstrained_molecules(
    lm, model, num_samples: int, temperature: float = 1.0, top_k: int | None = None,
    batch_size: int = 256
) -> list[str | None]:
    """
    Sample `num_samples` latent vectors from the VAE prior N(0, I)
    and decode them in batches for efficiency.
    """
    latent_dim = model.latent_dim  # attribute already present in `TransformerVAE`
    device = next(model.parameters()).device
    all_smiles = []
    
    with torch.no_grad():
        n_batches = math.ceil(num_samples / batch_size)
        for b in range(n_batches):
            this_bs = min(batch_size, num_samples - b * batch_size)
            z = torch.randn(this_bs, latent_dim, device=device)
            
            # Batch decode
            smiles_batch = decode_latent_vectors_batch(
                lm=lm,
                model=model,
                z=z,
                temperature=temperature,
                top_k=top_k,
            )
            all_smiles.extend(smiles_batch)
    
    return all_smiles


def visualize_molecules(
    target_smiles: str, molecules: list[tuple[str, float]], num_to_show: int = 10
):
    target_mol = Chem.MolFromSmiles(target_smiles) if target_smiles else None
    if target_smiles and target_mol is None:
        logging.info(f"Invalid target SMILES: {target_smiles}")
        return None
    molecules = [item for item in molecules if item[0]]
    molecules = sorted(molecules, key=lambda x: -x[1])[:num_to_show]
    mols = [Chem.MolFromSmiles(smi) for smi, _ in molecules]
    mols_legends = [f"Sim: {sim:.2f}" for _, sim in molecules]
    valid_mols = []
    valid_legends = []
    for mol, legend in zip(mols, mols_legends):
        if mol is not None:
            valid_mols.append(mol)
            valid_legends.append(legend)
    if target_mol is not None:
        valid_mols.insert(0, target_mol)
        valid_legends.insert(0, "Target")
    if not valid_mols:
        logging.error("No valid molecules to display.")
        return None
    img = Draw.MolsToGridImage(
        valid_mols, molsPerRow=5, subImgSize=(200, 200), legends=valid_legends
    )
    return img


@click.group(no_args_is_help=True)
def main():
    """CLI and Webapp for Molecular Similarity Search."""


def common_options(func):
    @click.option(
        "--seed",
        type=int,
        default=42,
        show_default=True,
        help="Random seed for Python, NumPy, and PyTorch.",
    )
    @click.option(
        "--ckpt-path",
        type=click.Path(exists=True),
        default=str(DEFAULT_CKPT_PATH),
        help="Path to the model checkpoint.",
    )
    @click.option(
        "--vocab-file",
        type=click.Path(exists=True),
        default=str(DEFAULT_VOCAB_PATH),
        help="Path to the vocabulary file.",
    )
    @click.option(
        "--num-samples",
        type=int,
        default=50,
        help="Number of samples to generate.",
    )
    @click.option(
        "--std-dev",
        type=float,
        default=1.0,
        help="Standard deviation for sampling.",
    )
    @click.option(
        "--sampling-scheme",
        type=click.Choice(["around_mu", "prior"]),
        default="around_mu",
        help="Sampling scheme.",
    )
    @click.option(
        "--temperature",
        type=float,
        default=1.1,
        help="Temperature for decoding.",
    )
    @click.option(
        "--top-k",
        type=int,
        default=5,
        help="Top K tokens to consider during decoding.",
    )
    @click.option(
        "--unconstrained/--conditioned",
        default=False,
        help="Generate molecules without using a target SMILES.",
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


@main.command()
@common_options
@click.option(
    "--target-smiles",
    type=str,
    required=False,
    default=None,
    help="Target SMILES string (required unless --unconstrained is used).",
)
@click.option(
    "--output-csv",
    type=click.Path(),
    default=None,
    help="If set, save results to this CSV and skip visualising the image.",
)
def cli(
    ckpt_path,
    vocab_file,
    target_smiles,
    num_samples,
    std_dev,
    sampling_scheme,
    temperature,
    top_k,
    unconstrained,
    seed,
    output_csv=None,
):
    if not unconstrained and not target_smiles:
        raise click.BadParameter(
            "--target-smiles must be provided unless --unconstrained is specified."
        )
    set_seed(seed)
    model, lm = load_model(ckpt_path=ckpt_path, vocab_file=vocab_file)
    if unconstrained:
        smiles_only = generate_unconstrained_molecules(
            lm, model, num_samples=num_samples, temperature=temperature, top_k=top_k
        )
        for smi in smiles_only:
            logging.info(f"SMILES: {smi}")
        img = visualize_molecules(
            "", [(smi, 0.0) for smi in smiles_only], num_to_show=20
        )
    else:
        similar_molecules = generate_similar_molecules(
            model,
            lm,
            target_smiles,
            num_samples,
            std_dev=std_dev,
            sampling_scheme=sampling_scheme,
            temperature=temperature,
            top_k=top_k,
        )
        for smi, sim in sorted(similar_molecules, key=lambda x: -x[1]):
            logging.info(f"SMILES: {smi}, Similarity: {sim:.2f}")

    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        if unconstrained:
            df = pd.DataFrame(smiles_only, columns=["smiles"])
        else:
            df = pd.DataFrame(similar_molecules, columns=["smiles", "similarity"])
        df.to_csv(output_csv, index=False)
        logging.info(f"Results saved to {output_csv}")
        return

    if unconstrained:
        img = visualize_molecules("", [(s, 0.0) for s in smiles_only], num_to_show=20)
    else:
        img = visualize_molecules(target_smiles, similar_molecules, num_to_show=20)

    if img:
        img.show()
    else:
        logging.error("No valid molecules to display.")


@main.command()
@common_options
@click.option(
    "--input-csv",
    type=click.Path(exists=True),
    required=True,
    help="Path to the input CSV file containing 'target_smiles'.",
)
@click.option(
    "--output-csv",
    type=click.Path(),
    required=True,
    help="Path to the output CSV file to save the results.",
)
@click.option(
    "--smiles-col",
    type=str,
    default='target_smiles',
    help="Column name containing SMILES strings.",
)
@click.option(
    "--cluster-sample",
    is_flag=True,
    default=False,
    help="Enable cluster-based sampling to maintain cluster size distribution.",
)
@click.option(
    "--cluster-col",
    type=str,
    default='cluster_index',
    help="Column name containing cluster indices (used with --cluster-sample).",
)
@click.option(
    "--max-molecules",
    type=int,
    default=10000,
    help="Target total number of molecules to generate (used with --cluster-sample).",
)
def cli_bulk(
    ckpt_path,
    vocab_file,
    input_csv,
    output_csv,
    smiles_col,
    num_samples,
    std_dev,
    sampling_scheme,
    temperature,
    top_k,
    seed,
    unconstrained=False,
    cluster_sample=False,
    cluster_col='cluster_index',
    max_molecules=10000,
):
    """Run the CLI in bulk mode, processing multiple target SMILES."""
    set_seed(seed)
    model, lm = load_model(ckpt_path=ckpt_path, vocab_file=vocab_file)

    input_data = pd.read_csv(input_csv)
    if smiles_col not in input_data.columns:
        raise ValueError(f"The input CSV must contain a '{smiles_col}' column.")

    results = []
    
    # Cluster-based sampling
    if cluster_sample:
        if cluster_col not in input_data.columns:
            raise ValueError(f"Cluster sampling requires '{cluster_col}' column in input CSV.")
        
        logging.info(f"Cluster-based sampling enabled with max_molecules={max_molecules}")
        
        # Group by cluster
        cluster_groups = input_data.groupby(cluster_col)
        cluster_sizes = cluster_groups.size()
        total_molecules = len(input_data)
        
        logging.info(f"Found {len(cluster_sizes)} clusters with {total_molecules} total molecules")
        
        # Calculate sampling distribution based on cluster sizes
        cluster_proportions = cluster_sizes / total_molecules
        # Use floor to avoid rounding issues, then distribute remainder
        cluster_sample_counts = (cluster_proportions * max_molecules).astype(int)
        
        # Distribute remainder to largest clusters
        remainder = max_molecules - cluster_sample_counts.sum()
        if remainder > 0:
            # Sort clusters by size (descending) and add 1 to each until remainder is 0
            sorted_clusters = cluster_sizes.sort_values(ascending=False).index
            for i in range(remainder):
                cluster_id = sorted_clusters[i % len(sorted_clusters)]
                cluster_sample_counts[cluster_id] += 1
        elif remainder < 0:
            # This shouldn't happen with floor, but handle it just in case
            logging.warning(f"Negative remainder {remainder}, adjusting largest cluster")
            largest_cluster = cluster_sample_counts.idxmax()
            cluster_sample_counts[largest_cluster] += remainder
            # Ensure no negative counts
            cluster_sample_counts = cluster_sample_counts.clip(lower=0)
        
        logging.info("Cluster sampling distribution:")
        for cluster_id, count in cluster_sample_counts.items():
            logging.info(f"  Cluster {cluster_id}: {count} samples ({cluster_sizes[cluster_id]} molecules, {cluster_proportions[cluster_id]*100:.1f}%)")
        
        # Sample from each cluster
        sampled_rows = []
        for cluster_id, n_samples_cluster in cluster_sample_counts.items():
            if n_samples_cluster > 0:
                cluster_data = cluster_groups.get_group(cluster_id)
                # Sample with replacement if needed
                replace = n_samples_cluster > len(cluster_data)
                sampled = cluster_data.sample(n=n_samples_cluster, replace=replace, random_state=seed)
                sampled_rows.append(sampled)
        
        input_data = pd.concat(sampled_rows, ignore_index=True)
        logging.info(f"Sampled {len(input_data)} molecules for generation")
        
        # Override num_samples to 1 per target when cluster sampling
        num_samples_per_target = 1
    else:
        num_samples_per_target = num_samples

    # Process target SMILES in batches for efficiency
    target_batch_size = 256  # Process 256 targets at once
    n_target_batches = math.ceil(len(input_data) / target_batch_size)
    
    logging.info(f"Processing {len(input_data)} targets in {n_target_batches} batches of up to {target_batch_size}")
    
    for batch_idx in tqdm(range(n_target_batches), desc="Processing target batches"):
        start_idx = batch_idx * target_batch_size
        end_idx = min((batch_idx + 1) * target_batch_size, len(input_data))
        batch_data = input_data.iloc[start_idx:end_idx]
        
        # Extract target SMILES for this batch
        target_smiles_list = batch_data[smiles_col].tolist()
        
        try:
            # Generate molecules for all targets in this batch
            batch_results = generate_from_multiple_targets_batch(
                model=model,
                lm=lm,
                target_smiles_list=target_smiles_list,
                num_samples_per_target=num_samples_per_target,
                std_dev=std_dev,
                sampling_scheme=sampling_scheme,
                temperature=temperature,
                top_k=top_k,
                include_errors=True,
            )
            
            expected_attempts = len(batch_data) * num_samples_per_target
            if len(batch_results) < expected_attempts:
                batch_results = list(batch_results) + [
                    (None, None, None, "generation returned no result")
                ] * (expected_attempts - len(batch_results))

            # Results are ordered by input row and then by per-target attempt.
            for result_idx, result in enumerate(batch_results[:expected_attempts]):
                input_offset = result_idx // num_samples_per_target
                source_row = batch_data.iloc[input_offset]
                target_smiles = source_row[smiles_col]
                if len(result) == 4:
                    _, generated_smiles, similarity, error = result
                else:
                    _, generated_smiles, similarity = result
                    error = None if generated_smiles else "generation failed"
                result_row = {
                    "attempt_id": len(results),
                    "target_smiles": target_smiles,
                    "generated_smiles": generated_smiles,
                    "similarity": similarity,
                    "valid": bool(generated_smiles),
                    "error": error,
                }
                
                # Add cluster info if available
                if cluster_sample and cluster_col in batch_data.columns:
                    result_row["target_cluster"] = source_row[cluster_col]
                
                results.append(result_row)
        
        except Exception as e:
            logging.error(f"Error processing batch {batch_idx}: {e}")
            # Preserve every requested attempt even when an entire batch fails.
            for source_offset, target_smiles in enumerate(target_smiles_list):
                for _ in range(num_samples_per_target):
                    result_row = {
                        "attempt_id": len(results),
                        "target_smiles": target_smiles,
                        "generated_smiles": None,
                        "similarity": None,
                        "valid": False,
                        "error": str(e),
                    }
                    if cluster_sample and cluster_col in batch_data.columns:
                        result_row["target_cluster"] = batch_data.iloc[source_offset][cluster_col]
                    results.append(result_row)

    # Save results
    output_data = pd.DataFrame(results)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    output_data.to_csv(output_csv, index=False)
    logging.info(f"Results saved to {output_csv}")
    valid_count = int(output_data["valid"].sum()) if not output_data.empty else 0
    logging.info(f"Generation summary: attempted={len(output_data)}, valid={valid_count}")


@main.command()
@common_options
@click.option(
    "--target-smiles",
    type=str,
    default="CC1=NC2=C(C(NC3=NC(C)=CS3)=N1)CCN(C(=O)CCC4=CC=CC=C4)C2",
    help="Target SMILES string.",
)
def webapp(
    ckpt_path,
    vocab_file,
    target_smiles,
    num_samples,
    std_dev,
    sampling_scheme,
    temperature,
    top_k,
    seed,
    unconstrained=False,
):
    import streamlit as st

    set_seed(seed)

    st.title("BMFM/SM Molecular Similarity Search")
    st.sidebar.header("Input Parameters")
    ckpt_path = st.sidebar.text_input("Model Checkpoint Path", ckpt_path)
    vocab_file = st.sidebar.text_input("Vocabulary File Path", vocab_file)
    target_smiles = st.sidebar.text_input("Target SMILES", target_smiles)
    num_samples = st.sidebar.number_input(
        "Number of Samples", min_value=1, max_value=1000, value=num_samples
    )
    std_dev = st.sidebar.number_input(
        "Standard Deviation", min_value=0.1, max_value=10.0, value=std_dev
    )
    sampling_scheme = st.sidebar.selectbox(
        "Sampling Scheme",
        ["around_mu", "prior"],
        index=0 if sampling_scheme == "around_mu" else 1,
    )
    temperature = st.sidebar.slider(
        "Temperature", min_value=0.1, max_value=5.0, value=temperature
    )
    top_k = st.sidebar.number_input("Top K", min_value=1, max_value=100, value=top_k)
    unconstrained = st.sidebar.checkbox("Unconstrained (ignore target)", value=False)

    if st.sidebar.button("Generate Molecules"):
        st.write("Loading model and generating molecules...")
        try:
            model, lm = load_model(ckpt_path=ckpt_path, vocab_file=vocab_file)
            if unconstrained:
                smiles_only = generate_unconstrained_molecules(
                    lm,
                    model,
                    num_samples=num_samples,
                    temperature=temperature,
                    top_k=top_k,
                )
                st.subheader("Unconstrained Samples")
                img = visualize_molecules("", [(s, 0.0) for s in smiles_only])
                if img:
                    st.image(img)
                st.table(pd.DataFrame(smiles_only, columns=["SMILES"]))
            else:
                similar_molecules = generate_similar_molecules(
                    model,
                    lm,
                    target_smiles,
                    num_samples,
                    std_dev=std_dev,
                    sampling_scheme=sampling_scheme,
                    temperature=temperature,
                    top_k=top_k,
                )
                if similar_molecules:
                    st.success("Molecules generated successfully!")
                    st.subheader("Generated Molecules")
                    img = visualize_molecules(
                        target_smiles, similar_molecules, num_to_show=20
                    )
                    if img:
                        st.image(img)
                    else:
                        st.warning("No valid molecules to display.")
                    df = pd.DataFrame(
                        similar_molecules, columns=["SMILES", "Similarity"]
                    )
                    df = df.sort_values(by="Similarity", ascending=False)
                    st.table(df.reset_index(drop=True))
                else:
                    st.warning("No molecules were generated.")
        except Exception as e:
            st.error(f"An error occurred: {e}")


@main.command()
def run_webapp():
    import sys

    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", __file__]
    stcli._main_run(__file__, None)


if __name__ == "__main__":
    main()
