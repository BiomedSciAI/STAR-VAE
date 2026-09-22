import json
import logging
import math
from collections import Counter

import numpy as np
import pandas as pd
import selfies as sf
from sklearn.feature_selection import mutual_info_regression
import torch
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import entropy, spearmanr, wasserstein_distance

_morgan = GetMorganGenerator(radius=2, fpSize=2048)


def run_conditional_generation_evaluation_compute_prop(
    model,
    tokenizer,
    label_name: str,
    device: torch.device,
    num_labels: int = 10,
    samples_per_label: int = 100,
):
    from bmfm_sm_gen.data.ds_mol import compute_prop

    model.eval()
    label_range = torch.linspace(1.0, num_labels, steps=45).to(device)

    all_preds = []
    all_targets = []
    labeled_groups = []

    for y_val in label_range:
        labels = torch.full((samples_per_label, 1), y_val.item(), device=device)
        z = torch.randn(samples_per_label, model.latent_dim).to(device)

        with torch.no_grad():
            smiles_list = generate_molecules_with_labels(model, tokenizer, z, labels)
            logging.info(
                f"Generated {len(smiles_list)} SMILES for label {y_val.item()}"
            )

        valid_entries = []
        for smi in smiles_list:
            if smi is None:
                continue
            try:
                score = compute_prop(
                    input_str=smi, input_type="smiles", names=[label_name]
                )[0]
                if score is not None:
                    all_preds.append(score)
                    all_targets.append(y_val.item())
                    valid_entries.append((smi, y_val.item(), score))
            except Exception:
                continue

        labeled_groups.append(valid_entries)

    results = {}
    if all_preds:
        results["conditional_label_spearman"] = spearmanr(all_preds, all_targets)[0]
        try:
            results["conditional_label_mutual_info"] = mutual_info_regression(
                X=torch.tensor(all_preds).reshape(-1, 1).numpy(),
                y=torch.tensor(all_targets).numpy(),
            )[0]
        except Exception:
            results["conditional_label_mutual_info"] = None
    else:
        results["conditional_label_spearman"] = None
        results["conditional_label_mutual_info"] = None

    return results, labeled_groups


def load_dataset(csv_path):
    return pd.read_csv(csv_path)


def extract_arrays(df, fp_type="morgan_fp"):
    # Compute Morgan FPs on-the-fly if the pre-computed column is missing
    if fp_type in df.columns:
        fps = np.stack(df[fp_type].apply(json.loads).apply(np.array))
    else:
        smiles_col = "smiles" if "smiles" in df.columns else df.columns[0]
        fps_list = []
        for smi in df[smiles_col]:
            m = Chem.MolFromSmiles(str(smi)) if smi else None
            if m is not None:
                fps_list.append(_morgan.GetFingerprintAsNumPy(m).astype(np.float32))
            else:
                fps_list.append(np.zeros(2048, dtype=np.float32))
        fps = np.stack(fps_list)

    # Build label dict with fallback: if a norm variant is missing, use whichever exists
    norm_variants = ["minmax", "zscore", "robust", "percentile"]
    labels = {}
    for norm in norm_variants:
        col = f"y_normalized_{norm}"
        if col in df.columns:
            labels[norm] = df[col].to_numpy()
        else:
            # Fall back to first available normalised column (or raw label)
            for fallback in norm_variants:
                fc = f"y_normalized_{fallback}"
                if fc in df.columns:
                    labels[norm] = df[fc].to_numpy()
                    break
            else:
                # Last resort: use any numeric column that looks like a label
                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                labels[norm] = df[num_cols[-1]].to_numpy() if num_cols else np.zeros(len(df))
    return fps, labels


def compute_label_stats(y):
    return {
        "min": float(y.min()),
        "max": float(y.max()),
        "mean": float(y.mean()),
        "std": float(y.std()),
    }


def compute_bin_edges(y: np.ndarray, num_bins: int, mode: str = "uniform"):
    y = np.asarray(y)
    if mode == "uniform":
        edges = np.linspace(float(y.min()), float(y.max()), num_bins + 1)
    elif mode == "equal_count":
        edges = np.quantile(y, np.linspace(0.0, 1.0, num_bins + 1))
    else:
        raise ValueError(f"unknown mode {mode}")
    return edges


def generate_molecules_with_labels(model, tokenizer, z, labels):
    logging.info(
        f"Generate_molecules_with_labels: start label={labels[0].item():.3f}, n={len(labels)}"
    )
    model.eval()
    device = z.device
    batch_size = z.size(0)
    max_len = model.max_len
    generated_tokens = torch.full(
        (batch_size, 1), tokenizer.bos_token_id, dtype=torch.long, device=device
    )
    generated_smiles = []
    
    # Get lora configuration from model if available, default to "both" for backward compatibility
    lora = getattr(model, 'lora', 'both')
    
    with torch.no_grad():
        for _ in range(max_len - 1):
            tgt = generated_tokens
            tgt_emb = model.decoder.embedding(tgt) * math.sqrt(model.decoder.d_model)
            tgt_emb = model.decoder.pos_decoder(tgt_emb).transpose(0, 1)
            tgt_mask = model.decoder.generate_square_subsequent_mask(
                tgt_emb.size(0)
            ).to(device)
            tgt_key_padding_mask = tgt == tokenizer.pad_token_id
            
            # Prepare memory based on lora configuration
            if lora in ["both", "decoder_only"] and model.modulation is not None:
                # Label-conditioned memory
                zc = model.modulation(z, labels)
                z_hidden = model.latent_to_hidden(zc)
            else:
                # No label conditioning in decoder
                z_hidden = model.latent_to_hidden(z)
            
            memory = z_hidden.unsqueeze(1).repeat(1, max_len, 1).transpose(0, 1)
            
            out = model.decoder.transformer_decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            ).transpose(0, 1)
            logits = model.decoder.output_layer(out[:, -1, :])
            pred = logits.argmax(dim=-1, keepdim=True)
            generated_tokens = torch.cat((generated_tokens, pred), dim=1)
            if (pred == tokenizer.eos_token_id).all():
                break
        for tok_seq in generated_tokens[:, 1:]:
            toks = []
            for idx in tok_seq:
                i = idx.item()
                if i == tokenizer.eos_token_id:
                    break
                if i != tokenizer.pad_token_id:
                    toks.append(tokenizer.convert_ids_to_tokens(i))
            selfies_str = "".join(toks)
            try:
                smi = sf.decoder(selfies_str)
                mol = Chem.MolFromSmiles(smi)
                generated_smiles.append(smi if mol else None)
            except Exception:
                generated_smiles.append(None)
    return generated_smiles


def build_generation_groups_within_bins(
    model,
    tokenizer,
    device,
    bins: np.ndarray,
    samples_per_bin: int,
):
    groups, counts, label_values = [], [], []
    for i in range(len(bins) - 1):
        low, high = float(bins[i]), float(bins[i + 1])
        y_vals = torch.empty(samples_per_bin, 1, device=device).uniform_(low, high)
        z = torch.randn(samples_per_bin, model.latent_dim, device=device)
        with torch.no_grad():
            smi_list = generate_molecules_with_labels(model, tokenizer, z, y_vals)
        valid = [s for s in smi_list if s]
        groups.append(valid)
        counts.append(len(valid))
        label_values.append(0.5 * (low + high))
    return groups, counts, np.asarray(label_values, dtype=float)


def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))


def hist_pdf(x, bins=50, rng=None, eps=1e-12):
    h, edges = np.histogram(x, bins=bins, range=rng, density=False)
    h = (h + eps) / (h.sum() + eps * len(h))
    return h, edges


def match_and_score_distributions(val_fps, gen_fps, bins=50):
    if len(val_fps) == 0 or len(gen_fps) == 0:
        return {"mmd": None, "ws": None, "jsd": None}
    mmd = float(np.sum((val_fps.mean(axis=0) - gen_fps.mean(axis=0)) ** 2))
    ws = wasserstein_distance(val_fps.flatten(), gen_fps.flatten())
    p, e = hist_pdf(val_fps.flatten(), bins=bins)
    q, _ = hist_pdf(gen_fps.flatten(), bins=bins, rng=(e[0], e[-1]))
    jsd = js_divergence(p, q)
    return {"mmd": mmd, "ws": ws, "jsd": jsd}


def top_scaffolds_by_bins(df, label_col, edges: np.ndarray, top_k=10):
    if "scaffold" not in df.columns:
        smiles_col = "smiles" if "smiles" in df.columns else df.columns[0]
        df = df.copy()
        df["scaffold"] = df[smiles_col].apply(_compute_scaffold)
    scaffold_sets = []
    for i in range(len(edges) - 1):
        low, high = edges[i], edges[i + 1]
        mask = (df[label_col] >= low) & (df[label_col] < high)
        sub = df.loc[mask]
        scafs = sub["scaffold"].dropna().tolist()
        top = [s for s, _ in Counter(scafs).most_common(top_k)]
        scaffold_sets.append(set(top))
    return scaffold_sets


def _compute_scaffold(smi: str):
    """Return Murcko scaffold SMILES for a SMILES string, or None."""
    m = Chem.MolFromSmiles(smi) if smi else None
    if m is None:
        return None
    scaf = MurckoScaffold.GetScaffoldForMol(m)
    return Chem.MolToSmiles(scaf, isomericSmiles=False)


def scaffold_overlap_and_distribution_stats(
    val_df,
    gen_groups_smiles,
    edges: np.ndarray,
    label_values: np.ndarray,
    label_col: str,
    top_k: int = 10,
):
    # Compute scaffold column on-the-fly if not present
    if "scaffold" not in val_df.columns:
        smiles_col = "smiles" if "smiles" in val_df.columns else val_df.columns[0]
        val_df = val_df.copy()
        val_df["scaffold"] = val_df[smiles_col].apply(_compute_scaffold)

    results = []
    for i, gen_smis in enumerate(gen_groups_smiles):
        low, high = edges[i], edges[i + 1]
        mask = (val_df[label_col] >= low) & (val_df[label_col] < high)
        val_sub = val_df.loc[mask]
        val_scafs = val_sub["scaffold"].dropna().tolist()
        val_counts = Counter(val_scafs)
        gen_counts = Counter()
        for s in gen_smis:
            m = Chem.MolFromSmiles(s)
            if m:
                scaf = MurckoScaffold.GetScaffoldForMol(m)
                scaf_smiles = Chem.MolToSmiles(scaf, isomericSmiles=False)
                gen_counts[scaf_smiles] += 1
        val_top = [s for s, _ in val_counts.most_common(top_k)]
        gen_top = [s for s, _ in gen_counts.most_common(top_k)]
        val_set = set(val_top)
        gen_set = set(gen_top)
        overlap = len(val_set & gen_set)
        jaccard = overlap / len(val_set | gen_set) if len(val_set | gen_set) else 0.0
        recovery = overlap / len(val_set) if val_set else 0.0
        all_scaffolds = list(set(val_counts.keys()) | set(gen_counts.keys()))
        v = np.array([val_counts.get(s, 0) for s in all_scaffolds], dtype=float)
        g = np.array([gen_counts.get(s, 0) for s in all_scaffolds], dtype=float)
        if v.sum() > 0 and g.sum() > 0:
            v /= v.sum()
            g /= g.sum()
            m = 0.5 * (v + g)
            jsd = 0.5 * (entropy(v, m) + entropy(g, m))
            ws = wasserstein_distance(v, g)
        else:
            jsd = None
            ws = None
        results.append(
            {
                "label_value": float(label_values[i]),
                "bin_low": low,
                "bin_high": high,
                "overlap": overlap,
                "jaccard": jaccard,
                "recovery_rate": recovery,
                "jsd_scaffold_dist": jsd,
                "ws_scaffold_dist": ws,
                "n_val_scaffolds": len(val_counts),
                "n_gen_scaffolds": len(gen_counts),
            }
        )
    return pd.DataFrame(results)


def run_validation_distribution_match(
    model,
    tokenizer,
    device,
    val_df,
    val_fps,
    label_points,
    label_col,
    samples_per_bin=100,
    bins=50,
    binning_mode: str = "equal_count",
):
    num_bins = len(label_points)
    edges = compute_bin_edges(
        val_df[label_col].to_numpy(), num_bins=num_bins, mode=binning_mode
    )
    gen_groups, counts, label_values = build_generation_groups_within_bins(
        model, tokenizer, device, edges, samples_per_bin=samples_per_bin
    )
    gen_fps_all = []
    for group in gen_groups:
        fps = []
        for smi in group:
            m = Chem.MolFromSmiles(smi)
            if m:
                arr = _morgan.GetFingerprintAsNumPy(m).astype(np.float32)
                fps.append(arr)
        if fps:
            gen_fps_all.append(np.stack(fps))
        else:
            gen_fps_all.append(np.zeros((1, 2048), dtype=np.float32))
    results = []
    for i, gen_fps in enumerate(gen_fps_all):
        k = len(gen_fps)
        if k == 0:
            stats = {"mmd": None, "ws": None, "jsd": None}
        else:
            val_subset = val_fps[np.random.choice(len(val_fps), k, replace=False)]
            stats = match_and_score_distributions(val_subset, gen_fps, bins=bins)
        stats["label_value"] = float(label_values[i])
        stats["valid_count"] = len(gen_groups[i])
        results.append(stats)
    scaf_df = scaffold_overlap_and_distribution_stats(
        val_df=val_df,
        gen_groups_smiles=gen_groups,
        edges=edges,
        label_values=label_values,
        label_col=label_col,
        top_k=10,
    )
    agg = {
        "mmd_mean": float(np.nanmean([r["mmd"] for r in results])),
        "ws_mean": float(np.nanmean([r["ws"] for r in results])),
        "jsd_mean": float(np.nanmean([r["jsd"] for r in results])),
        "mean_valid": float(np.mean([r["valid_count"] for r in results])),
        "scaffold_mean_jaccard": float(scaf_df["jaccard"].mean()),
        "scaffold_mean_recovery": float(scaf_df["recovery_rate"].mean()),
        "scaffold_dist_jsd_mean": float(scaf_df["jsd_scaffold_dist"].mean()),
        "scaffold_dist_ws_mean": float(scaf_df["ws_scaffold_dist"].mean()),
    }
    return {"aggregate": agg, "per_label": results, "scaffold_stats": scaf_df}


def prepare_validation_info(val_csv_path, label_key="zscore", num_bins=10):
    val_df = load_dataset(val_csv_path)
    val_fps, val_labels = extract_arrays(val_df)
    if label_key not in val_labels:
        available = ", ".join(sorted(val_labels))
        raise ValueError(
            f"Unknown label normalization {label_key!r}; available labels: {available}"
        )
    y = val_labels[label_key]
    edges_q = compute_bin_edges(y, num_bins=num_bins, mode="equal_count")
    edges_u = compute_bin_edges(y, num_bins=num_bins, mode="uniform")
    points_q = 0.5 * (edges_q[:-1] + edges_q[1:])
    points_u = 0.5 * (edges_u[:-1] + edges_u[1:])
    return val_df, val_fps, points_q, points_u
