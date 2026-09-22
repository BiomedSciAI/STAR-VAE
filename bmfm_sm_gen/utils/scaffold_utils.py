import hashlib
import logging
from collections import defaultdict
from typing import Optional, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import selfies as sf
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


def group_by_bemis_murcko_scaffold(smiles_list):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    scaffold_dict = defaultdict(list)
    for s, mol in zip(smiles_list, mols):
        if mol is None:
            continue
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold_mol is None:
            continue
        scaffold_smiles = Chem.MolToSmiles(scaffold_mol, isomericSmiles=True)
        if not scaffold_smiles:
            continue
        scaffold_dict[scaffold_smiles].append(s)
    return scaffold_dict


def smiles_to_selfies(smiles_str: str) -> Optional[str]:
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles_str}")
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        selfies_str = sf.encoder(canonical_smiles)
        return selfies_str
    except Exception as e:
        logging.error(f"Error converting SMILES to SELFIES: {e}")
        return None


def add_selfies_column(
    df: pd.DataFrame,
    smiles_col: str = "std_smiles",
    selfies_col: str = "canonical_selfies",
    drop_failed: bool = True,
) -> pd.DataFrame:
    out = df.copy()
    out[selfies_col] = out[smiles_col].apply(smiles_to_selfies)
    if drop_failed:
        before = len(out)
        out = out[~out[selfies_col].isna()].copy()
        dropped = before - len(out)
        if dropped:
            logging.info(f"[add_selfies_column] Dropped {dropped} invalid SMILES.")
    return out


def murcko_from_smiles(smi: str) -> str | None:
    if not isinstance(smi, str) or not smi.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is None or scaf.GetNumAtoms() == 0:
            return None
        return Chem.MolToSmiles(scaf, isomericSmiles=True, canonical=True)
    except Exception:
        return None


def scaffold_id(scaffold_smiles: str | None) -> str | None:
    if not scaffold_smiles:
        return None
    return hashlib.md5(scaffold_smiles.encode("utf-8")).hexdigest()[:10]


def compute_scaffolds(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
    scaf_col: str = "Murcko Scaffold",
    scaf_id_col: str = "Murcko Scaffold ID",
    scaf_size_col: str = "Murcko Scaffold Cluster Size",
) -> pd.DataFrame:
    out = df.copy()
    out[scaf_col] = out[smiles_col].apply(murcko_from_smiles)
    out[scaf_id_col] = out[scaf_col].apply(scaffold_id)
    sizes = out[scaf_col].value_counts(dropna=False)
    out[scaf_size_col] = out[scaf_col].map(sizes)
    n_total = len(out)
    n_missing = out[scaf_col].isna().sum()
    n_unique = out[scaf_col].nunique(dropna=True)
    print(f"[scaffolds] rows={n_total:,} | missing={n_missing:,} | unique={n_unique:,}")
    return out


def scaffold_variation_stats(
    df: pd.DataFrame,
    value_col: str = "standard_value_nm_log10",
    scaf_col: str = "Murcko Scaffold",
) -> tuple[pd.DataFrame, dict]:
    vals = pd.to_numeric(df[value_col], errors="coerce")
    keep = vals.notna()
    X = vals[keep]
    G = df.loc[keep, scaf_col]
    per_scaf = (
        pd.DataFrame({"n": G.value_counts()})
        .merge(
            df.loc[keep, [scaf_col, value_col]]
            .groupby(scaf_col)[value_col]
            .agg(mean="mean", std="std", min="min", max="max"),
            left_index=True,
            right_index=True,
            how="left",
        )
        .rename_axis(scaf_col)
        .reset_index()
        .sort_values("n", ascending=False)
    )
    overall_mean = X.mean()
    total_var = ((X - overall_mean) ** 2).sum() / (len(X) - 1) if len(X) > 1 else 0.0
    groups = df.loc[keep, [scaf_col, value_col]].groupby(scaf_col)
    ss_within = 0.0
    dof_within = 0
    for _, sub in groups:
        xi = sub[value_col].to_numpy(float)
        if len(xi) <= 1:
            continue
        ss_within += np.sum((xi - xi.mean()) ** 2)
        dof_within += len(xi) - 1
    within_var = ss_within / dof_within if dof_within > 0 else 0.0
    ss_between = 0.0
    for scaf, sub in groups:
        ni = len(sub)
        if ni == 0:
            continue
        mu_i = sub[value_col].mean()
        ss_between += ni * (mu_i - overall_mean) ** 2
    dof_between = groups.ngroups - 1 if groups.ngroups > 1 else 0
    between_var = ss_between / dof_between if dof_between > 0 else 0.0
    r2 = (
        (ss_between / (ss_between + ss_within)) if (ss_between + ss_within) > 0 else 0.0
    )
    global_stats = {
        "N_rows": int(len(X)),
        "N_scaffolds": int(groups.ngroups),
        "overall_mean_label": float(overall_mean),
        "total_variance": float(total_var),
        "within_variance": float(within_var),
        "between_variance": float(between_var),
        "R2_scaffold_means": float(r2),
        "mean_within_std": (
            float(per_scaf["std"].dropna().mean()) if "std" in per_scaf else np.nan
        ),
        "median_within_std": (
            float(per_scaf["std"].dropna().median()) if "std" in per_scaf else np.nan
        ),
    }
    print("\n[scaffold variation]")
    print(
        f"Rows: {global_stats['N_rows']:,} | Scaffolds: {global_stats['N_scaffolds']:,}"
    )
    print(f"Overall mean: {global_stats['overall_mean_label']:.3f}")
    print(f"Variance total: {global_stats['total_variance']:.4f}")
    print(f"Variance within: {global_stats['within_variance']:.4f}")
    print(f"Variance between: {global_stats['between_variance']:.4f}")
    print(f"R^2 (by scaffold means): {global_stats['R2_scaffold_means']:.3f}")
    return per_scaf, global_stats


def update_scaffolds_and_summarize(
    df: pd.DataFrame,
    smiles_col: str = "std_smiles",
    value_col: str = "standard_value_nm_log10",
    scaf_col: str = "Murcko Scaffold",
    scaf_id_col: str = "Murcko Scaffold ID",
    scaf_size_col: str = "Murcko Scaffold Cluster Size",
):
    df_upd = compute_scaffolds(
        df,
        smiles_col=smiles_col,
        scaf_col=scaf_col,
        scaf_id_col=scaf_id_col,
        scaf_size_col=scaf_size_col,
    )
    per_scaf, global_stats = scaffold_variation_stats(
        df_upd, value_col=value_col, scaf_col=scaf_col
    )
    return df_upd, per_scaf, global_stats


def rank_scaffolds_label(
    df: pd.DataFrame,
    scaf_col: str = "Murcko Scaffold",
    y_col: str = "standard_value_nm_log10",
    min_n: int = 2,
    agg: str = "mean",  # new parameter: 'mean' or 'max'
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Rank scaffolds by aggregate property (mean or max).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns for scaffold and property.
    scaf_col : str
        Column containing scaffold identifiers.
    y_col : str
        Property/score column to aggregate.
    min_n : int
        Minimum number of molecules per scaffold to include.
    agg : str
        Aggregation function: 'mean' (default) or 'max'.

    Returns
    -------
    pd.DataFrame
        Aggregated scaffold statistics, sorted by chosen aggregate descending.
    """
    d = df[[scaf_col, y_col]].dropna().copy()

    # Group and aggregate scaffold statistics
    t = (
        d.groupby(scaf_col)[y_col]
        .agg(n="count", mean="mean", median="median", std="std", min="min", max="max")
        .reset_index()
    )

    # Keep scaffolds with enough members
    t = t[t["n"] >= min_n].copy()

    # Compute standard error and confidence interval on the mean
    t["se"] = t["std"] / np.sqrt(t["n"]).replace(0, np.nan)
    t["mean_ci_lo"] = t["mean"] - 1.96 * t["se"]
    t["mean_ci_hi"] = t["mean"] + 1.96 * t["se"]

    # Choose aggregation column
    if agg == "max":
        sort_col = "max"
    elif agg == "mean":
        sort_col = "mean"
    else:
        raise ValueError(f"Unsupported agg='{agg}'. Use 'mean' or 'max'.")

    # Sort in descending order (best first)
    t = t.sort_values(sort_col, ascending=ascending, na_position="last").reset_index(
        drop=True
    )
    return t


def _fps_from_smiles_list(smiles_list, radius=2, nBits=2048):
    fps = []
    for smi in smiles_list:
        m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if m:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits))
        else:
            fps.append(None)
    return fps


def _maxmin_diverse_indices(fps, k, rng):
    valid = [i for i, fp in enumerate(fps) if fp is not None]
    if len(valid) <= k:
        return valid
    selected = [valid[0]]
    min_d = np.ones(len(fps))
    while len(selected) < k:
        for j in valid:
            if j in selected:
                continue
            d = min(
                1.0 - DataStructs.TanimotoSimilarity(fps[j], fps[s])
                for s in selected
                if fps[j] is not None and fps[s] is not None
            )
            min_d[j] = d
        farthest = max(valid, key=lambda j: min_d[j])
        selected.append(farthest)
    return selected


def top_diverse_molecules_within_top_scaffolds(
    df: pd.DataFrame,
    ranked_scaffolds: pd.DataFrame,
    scaf_col="Murcko Scaffold",
    y_col="standard_value_nm_log10",
    smiles_col="std_smiles",
    selfies_col="canonical_selfies",
    score_col="mean",
    top_scaffolds=20,
    per_scaffold=5,
    pool_top_m=50,
    radius=2,
    nBits=2048,
    seed=0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    scaf_top = ranked_scaffolds.sort_values(score_col, ascending=True).head(
        top_scaffolds
    )

    all_rows = []
    for scaf in scaf_top[scaf_col]:
        sub = df[df[scaf_col] == scaf].dropna(subset=[y_col, smiles_col])
        if sub.empty:
            continue

        # remove duplicates within scaffold (by SMILES or SELFIES if available)
        if selfies_col in sub.columns:
            sub = sub.drop_duplicates(subset=[smiles_col, selfies_col])
        else:
            sub = sub.drop_duplicates(subset=[smiles_col])

        sub = sub.sort_values(y_col, ascending=True).head(pool_top_m)
        fps = _fps_from_smiles_list(sub[smiles_col].tolist(), radius, nBits)
        sel = _maxmin_diverse_indices(fps, min(per_scaffold, len(sub)), rng)
        chosen = sub.iloc[sel].copy()
        chosen["_rank_in_scaffold_diverse"] = np.arange(1, len(chosen) + 1)
        all_rows.append(chosen)

    if not all_rows:
        return pd.DataFrame()

    out = pd.concat(all_rows, ignore_index=True)

    if selfies_col in out.columns:
        out = out.drop_duplicates(subset=[smiles_col, selfies_col]).copy()
    else:
        out = out.drop_duplicates(subset=[smiles_col]).copy()

    out["value_nM"] = (10 ** out[y_col].astype(float)).astype(float)

    n_final = len(out)
    n_unique = out[smiles_col].nunique()
    if n_final != n_unique:
        print(
            f"[dedup] Removed {n_final - n_unique} duplicate molecules; final unique = {n_unique}"
        )
    else:
        print(f"[dedup] All {n_final} molecules are unique.")

    return out


def plot_top_10_scaffolds_with_example(scaffold_groups):

    scaffold_counts = [(scaf, len(mols)) for scaf, mols in scaffold_groups.items()]
    scaffold_counts.sort(key=lambda x: x[1], reverse=True)

    top_10 = scaffold_counts[:10]
    if not top_10:
        print("No scaffolds available to plot.")
        return

    print(top_10)

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))
    axes = axes.flatten()

    highlight_color = (1.0, 1.0, 0.0)  # yellow in (R, G, B)

    for i, (scaffold_smiles, count) in enumerate(top_10):
        ax = axes[i]

        scaffold_mol = Chem.MolFromSmiles(scaffold_smiles)
        if not scaffold_mol:
            ax.set_axis_off()
            continue

        example_smi = scaffold_groups[scaffold_smiles][0]
        example_mol = Chem.MolFromSmiles(example_smi)
        if not example_mol:
            ax.set_axis_off()
            continue

        scaffold_match = example_mol.GetSubstructMatch(scaffold_mol)

        img = Draw.MolToImage(
            example_mol,
            size=(300, 300),
            highlightAtoms=scaffold_match,
            highlightColor=highlight_color,
        )
        ax.imshow(img)
        ax.set_axis_off()

        ax.set_title(f"{scaffold_smiles}\nCount: {count}", fontsize=8)

    for j in range(len(top_10), 10):
        axes[j].set_axis_off()

    plt.tight_layout()
    plt.show()


def plot_top_10_scaffolds_with_example(scaffold_groups):
    scaffold_counts = [(scaf, len(mols)) for scaf, mols in scaffold_groups.items()]
    scaffold_counts.sort(key=lambda x: x[1], reverse=True)
    top_10 = scaffold_counts[:10]
    if not top_10:
        print("No scaffolds available to plot.")
        return
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))
    axes = axes.flatten()
    highlight_color = (1.0, 1.0, 0.0)
    for i, (scaffold_smiles, count) in enumerate(top_10):
        ax = axes[i]
        scaffold_mol = Chem.MolFromSmiles(scaffold_smiles)
        if not scaffold_mol:
            ax.set_axis_off()
            continue
        example_smi = scaffold_groups[scaffold_smiles][0]
        example_mol = Chem.MolFromSmiles(example_smi)
        if not example_mol:
            ax.set_axis_off()
            continue
        scaffold_match = example_mol.GetSubstructMatch(scaffold_mol)
        img = Draw.MolToImage(
            example_mol,
            size=(300, 300),
            highlightAtoms=scaffold_match,
            highlightColor=highlight_color,
        )
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(f"{scaffold_smiles}\nCount: {count}", fontsize=8)
    for j in range(len(top_10), 10):
        axes[j].set_axis_off()
    plt.tight_layout()
    plt.show()


def plot_top_scaffold_means(
    per_scaf_df,
    scaf_col="Murcko Scaffold",
    mean_col="mean",
    std_col="std",
    top_k=20,
):
    df = (
        per_scaf_df.sort_values(mean_col, ascending=True)
        .head(top_k)
        .reset_index(drop=True)
    )
    y = np.arange(len(df))
    plt.figure(figsize=(9, 0.5 * top_k + 1))
    plt.errorbar(
        df[mean_col],
        y,
        xerr=df[std_col],
        fmt="o",
        color="navy",
        capsize=3,
        label=f"{mean_col} ± {std_col}",
    )
    plt.yticks(y, [f"{i+1}. {s[:60]}" for i, s in enumerate(df[scaf_col].astype(str))])
    plt.gca().invert_yaxis()
    plt.xlabel("label (lower = better)")
    plt.title(f"Top {top_k} scaffolds by mean {mean_col}")
    plt.grid(axis="x", ls="--", alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()


def inter_fingerprint_similarity(fp_dict):
    total_sim = 0.0
    num_pairs = 0
    fp = list(fp_dict.values())
    for i in range(len(fp)):
        for j in range(i+1, len(fp)):
            sim = DataStructs.TanimotoSimilarity(fp[i], fp[j])
            total_sim += sim
            num_pairs += 1
    mean_sim = total_sim / num_pairs
    return mean_sim

def cross_fingerprint_similarity(fp_dict_1, fp_dict_2):
    total_sim = 0.0
    num_pairs = 0
    fp_1 = list(fp_dict_1.values())
    fp_2 = list(fp_dict_2.values())
    for i in range(len(fp_1)):
        for j in range(len(fp_2)):
            sim = DataStructs.TanimotoSimilarity(fp_1[i], fp_2[j])
            total_sim += sim
            num_pairs += 1
    mean_sim = total_sim / num_pairs
    return mean_sim

def build_fp_dict(smiles_set: Iterable[str], radius:int=2, nbits:int=2048) -> dict:
    d = {}
    for smi in smiles_set:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            morgan_gen = GetMorganGenerator(radius=radius, fpSize=nbits)
            fp = morgan_gen.GetFingerprint(mol)
            d[smi] = fp
        except Exception:
            continue
    return d

def interscaffoldssimilarity(scaff_list: Iterable[str]):
    fp_dict = build_fp_dict(scaff_list)
    intscaffdist = inter_fingerprint_similarity(fp_dict)
    return intscaffdist
    
def crossscaffoldsimilarity(scaff_list_1: Iterable[str], scaff_list_2: Iterable[str]):
    fp_dict_1 = build_fp_dict(scaff_list_1)
    fp_dict_2 = build_fp_dict(scaff_list_2)
    crossscaffdist = cross_fingerprint_similarity(fp_dict_1, fp_dict_2)
    return crossscaffdist