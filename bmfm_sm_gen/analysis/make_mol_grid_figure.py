#!/usr/bin/env python3
"""
6-row × 5-column high-quality RDKit molecule grid figure.

Pre-verified candidate assignments:
  Col 1  - Exact Rediscovery VAE: one molecule from each of ABL1, BRD4, GRL
            (hiv1protease has zero exact matches across all N), then 2 more from ABL1/BRD4
  Col 2  - Scaffold Rediscovery VAE:   ABL1  CHEMBL4562851  N=50000  (sim>0.6, unique top-4)
  Col 3  - Cluster  Rediscovery VAE:   BRD4  CHEMBL2181709  N=50000  (sim>0.6, unique top-4)
  Col 4  - Scaffold Rediscovery CVAE:  GRL   CHEMBL1231530  N=50000  (best available)
  Col 5  - Cluster  Rediscovery CVAE:  hiv1protease CHEMBL39772 N=50000 (best available)
"""

import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, DataStructs, AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec

# ── paths ──────────────────────────────────────────────────────────────────────
BASE     = os.environ.get("OUTPUT_DIR", "outputs")
DATA_DIR = os.environ.get("CHEMBL_DATA_DIR", "data/chembl/cluster_split")
DATASETS = ["ABL1", "BRD4", "GRL", "hiv1protease"]
N_VALUES = [10000, 20000, 30000, 40000, 50000]

DATASET_LABEL = {
    "ABL1": "ABL1",
    "BRD4": "BRD4",
    "GRL": "GRL",
    "hiv1protease": "HIV 1 Protease",
}

# ── layout constants ────────────────────────────────────────────────────────────
MOL_PX   = (500, 500)   # per-molecule image resolution
FIG_W    = 24.0          # figure width  (inches)
FIG_H    = 18.0          # figure height (inches), extra tall for clear heading row
HEAD_FS  = 20            # heading font size (pt)
LAB_FS   = 14            # molecule label font size (pt)
HEADING_BG = "#1f3a5f"

HEADINGS = [
    "Exact Rediscovery\nby VAE",
    "Scaffold Rediscovery\nby VAE",
    "Cluster Rediscovery\nby VAE",
    "Scaffold Rediscovery\nby CVAE",
    "Cluster Rediscovery\nby CVAE",
]

# ── RDKit helpers ──────────────────────────────────────────────────────────────
def mol_from(smi):
    return Chem.MolFromSmiles(str(smi)) if smi else None

def canon(smi):
    m = mol_from(smi)
    return Chem.MolToSmiles(m, canonical=True) if m else None

def fp(smi):
    m = mol_from(smi)
    if m is None: return None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)

def tansim(f1, f2):
    return DataStructs.TanimotoSimilarity(f1, f2)

def bertz(smi):
    m = mol_from(smi)
    return Descriptors.BertzCT(m) if m else float("inf")

def murcko(smi):
    m = mol_from(smi)
    if m is None: return None
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m), canonical=True)

def mol_img(smi, size=(500, 500)):
    m = mol_from(smi)
    if m is None:
        from PIL import Image
        return Image.new("RGB", size, "white")
    return Draw.MolToImage(m, size=size, kekulize=True)

def smiles_col(df):
    for c in ("generated_smiles", "smiles", "Smiles"):
        if c in df.columns: return c
    raise KeyError(list(df.columns))

def load_test(dataset):
    for f in os.listdir(DATA_DIR):
        if f.endswith("_test.csv") and dataset in f:
            return pd.read_csv(os.path.join(DATA_DIR, f))
    raise FileNotFoundError(dataset)

def vae_folder(n):  return f"chembl_comp_validation_vae_{n}"
def cvae_folder(n): return f"chembl_comp_validation_cvae_decoder_only_{n}"

# ══════════════════════════════════════════════════════════════════════════════
# Col 1 - Exact rediscoveries by VAE
# Rule: at least one from each dataset that has matches; hiv1protease has none.
# Fill 5 slots: one best (lowest BertzCT) from ABL1, BRD4, GRL, then next-best
# from ABL1 and BRD4.
# ══════════════════════════════════════════════════════════════════════════════

# Pre-verified best exact matches per dataset (from exhaustive search).
# Row 0 replaced: CHEMBL4562851 (ABL1) was also the heldout shown in Col 2, clash removed.
# Row 2 replaced: CHEMBL44814   (GRL)  swapped for a non-clashing molecule with diversity.
# Both replacements verified absent from all molecules shown in cols 2-5.
EXACT_TARGETS = [
    # (dataset, chembl_id, N, note)
    ("GRL",          "CHEMBL9419",    40000, "GRL"),    # row 0, replaces CHEMBL4562851
    ("BRD4",         "CHEMBL1828985", 20000, "BRD4"),   # row 1, unchanged
    ("ABL1",         "CHEMBL3810382", 20000, "ABL1"),   # row 2, replaces CHEMBL44814
    ("ABL1",         "CHEMBL2316869", 20000, "ABL1"),   # row 3, unchanged
    ("BRD4",         "CHEMBL1828984", 10000, "BRD4"),   # row 4, unchanged
]

def collect_exact_rediscoveries():
    """
    Collect the 5 pre-identified exact rediscovery molecules, verify they exist.
    Returns list of (smi, chembl_id, dataset, N).
    """
    results = []
    for dataset, target_cid, n, _ in EXACT_TARGETS:
        test_df = load_test(dataset)
        t_row = test_df[test_df["Molecule ChEMBL ID"] == target_cid]
        if t_row.empty:
            raise RuntimeError(f"Exact target {target_cid} not in {dataset} test set")
        h_smi = str(t_row.iloc[0]["Smiles"])
        # Verify it appears in generated set
        gen_path = os.path.join(BASE, vae_folder(n),
                                f"{dataset}_generated-scaffolds.csv")
        gen_df  = pd.read_csv(gen_path)
        gen_c   = smiles_col(gen_df)
        h_c     = canon(h_smi)
        found   = any(canon(str(r[gen_c])) == h_c for _, r in gen_df.iterrows())
        if not found:
            raise RuntimeError(
                f"Exact target {target_cid} not found in VAE generated set "
                f"(dataset={dataset}, N={n})")
        results.append((h_smi, target_cid, dataset, n))
        print(f"  [exact] {dataset} {target_cid} N={n} BertzCT={bertz(h_smi):.1f} ✓")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Cols 2 / 4 - Scaffold rediscovery
# ══════════════════════════════════════════════════════════════════════════════
SCAFFOLD_SPEC = {
    # VAE: ABL1, sim>0.6 confirmed
    "VAE":  dict(n=50000, dataset="ABL1",
                 target_id="CHEMBL4562851", sim_threshold=0.6),
    # CVAE: GRL, best available (no sim>0.6 exists anywhere for CVAE)
    "CVAE": dict(n=50000, dataset="GRL",
                 target_id="CHEMBL1231530", sim_threshold=0.0),
}

def collect_scaffold_rediscovery(model):
    spec      = SCAFFOLD_SPEC[model]
    n         = spec["n"]
    dataset   = spec["dataset"]
    target_id = spec["target_id"]
    sim_thr   = spec["sim_threshold"]

    folder  = vae_folder(n) if model == "VAE" else cvae_folder(n)
    gen_df  = pd.read_csv(os.path.join(
        BASE, folder,
        f"{dataset}_generated-scaffolds.csv" if model == "VAE"
        else f"{dataset}_cvae_generated-scaffolds.csv"))
    test_df = load_test(dataset)
    gen_c   = smiles_col(gen_df)

    h_row = test_df[test_df["Molecule ChEMBL ID"] == target_id]
    if h_row.empty:
        raise RuntimeError(f"{target_id} not found in {dataset} test set")
    h_smi = str(h_row.iloc[0]["Smiles"])
    h_c   = canon(h_smi)
    h_sc  = murcko(h_smi)
    h_fp  = fp(h_smi)

    # Build scaffold → [(smi, fp)]
    scaff_to_gen = {}
    for _, row in gen_df.iterrows():
        smi = str(row[gen_c])
        sc  = (str(row["scaffold"]) if "scaffold" in gen_df.columns
               and isinstance(row.get("scaffold"), str) else murcko(smi))
        if sc:
            fpi = fp(smi)
            if fpi is not None:
                scaff_to_gen.setdefault(sc, []).append((smi, fpi))

    if h_sc not in scaff_to_gen:
        raise RuntimeError(f"Scaffold of {target_id} not in generated set")

    # Collect hits: same scaffold, not identical, sim > threshold, UNIQUE canonical
    raw = [(s, tansim(h_fp, f)) for s, f in scaff_to_gen[h_sc]
           if canon(s) != h_c]
    raw = [(s, sim) for s, sim in raw if sim > sim_thr]
    raw.sort(key=lambda x: -x[1])
    seen_c, unique_hits = set(), []
    for s, sim in raw:
        c = canon(s)
        if c not in seen_c:
            seen_c.add(c)
            unique_hits.append((s, sim))

    if len(unique_hits) < 4:
        raise RuntimeError(
            f"Only {len(unique_hits)} unique generated mols for scaffold of "
            f"{target_id} with sim>{sim_thr}")

    top4 = unique_hits[:4]
    print(f"  [{model} scaffold] {dataset} {target_id} BertzCT={bertz(h_smi):.1f}  "
          f"sims={[round(s,3) for _,s in top4]}")
    return h_smi, target_id, dataset, top4


# ══════════════════════════════════════════════════════════════════════════════
# Cols 3 / 5 - Cluster rediscovery
# ══════════════════════════════════════════════════════════════════════════════
CLUSTER_SPEC = {
    # VAE: BRD4, sim>0.6 confirmed
    "VAE":  dict(n=50000, dataset="BRD4",
                 target_id="CHEMBL2181709", sim_threshold=0.6),
    # CVAE: hiv1protease, best available
    "CVAE": dict(n=50000, dataset="hiv1protease",
                 target_id="CHEMBL39772",  sim_threshold=0.0),
}

def collect_cluster_rediscovery(model):
    spec      = CLUSTER_SPEC[model]
    n         = spec["n"]
    dataset   = spec["dataset"]
    target_id = spec["target_id"]
    sim_thr   = spec["sim_threshold"]

    folder    = vae_folder(n) if model == "VAE" else cvae_folder(n)
    clust_csv = (f"{dataset}_generated-scaffolds-redis-best-clustering-centroid.csv"
                 if model == "VAE"
                 else f"{dataset}_cvae_generated-scaffolds-redis-best-clustering-centroid.csv")
    clust_df = pd.read_csv(os.path.join(BASE, folder, clust_csv))
    test_df  = load_test(dataset)

    hc_df = test_df[test_df["centroid"] >= 0]
    h_row = hc_df[hc_df["Molecule ChEMBL ID"] == target_id]
    if h_row.empty:
        raise RuntimeError(f"{target_id} not found in {dataset} heldout centroids")
    hrow         = h_row.iloc[0]
    centroid_smi = str(hrow["Smiles"])
    cluster_id   = int(hrow["cluster_index"])

    cfp = fp(centroid_smi)
    cc  = canon(centroid_smi)
    near = clust_df[clust_df["best_cluster_id"] == cluster_id]

    raw = []
    for _, nr in near.iterrows():
        gsmi = str(nr["generated_smiles"])
        if canon(gsmi) == cc: continue
        gfp = fp(gsmi)
        if gfp is None: continue
        sim = tansim(cfp, gfp)
        if sim > sim_thr:
            raw.append((gsmi, sim))

    raw.sort(key=lambda x: -x[1])
    seen_c, unique_hits = set(), []
    for s, sim in raw:
        c = canon(s)
        if c not in seen_c:
            seen_c.add(c)
            unique_hits.append((s, sim))

    if len(unique_hits) < 4:
        raise RuntimeError(
            f"Only {len(unique_hits)} unique cluster hits for {target_id}")

    top4 = unique_hits[:4]
    print(f"  [{model} cluster] {dataset} {target_id} BertzCT={bertz(centroid_smi):.1f}  "
          f"sims={[round(s,3) for _,s in top4]}")
    return centroid_smi, target_id, dataset, top4


# ══════════════════════════════════════════════════════════════════════════════
# Figure
# ══════════════════════════════════════════════════════════════════════════════

def render_mol(ax, smi, label, label_color="black", bg_color=None):
    if bg_color:
        ax.set_facecolor(bg_color)
    ax.imshow(np.array(mol_img(smi, size=MOL_PX)))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel(label, fontsize=LAB_FS, color=label_color,
                  labelpad=5, multialignment="center")


def make_heading_axes(fig, rect):
    """
    Create a standalone Axes for a column heading using figure-fraction rect
    [left, bottom, width, height].  Returns the axes.
    """
    ax = fig.add_axes(rect)
    return ax


def main():
    # ── collect data ──────────────────────────────────────────────────────────
    print("Col 1: exact rediscoveries …")
    exact5 = collect_exact_rediscoveries()

    print("Col 2: scaffold VAE (ABL1) …")
    sv_h, sv_id, sv_ds, sv_gen = collect_scaffold_rediscovery("VAE")

    print("Col 3: cluster VAE (BRD4) …")
    cv_h, cv_id, cv_ds, cv_gen = collect_cluster_rediscovery("VAE")

    print("Col 4: scaffold CVAE (GRL) …")
    sc_h, sc_id, sc_ds, sc_gen = collect_scaffold_rediscovery("CVAE")

    print("Col 5: cluster CVAE (hiv1protease) …")
    cc_h, cc_id, cc_ds, cc_gen = collect_cluster_rediscovery("CVAE")

    # ── figure layout ─────────────────────────────────────────────────────────
    # We use a two-level GridSpec:
    #   Outer: 2 rows: row 0 = heading band, row 1 = molecule grid
    #   Inner molecule grid: 5 rows × 5 cols
    #
    # The heading band height is set in inches so the two-line text fits
    # comfortably regardless of figure proportions.

    NCOLS    = 5
    MOL_ROWS = 5

    # Heading band: fixed height in data units via explicit axes placement
    HEAD_H_IN = 1.4          # inches allocated to heading row
    MOL_H_IN  = (FIG_H - HEAD_H_IN)   # remaining height for 5 mol rows

    HEAD_FRAC = HEAD_H_IN / FIG_H     # fraction of figure height
    MOL_FRAC  = MOL_H_IN  / FIG_H

    # margins (figure fractions)
    LEFT, RIGHT = 0.005, 0.995
    TOP_MAR, BOT_MAR = 0.005, 0.02
    HGAP   = 0.04   # gap between heading band and mol grid (figure fraction)
    HSPACE = 0.42   # relative vertical spacing between mol rows
    WSPACE = 0.06   # relative horizontal spacing between columns

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=150)
    fig.patch.set_facecolor("white")

    # ── heading axes: one per column, placed manually ────────────────────────
    # heading band occupies [TOP_MAR .. TOP_MAR+HEAD_FRAC] from top
    head_bottom = 1.0 - TOP_MAR - HEAD_FRAC
    head_top    = 1.0 - TOP_MAR
    col_w = (RIGHT - LEFT) / NCOLS

    head_axes = []
    for ci in range(NCOLS):
        ax_left = LEFT + ci * col_w
        ax = fig.add_axes([ax_left, head_bottom, col_w, HEAD_FRAC])
        ax.set_facecolor(HEADING_BG)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

        ax.text(
            0.5, 0.5,
            HEADINGS[ci],
            ha="center", va="center",
            fontsize=HEAD_FS, fontweight="bold", color="white",
            transform=ax.transAxes,
            multialignment="center",
            linespacing=1.5,
        )
        head_axes.append(ax)

    # ── molecule GridSpec below heading band ──────────────────────────────────
    mol_top    = head_bottom - HGAP
    mol_bottom = BOT_MAR
    mol_height = mol_top - mol_bottom

    gs = mgridspec.GridSpec(
        MOL_ROWS, NCOLS,
        left=LEFT, right=RIGHT,
        bottom=mol_bottom, top=mol_top,
        hspace=HSPACE, wspace=WSPACE,
    )

    def add_mol(row, col, smi, label, label_color="black", bg_color=None):
        ax = fig.add_subplot(gs[row, col])
        render_mol(ax, smi, label, label_color=label_color, bg_color=bg_color)
        return ax

    # ── Col 0: exact rediscoveries ───────────────────────────────────────────
    for ri, (smi, cid, ds, n) in enumerate(exact5):
        add_mol(ri, 0, smi, f"{cid}\n{DATASET_LABEL[ds]}  N={n:,}")

    # ── Col 1: scaffold VAE  (row 0 = heldout, rows 1-4 = generated) ─────────
    add_mol(0, 1, sv_h,
            f"{sv_id}\n{DATASET_LABEL[sv_ds]}",
            label_color="#1a5c1a", bg_color="#f0fff0")
    for ri, (smi, sim) in enumerate(sv_gen):
        add_mol(ri + 1, 1, smi, f"Sim = {sim:.3f}")

    # ── Col 2: cluster VAE ────────────────────────────────────────────────────
    add_mol(0, 2, cv_h,
            f"{cv_id}\n{DATASET_LABEL[cv_ds]}",
            label_color="#1a5c1a", bg_color="#f0fff0")
    for ri, (smi, sim) in enumerate(cv_gen):
        add_mol(ri + 1, 2, smi, f"Sim = {sim:.3f}")

    # ── Col 3: scaffold CVAE ──────────────────────────────────────────────────
    add_mol(0, 3, sc_h,
            f"{sc_id}\n{DATASET_LABEL[sc_ds]}",
            label_color="#1a5c1a", bg_color="#f0fff0")
    for ri, (smi, sim) in enumerate(sc_gen):
        add_mol(ri + 1, 3, smi, f"Sim = {sim:.3f}")

    # ── Col 4: cluster CVAE ───────────────────────────────────────────────────
    add_mol(0, 4, cc_h,
            f"{cc_id}\n{DATASET_LABEL[cc_ds]}",
            label_color="#1a5c1a", bg_color="#f0fff0")
    for ri, (smi, sim) in enumerate(cc_gen):
        add_mol(ri + 1, 4, smi, f"Sim = {sim:.3f}")

    # ── Dotted vertical dividers (figure coordinates) ────────────────────────
    # Line 1: between col 0 and col 1  → x = LEFT + 1 * col_w
    # Line 2: between col 2 and col 3  → x = LEFT + 3 * col_w
    # Each line runs from the bottom of the molecule grid to the top of the
    # heading band, with a small margin on each side.
    for div_col in (1, 3):
        x_fig = LEFT + div_col * col_w
        fig.add_artist(
            plt.Line2D(
                [x_fig, x_fig],
                [0.0, 1.0],              # full page height
                transform=fig.transFigure,
                color="#555555",
                linewidth=1.5,
                linestyle=(0, (6, 4)),   # loosely dotted
                clip_on=False,
                zorder=10,
            )
        )

    out = "mol_grid_figure.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight", dpi=300)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
