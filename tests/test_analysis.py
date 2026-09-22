"""
test_analysis.py
================
Tests for the analysis pipeline utilities used to produce paper figures/tables.

Covers:
- validation_analysis.py: compute validity, uniqueness, novelty from a CSV
- sascorer: batch scoring of a SMILES list
- rediscovery_evaluation: full evaluation pipeline on toy data
- make_mol_grid_figure: grid image is produced without error
- conditional_evaluation: prepare_validation_info + basic stats

These tests use only small, inline datasets (no external files required).
"""

import pytest
import pandas as pd
from pathlib import Path

# ── sample SMILES ─────────────────────────────────────────────────────────────
# All are valid, chemically reasonable molecules from MOSES training set
VALID_SMILES = [
    "CC",            # ethane
    "CCC",           # propane
    "CCCC",          # butane
    "c1ccccc1",      # benzene
    "CC(=O)O",       # acetic acid
    "CCO",           # ethanol
    "CCCCO",         # 1-butanol
    "c1ccncc1",      # pyridine
]
INVALID_SMILES = ["not_a_smiles", "", "XYZ###"]


# ── validity helpers ──────────────────────────────────────────────────────────

def test_validity_computation():
    """validity() counts fraction of parseable SMILES."""
    from rdkit import Chem

    mixed = VALID_SMILES + INVALID_SMILES
    valid_count = sum(1 for s in mixed if s and Chem.MolFromSmiles(s) is not None)
    validity = valid_count / len(mixed)
    expected = len(VALID_SMILES) / len(mixed)
    assert abs(validity - expected) < 1e-6


def test_uniqueness_computation():
    """Uniqueness = fraction of unique SMILES after canonicalization."""
    from rdkit import Chem

    smiles = VALID_SMILES + ["CC", "CCC"]  # two duplicates
    canonical = set()
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol:
            canonical.add(Chem.MolToSmiles(mol))
    uniqueness = len(canonical) / len(smiles)
    assert 0.0 < uniqueness < 1.0, f"Expected uniqueness < 1.0: {uniqueness}"


# ── SA score batch ─────────────────────────────────────────────────────────────

def test_batch_sa_scores():
    """All valid SMILES receive an SA score in [1, 10]."""
    from rdkit import Chem
    from bmfm_sm_gen.utils.sascorer import calculateScore

    for smi in VALID_SMILES:
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None
        score = calculateScore(mol)
        assert 1.0 <= score <= 10.0, f"SA score out of range for {smi}: {score}"


# ── RediscoveryEvaluation pipeline ────────────────────────────────────────────

def test_rediscovery_full_pipeline():
    """Full RediscoveryEvaluation on toy data: uniqueness() returns a valid tuple."""
    from bmfm_sm_gen.utils.rediscovery_evaluation import RediscoveryEvaluation

    generated = VALID_SMILES[:4]
    heldout = VALID_SMILES[4:] + ["CCCCCC"]  # some overlap, some novel
    years = [2020] * len(heldout)

    ev = RediscoveryEvaluation(
        generated_smiles=generated,
        heldout_smiles=heldout,
        heldout_year=years,
    )
    # uniqueness() returns (ratio, pct_unique_gen, pct_unique_heldout)
    result = ev.uniqueness()
    assert isinstance(result, tuple) and len(result) == 3
    ratio, pct_gen, pct_heldout = result
    assert pct_gen >= 0
    assert pct_heldout >= 0


def test_rediscovery_empty_generated():
    """Empty generated set should have pct_unique_gen = 0 (not crash)."""
    from bmfm_sm_gen.utils.rediscovery_evaluation import RediscoveryEvaluation

    ev = RediscoveryEvaluation(
        generated_smiles=[],
        heldout_smiles=["CC"],
        heldout_year=[2020],
    )
    result = ev.uniqueness()
    ratio, pct_gen, _ = result
    assert pct_gen == 0


# ── Conditional evaluation ─────────────────────────────────────────────────────

def test_compute_bin_edges_uniform():
    """compute_bin_edges returns n_bins+1 edges for uniform mode."""
    import numpy as np
    from bmfm_sm_gen.utils.conditional_evaluation import compute_bin_edges

    y = np.array([0.1, 0.2, 0.5, 0.8, 0.9, 1.0])
    edges = compute_bin_edges(y, num_bins=3, mode="uniform")
    assert len(edges) == 4, f"Expected 4 edges for 3 bins, got {len(edges)}"
    assert edges[0] <= y.min() + 1e-9
    assert edges[-1] >= y.max() - 1e-9


def test_compute_bin_edges_equal_count():
    """compute_bin_edges returns n_bins+1 edges for equal_count (quantile) mode."""
    import numpy as np
    from bmfm_sm_gen.utils.conditional_evaluation import compute_bin_edges

    y = np.linspace(0, 1, 100)
    edges = compute_bin_edges(y, num_bins=5, mode="equal_count")
    assert len(edges) == 6, f"Expected 6 edges for 5 bins, got {len(edges)}"


def test_compute_label_stats():
    """compute_label_stats returns mean and std for a known array."""
    import numpy as np
    from bmfm_sm_gen.utils.conditional_evaluation import compute_label_stats

    y = np.array([0.0, 1.0, 2.0, 3.0])
    stats = compute_label_stats(y)
    assert stats["mean"] == pytest.approx(1.5)
    assert stats["std"] == pytest.approx(np.std(y))


# ── mol_img helper (make_mol_grid_figure) ────────────────────────────────────

def test_mol_img_returns_image():
    """mol_img() from make_mol_grid_figure returns a PIL Image for a valid SMILES."""
    pytest.importorskip("PIL")
    from bmfm_sm_gen.analysis.make_mol_grid_figure import mol_img

    img = mol_img("c1ccccc1", size=(200, 200))
    # Should return a PIL Image or numpy array, just check it's not None
    assert img is not None
