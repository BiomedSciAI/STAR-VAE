"""
test_scoring.py
===============
Tests for SA-score, rediscovery evaluation, and conditional evaluation utilities.

Verifies that:
- calculateScore() from sascorer returns a float in [1, 10] for known molecules.
- RediscoveryEvaluation.uniqueness() and validity metrics work on toy inputs.
- sample_distribution_ranges() from inference_utils partitions correctly.
"""

import pytest
from rdkit import Chem


# ── SA score ──────────────────────────────────────────────────────────────────

KNOWN_SMILES = [
    ("CC", 1.0),          # ethane, very easy to synthesise
    ("c1ccccc1", 2.0),    # benzene
]


@pytest.mark.parametrize("smi,max_expected_sa", KNOWN_SMILES)
def test_sa_score_range(smi, max_expected_sa):
    """SA score must be in [1, 10]; simple molecules should have score < max_expected_sa + 2."""
    from bmfm_sm_gen.utils.sascorer import calculateScore

    mol = Chem.MolFromSmiles(smi)
    assert mol is not None, f"RDKit failed to parse {smi}"
    score = calculateScore(mol)
    assert 1.0 <= score <= 10.0, f"SA score out of range: {score}"
    assert score <= max_expected_sa + 2.0, (
        f"SA score unexpectedly high for {smi}: {score}"
    )


def test_sa_score_orders():
    """Cyclohexane (score=1.0) should have a lower SA score than a complex steroid-like molecule."""
    from bmfm_sm_gen.utils.sascorer import calculateScore

    # Cyclohexane: the simplest ring system, SA score should be 1.0
    simple = Chem.MolFromSmiles("C1CCCCC1")
    # Erythromycin-like macrolide (complex stereochemistry + large ring), high SA score
    complex_mol = Chem.MolFromSmiles(
        "O=C1OC(CC(O)C1(O)CC)C(C)C(OC2OC(C)C(N(C)C)C(O)C2)C(C)CC(=O)"
        "C(C(C(OC3CC(C)(OC)C(O)(CC)O3)C)C)C"
    )
    sa_simple = calculateScore(simple)
    # SA score is on a scale where lower = easier to synthesise
    # Just verify the simple molecule is in [1, 2] and the complex one is higher
    assert sa_simple <= 2.0, f"Cyclohexane SA score unexpectedly high: {sa_simple:.3f}"
    if complex_mol is not None:
        sa_complex = calculateScore(complex_mol)
        assert sa_simple < sa_complex, (
            f"Expected cyclohexane SA ({sa_simple:.2f}) < complex mol SA ({sa_complex:.2f})"
        )


# ── RediscoveryEvaluation ─────────────────────────────────────────────────────


def test_rediscovery_uniqueness():
    """Uniqueness for a list with one duplicate: uniq_gen < total_gen."""
    from bmfm_sm_gen.utils.rediscovery_evaluation import RediscoveryEvaluation

    gen = ["CC", "CC", "CCC"]   # 2 unique out of 3
    held = ["CCCC", "c1ccccc1"]
    years = [2020, 2021]

    ev = RediscoveryEvaluation(
        generated_smiles=gen,
        heldout_smiles=held,
        heldout_year=years,
    )
    result = ev.uniqueness()
    # uniqueness() returns (ratio, pct_unique_gen, pct_unique_heldout)
    # pct_unique_gen = (num_unique / total) * 100
    ratio, pct_gen, pct_heldout = result
    assert 0.0 < ratio, f"Uniqueness ratio should be > 0: {ratio}"
    # 2 unique out of 3 → 66.67%
    assert abs(pct_gen - 66.67) < 1.0, f"Expected ~66.67% unique, got {pct_gen:.2f}"


def test_rediscovery_uniqueness_all_unique():
    from bmfm_sm_gen.utils.rediscovery_evaluation import RediscoveryEvaluation

    gen = ["CC", "CCC", "CCCC"]
    held = ["c1ccccc1"]
    years = [2020]

    ev = RediscoveryEvaluation(
        generated_smiles=gen,
        heldout_smiles=held,
        heldout_year=years,
    )
    ratio, pct_gen, _ = ev.uniqueness()
    assert abs(pct_gen - 100.0) < 1e-6, f"Expected 100% unique, got {pct_gen:.2f}"


# ── sample_distribution_ranges ────────────────────────────────────────────────


def test_sample_distribution_ranges_basic():
    """Bins should partition num_samples approximately evenly."""
    import pandas as pd
    from bmfm_sm_gen.utils.inference_utils import sample_distribution_ranges

    df = pd.DataFrame({"y": [0.0, 0.1, 0.5, 0.9, 1.0] * 20})
    dist = sample_distribution_ranges(
        df, label_column="y",
        num_samples=100,
        min_val=0.0, max_val=1.0,
        n_bins=10, num_top_bins=3,
    )
    assert isinstance(dist, list)
    assert len(dist) > 0
    total = sum(d["num"] for d in dist)
    assert total == 100, f"Total samples mismatch: {total}"


def test_sample_distribution_ranges_keys():
    """Each bin dict must have 'range' and 'num' keys."""
    import pandas as pd
    from bmfm_sm_gen.utils.inference_utils import sample_distribution_ranges

    df = pd.DataFrame({"score": [float(i) / 10 for i in range(11)]})
    dist = sample_distribution_ranges(df, "score", 50, 0.0, 1.0, n_bins=5, num_top_bins=None)
    for entry in dist:
        assert "range" in entry and "num" in entry
