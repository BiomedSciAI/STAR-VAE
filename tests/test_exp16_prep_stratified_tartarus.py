"""Regression tests for leakage-free exp16 dataset construction."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
from rdkit import Chem

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "bmfm_sm_gen" / "scripts" / "exp16_prep_stratified_tartarus.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("exp16_prep", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exp16 = _load_script()


def _canonical(smiles):
    return {Chem.MolToSmiles(Chem.MolFromSmiles(value), canonical=True) for value in smiles}


def test_split_precedes_training_only_oversampling_and_jitter():
    smiles = ["C" * length for length in range(1, 19)]
    scores = [-9.0] * 6 + [-6.0] * 6 + [-2.0] * 6
    raw = pd.DataFrame(
        {
            "smiles": smiles + ["C(C)", "not-a-smiles"],
            "selfies": [f"selfies-{i}" for i in range(20)],
            "6y2f_score": scores + [-9.0, -6.0],
        }
    )

    train, validation, summary = exp16.build_target_splits(
        raw, "6y2f", val_fraction=0.2, seed=7
    )

    assert len(train) == 2997
    assert len(validation) == 3
    assert validation["smiles"].is_unique
    assert set(validation["6y2f_score"]) == {-9.0, -6.0, -2.0}
    assert _canonical(train["smiles"]).isdisjoint(_canonical(validation["smiles"]))
    assert summary["canonical_overlap_count"] == 0
    assert summary["n_invalid_smiles_dropped"] == 1
    assert summary["n_duplicate_canonical_dropped"] == 1
    assert summary["n_train_unique_canonical"] == 15
    assert summary["n_validation_unique_canonical"] == 3
    assert all(
        counts["validation_rows"] == counts["validation_pool_unique"] == 1
        and counts["validation_augmented_rows"] == 0
        and counts["train_oversampled_rows"] == 994
        for counts in summary["bucket_counts"].values()
    )
    assert any(
        train.groupby("smiles")["6y2f_score"].nunique() > 1
    ), "training duplicates should receive jittered labels"
