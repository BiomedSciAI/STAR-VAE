"""
test_data.py
============
Tests for data loading (MolecularDataset, MolecularDataModule).

Verifies that:
- A synthetic SELFIES CSV is readable and produces correctly-shaped batches.
- Batched tensors have the expected dtype and are padded to max_len.
- Label columns (when present) are returned alongside input_ids.
"""

import os
import tempfile

import pandas as pd
import pytest
import torch

# Six short SELFIES strings that fit within max_len=74
_TINY_SELFIES = [
    "[C][C][=O][O]",
    "[C][Branch1][C][F][C][C]",
    "[O][=C][C][=C][C][=C][Ring1][=A]",
    "[N][C][C][=O][O]",
    "[C][C][C][C][C][C]",
    "[C][=C][C][=C][C][=C][Ring1][=A]",
    "[C][O][C]",
    "[N][=C][N][C][=O]",
]


@pytest.fixture
def tiny_selfies_csv(tmp_path):
    """Write a tiny SELFIES CSV into a temp file and return the path."""
    df = pd.DataFrame({"selfies": _TINY_SELFIES})
    p = tmp_path / "tiny_selfies.csv"
    df.to_csv(p, index=False)
    return str(p)


def test_datamodule_batch_shape(tiny_selfies_csv, vocab_file):
    """Train dataloader produces batches with batch_size=4 and seq_len ≤ max_len."""
    from bmfm_sm_gen.data.dm_mol import MolecularDataModule

    dm = MolecularDataModule(
        data_dir="", train_mode="mem",
        train_csv=tiny_selfies_csv, val_csv=tiny_selfies_csv,
        tokenizer="bmfm_sm_gen.data.base_tokenizer.SELFIESTokenizer",
        vocab_file=vocab_file,
        max_len=74, batch_size=4,
        allow_same_split=True,
        dataset_args={"modality": "selfies", "padding": "max_length"},
    )
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    assert "input_ids" in batch
    B, S = batch["input_ids"].shape
    assert B == 4, f"Expected batch_size=4, got {B}"
    assert S <= 74, f"Expected seq_len ≤ max_len=74, got {S}"
    assert batch["input_ids"].dtype == torch.long


def test_datamodule_val_batch_shape(tiny_selfies_csv, vocab_file):
    from bmfm_sm_gen.data.dm_mol import MolecularDataModule

    dm = MolecularDataModule(
        data_dir="", train_mode="mem",
        train_csv=tiny_selfies_csv, val_csv=tiny_selfies_csv,
        tokenizer="bmfm_sm_gen.data.base_tokenizer.SELFIESTokenizer",
        vocab_file=vocab_file,
        max_len=74, batch_size=4,
        allow_same_split=True,
        dataset_args={"modality": "selfies", "padding": "max_length"},
    )
    dm.setup()
    batch = next(iter(dm.val_dataloader()))
    B, S = batch["input_ids"].shape
    assert S <= 74, f"Val batch seq_len {S} exceeds max_len=74"


def test_dataset_length(tiny_selfies_csv, vocab_file):
    """Dataset length matches the number of rows in the CSV."""
    from bmfm_sm_gen.data.ds_mol import build_dataset
    from bmfm_sm_gen.data.base_tokenizer import SELFIESTokenizer

    tok = SELFIESTokenizer(vocab=vocab_file)
    ds = build_dataset(
        mode="mem",
        max_len=74,
        tokenizer=tok,
        dataset_args={"modality": "selfies", "padding": "max_length"},
        csv_file=tiny_selfies_csv,
    )
    assert len(ds) == len(_TINY_SELFIES), f"Unexpected dataset size: {len(ds)}"


def test_csv_label_columns(tiny_selfies_csv, vocab_file):
    """Rows with a label column are returned with a 'labels' key."""
    from bmfm_sm_gen.data.dm_mol import MolecularDataModule

    df = pd.read_csv(tiny_selfies_csv)
    df["y_normalized_minmax"] = 0.5
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f, index=False)
        fpath = f.name

    try:
        dm = MolecularDataModule(
            data_dir="", train_mode="mem",
            train_csv=fpath, val_csv=fpath,
            tokenizer="bmfm_sm_gen.data.base_tokenizer.SELFIESTokenizer",
            vocab_file=vocab_file,
            max_len=74, batch_size=4,
            allow_same_split=True,
            dataset_args={
                "modality": "selfies",
                "padding": "max_length",
                "label_columns": ["y_normalized_minmax"],
            },
        )
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        assert "labels" in batch, "Expected 'labels' key in batch when label_columns set"
    finally:
        os.unlink(fpath)


def test_split_by_cluster_keeps_all_clusters_disjoint():
    from bmfm_sm_gen.utils.data_split import split_by_cluster

    df = pd.DataFrame({
        "cluster_index": [0, 0, 0, 1, 1, 2, 2, 3, 4, 5],
        "value": range(10),
    })
    train, valid, test = split_by_cluster(
        df, train_frac=0.6, valid_frac=0.2, test_frac=0.2, random_state=7
    )
    cluster_sets = [set(part["cluster_index"]) for part in (train, valid, test)]
    assert cluster_sets[0].isdisjoint(cluster_sets[1])
    assert cluster_sets[0].isdisjoint(cluster_sets[2])
    assert cluster_sets[1].isdisjoint(cluster_sets[2])
    assert set().union(*cluster_sets) == set(df["cluster_index"])
    assert sorted(pd.concat([train, valid, test]).index) == sorted(df.index)
