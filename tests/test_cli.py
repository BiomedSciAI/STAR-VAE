"""Cheap import and command-line contract tests for installed entry points."""

import importlib
import random

import numpy as np
import pandas as pd
import pytest
import torch
from click.testing import CliRunner


@pytest.mark.parametrize(
    "module_name",
    [
        "bmfm_sm_gen",
        "bmfm_sm_gen.data.base_tokenizer",
        "bmfm_sm_gen.models.vae",
        "bmfm_sm_gen.models.cvae",
        "bmfm_sm_gen.trainers.lm_vae",
        "bmfm_sm_gen.trainers.lm_cvae",
        "bmfm_sm_gen.launch.run_train",
        "bmfm_sm_gen.launch.run_generate_molecules",
    ],
)
def test_public_modules_import(module_name):
    assert importlib.import_module(module_name).__name__ == module_name


def test_generate_entry_point_help_does_not_load_checkpoint():
    from bmfm_sm_gen.launch.run_generate_molecules import main

    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0, result.output
    assert "cli" in result.output
    assert "cli-bulk" in result.output


def test_generate_group_without_subcommand_shows_help(monkeypatch):
    from bmfm_sm_gen.launch import run_generate_molecules as generation

    monkeypatch.setattr(
        generation, "load_model", lambda **kwargs: pytest.fail("checkpoint loaded")
    )
    result = CliRunner().invoke(generation.main, [])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "cli-bulk" in result.output


def test_generation_seed_covers_python_numpy_and_torch():
    from bmfm_sm_gen.launch.run_generate_molecules import set_seed

    set_seed(17)
    first = (random.random(), np.random.random(), torch.rand(1).item())
    set_seed(17)
    second = (random.random(), np.random.random(), torch.rand(1).item())
    assert first == second


@pytest.mark.parametrize("command", ["cli", "cli-bulk"])
def test_generate_subcommand_help_exposes_sampling_options(command):
    from bmfm_sm_gen.launch.run_generate_molecules import main

    result = CliRunner().invoke(main, [command, "--help"])
    assert result.exit_code == 0, result.output
    assert "--temperature" in result.output
    assert "--top-k" in result.output
    assert "--seed" in result.output


def test_generate_cli_creates_output_parent(tmp_path, monkeypatch):
    from bmfm_sm_gen.launch import run_generate_molecules as generation

    ckpt = tmp_path / "model.ckpt"
    vocab = tmp_path / "vocab.json"
    ckpt.touch()
    vocab.touch()
    monkeypatch.setattr(generation, "load_model", lambda **kwargs: (object(), object()))
    monkeypatch.setattr(generation, "generate_unconstrained_molecules", lambda *args, **kwargs: ["C", None])
    monkeypatch.setattr(generation, "visualize_molecules", lambda *args, **kwargs: None)
    output = tmp_path / "nested" / "generated.csv"

    result = CliRunner().invoke(generation.main, [
        "cli", "--ckpt-path", str(ckpt), "--vocab-file", str(vocab),
        "--unconstrained", "--num-samples", "2", "--output-csv", str(output),
    ])

    assert result.exit_code == 0, result.output
    assert output.exists()
    assert len(pd.read_csv(output)) == 2


def test_generate_bulk_cli_creates_output_parent(tmp_path, monkeypatch):
    from bmfm_sm_gen.launch import run_generate_molecules as generation

    ckpt = tmp_path / "model.ckpt"
    vocab = tmp_path / "vocab.json"
    input_csv = tmp_path / "input.csv"
    ckpt.touch()
    vocab.touch()
    pd.DataFrame({"target_smiles": ["C"]}).to_csv(input_csv, index=False)
    monkeypatch.setattr(generation, "load_model", lambda **kwargs: (object(), object()))
    monkeypatch.setattr(
        generation,
        "generate_from_multiple_targets_batch",
        lambda **kwargs: [("C", "CC", 0.5)],
    )
    output = tmp_path / "bulk" / "generated.csv"

    result = CliRunner().invoke(generation.main, [
        "cli-bulk", "--ckpt-path", str(ckpt), "--vocab-file", str(vocab),
        "--input-csv", str(input_csv), "--output-csv", str(output),
    ])

    assert result.exit_code == 0, result.output
    assert output.exists()
    output_data = pd.read_csv(output)
    assert len(output_data) == 50
    assert list(output_data["attempt_id"]) == list(range(50))
    assert output_data["valid"].sum() == 1
    assert output_data.loc[1:, "error"].notna().all()
