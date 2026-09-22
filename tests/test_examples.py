"""
test_examples.py
================
Smoke-tests for the two example generation scripts in examples/.

These tests exercise the full generation pipeline end-to-end using a tiny
randomly-initialised model (no checkpoints required).  They verify that:

  • generate_unconditional.py's decode path runs without error (random mode
    and input-seeded mode) using a tiny VAE saved to a temp checkpoint.
  • generate_conditional.py's decode path runs without error (random mode
    and input-seeded mode) using a tiny CVAE saved to a temp checkpoint.

Checkpoints are saved via pytorch_lightning Trainer.save_checkpoint() so
the PL format is correct and load_from_checkpoint() works.

All tests are marked @pytest.mark.slow because they run a Trainer to create
the checkpoint files.
"""

import sys
import csv
from pathlib import Path

import pytest
import pandas as pd
from tests.conftest import TINY_VAE_PARAMS, TINY_CVAE_PARAMS

# Tiny synthetic SELFIES CSV used in place of the real MOSES dataset
_TINY_SELFIES = [
    "[C][C][=O][O]", "[C][Branch1][C][F][C][C]", "[O][=C][C][=C][C][=C][Ring1][=A]",
    "[N][C][C][=O][O]", "[C][C][C][C][C][C]", "[C][=C][C][=C][C][=C][Ring1][=A]",
    "[C][O][C]", "[N][=C][N][C][=O]",
]

REPO_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Helpers, build tiny PL checkpoints using fast_dev_run
# ---------------------------------------------------------------------------

def _make_vae_ckpt(tmp_path: Path, vocab_file: str) -> Path:
    """
    Run fast_dev_run=1 and save a proper PL checkpoint for VAELightningModule.
    """
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import TensorBoardLogger
    from bmfm_sm_gen.trainers.lm_vae import VAELightningModule
    from bmfm_sm_gen.data.dm_mol import MolecularDataModule

    model = VAELightningModule(
        model_class="bmfm_sm_gen.models.vae.TransformerVAE",
        model_params=dict(**TINY_VAE_PARAMS),
        tokenizer="bmfm_sm_gen.data.base_tokenizer.SELFIESTokenizer",
        vocab_file=vocab_file,
    )
    _csv = str(tmp_path / "tiny_train.csv")
    pd.DataFrame({"selfies": _TINY_SELFIES}).to_csv(_csv, index=False)
    dm = MolecularDataModule(
        data_dir="", train_mode="mem",
        train_csv=_csv, val_csv=_csv,
        allow_same_split=True,
        tokenizer="bmfm_sm_gen.data.base_tokenizer.SELFIESTokenizer",
        vocab_file=vocab_file,
        max_len=20, batch_size=2,
        dataset_args={"modality": "selfies", "padding": "max_length"},
    )
    trainer = pl.Trainer(
        fast_dev_run=1, accelerator="cpu",
        logger=TensorBoardLogger(save_dir=str(tmp_path), name="vae"),
        enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False,
    )
    trainer.fit(model, datamodule=dm)
    ckpt_path = tmp_path / "tiny_vae.ckpt"
    trainer.save_checkpoint(str(ckpt_path))
    return ckpt_path


def _make_cvae_ckpt(tmp_path: Path, vocab_file: str) -> Path:
    """
    Run fast_dev_run=1 and save a proper PL checkpoint for CVAELightningModule.
    """
    import pandas as pd
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import TensorBoardLogger
    from bmfm_sm_gen.trainers.lm_cvae import CVAELightningModule
    from bmfm_sm_gen.data.dm_mol import MolecularDataModule

    df = pd.DataFrame({"selfies": _TINY_SELFIES})
    df["y_normalized_minmax"] = 0.5
    csv_path = str(tmp_path / "tiny_label.csv")
    df.to_csv(csv_path, index=False)

    model = CVAELightningModule(
        model_class="bmfm_sm_gen.models.cvae.ConditionalTransformerVAE",
        model_params=dict(**TINY_CVAE_PARAMS),
        tokenizer="bmfm_sm_gen.data.base_tokenizer.SELFIESTokenizer",
        vocab_file=vocab_file,
        lora=None,
        use_conditional_prior=False,
    )
    dm = MolecularDataModule(
        data_dir="", train_mode="mem",
        train_csv=csv_path, val_csv=csv_path,
        allow_same_split=True,
        tokenizer="bmfm_sm_gen.data.base_tokenizer.SELFIESTokenizer",
        vocab_file=vocab_file,
        max_len=20, batch_size=2,
        dataset_args={
            "modality": "selfies", "padding": "max_length",
            "label_columns": ["y_normalized_minmax"],
        },
    )
    trainer = pl.Trainer(
        fast_dev_run=1, accelerator="cpu",
        logger=TensorBoardLogger(save_dir=str(tmp_path), name="cvae"),
        enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False,
    )
    trainer.fit(model, datamodule=dm)
    ckpt_path = tmp_path / "tiny_cvae.ckpt"
    trainer.save_checkpoint(str(ckpt_path))
    return ckpt_path


# ---------------------------------------------------------------------------
# Tests, generate_unconditional (in-process via sys.argv patch)
# ---------------------------------------------------------------------------

# Add examples/ to sys.path once at module load so all tests can import the scripts.
_examples_dir = str(REPO_ROOT / "examples")
if _examples_dir not in sys.path:
    sys.path.insert(0, _examples_dir)


def _run_unconditional(argv: list):
    """Import and call generate_unconditional.main() with patched sys.argv."""
    saved = sys.argv[:]
    sys.argv = argv
    try:
        sys.modules.pop("generate_unconditional", None)
        import generate_unconditional
        generate_unconditional.main()
    finally:
        sys.argv = saved


def _run_conditional(argv: list):
    """Import and call generate_conditional.main() with patched sys.argv."""
    saved = sys.argv[:]
    sys.argv = argv
    try:
        sys.modules.pop("generate_conditional", None)
        import generate_conditional
        generate_conditional.main()
    finally:
        sys.argv = saved


def test_conditional_csv_preserves_attempts_and_reports_counts(tmp_path, capsys):
    import generate_conditional

    output = tmp_path / "nested" / "attempts.csv"
    generate_conditional.save_csv(["C", None, "O"], str(output), 0.25)

    with output.open(newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows == [
        ["smiles", "target_label"],
        ["C", "0.25"],
        ["", "0.25"],
        ["O", "0.25"],
    ]
    assert "Saved 3 attempts (2 valid)" in capsys.readouterr().out


def test_unconditional_csv_preserves_attempts_and_reports_counts(tmp_path, capsys):
    import generate_unconditional

    output = tmp_path / "nested" / "attempts.csv"
    generate_unconditional.save_csv(["C", None, "O"], str(output))

    with output.open(newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows == [["smiles"], ["C"], [""], ["O"]]
    assert "Saved 3 attempts (2 valid)" in capsys.readouterr().out


def test_unconditional_example_imports_math_at_module_scope():
    import generate_unconditional

    assert generate_unconditional.math.isfinite(1.0)


@pytest.mark.slow
def test_generate_unconditional_random(tmp_path, vocab_file):
    """
    generate_unconditional.py main(), random mode, tiny checkpoint.
    """
    ckpt = _make_vae_ckpt(tmp_path, vocab_file)
    out_csv = str(tmp_path / "out.csv")

    _run_unconditional([
        "generate_unconditional.py",
        "--checkpoint", str(ckpt),
        "--vocab", vocab_file,
        "--num-samples", "4",
        "--seed", "0",
        "--device", "cpu",
        "--output", out_csv,
    ])

    import pandas as pd
    assert Path(out_csv).exists(), "Output CSV not created"
    df = pd.read_csv(out_csv)
    assert "smiles" in df.columns


@pytest.mark.slow
def test_generate_unconditional_seeded(tmp_path, vocab_file):
    """
    generate_unconditional.py main(), input-seeded mode, tiny checkpoint.
    """
    ckpt = _make_vae_ckpt(tmp_path, vocab_file)
    _run_unconditional([
        "generate_unconditional.py",
        "--checkpoint", str(ckpt),
        "--vocab", vocab_file,
        "--input-smiles", "CC",
        "--noise-scale", "0.3",
        "--num-samples", "4",
        "--seed", "0",
        "--device", "cpu",
    ])


# ---------------------------------------------------------------------------
# Tests, generate_conditional (in-process)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_generate_conditional_random(tmp_path, vocab_file):
    """
    generate_conditional.py main(), random mode, tiny CVAE checkpoint.
    """
    ckpt = _make_cvae_ckpt(tmp_path, vocab_file)
    out_csv = str(tmp_path / "out_cond.csv")

    _run_conditional([
        "generate_conditional.py",
        "--checkpoint", str(ckpt),
        "--vocab", vocab_file,
        "--target-label", "0.25",
        "--num-samples", "4",
        "--seed", "0",
        "--device", "cpu",
        "--output", out_csv,
    ])

    import pandas as pd
    assert Path(out_csv).exists(), "Output CSV not created"
    df = pd.read_csv(out_csv)
    assert "smiles" in df.columns
    assert "target_label" in df.columns


@pytest.mark.slow
def test_generate_conditional_seeded(tmp_path, vocab_file):
    """
    generate_conditional.py main(), input-seeded mode, tiny CVAE checkpoint.
    """
    ckpt = _make_cvae_ckpt(tmp_path, vocab_file)
    _run_conditional([
        "generate_conditional.py",
        "--checkpoint", str(ckpt),
        "--vocab", vocab_file,
        "--input-smiles", "CC",
        "--target-label", "0.5",
        "--noise-scale", "0.4",
        "--num-samples", "4",
        "--seed", "0",
        "--device", "cpu",
    ])


@pytest.mark.slow
def test_generate_conditional_boundary_labels(tmp_path, vocab_file):
    """Labels at 0.0 and 1.0 must not crash."""
    ckpt = _make_cvae_ckpt(tmp_path, vocab_file)
    for label in ["0.0", "1.0"]:
        _run_conditional([
            "generate_conditional.py",
            "--checkpoint", str(ckpt),
            "--vocab", vocab_file,
            "--target-label", label,
            "--num-samples", "2",
            "--seed", "0",
            "--device", "cpu",
        ])
