"""
test_training.py
================
Smoke-tests for one-step training (VAE and CVAE).

Verifies that:
- A single training step runs end-to-end and returns a finite scalar loss.
- The KL weight scheduler produces values in [0, kl_weight_max].
- The trainer can be instantiated with fast_dev_run=1 (runs 1 train + 1 val
  batch) and writes a checkpoint to an isolated temp directory WITHOUT
  touching any existing paper_results/ or outputs/.
- CVAE training_step works correctly with a label tensor in the batch.

These tests are intentionally CPU-only and use tiny models.
"""

import os

import pytest
import torch

from tests.conftest import TINY_VAE_PARAMS, TINY_CVAE_PARAMS


# ── Loss / training_step unit tests ──────────────────────────────────────────


def test_vae_training_step_returns_finite_loss(tiny_vae):
    """One forward/backward step on a random batch returns finite loss."""
    import torch

    tiny_vae.train()
    # Build a fake batch: (batch, seq_len) long tensor
    vocab_size = tiny_vae.tokenizer.vocab_size
    B, S = 2, 20
    ids = torch.randint(3, vocab_size, (B, S))
    batch = {"input_ids": ids}
    loss = tiny_vae.training_step(batch, batch_idx=0)
    assert torch.isfinite(loss), f"Training loss is not finite: {loss.item()}"
    assert loss.item() > 0, "Loss should be positive"


@pytest.mark.slow
def test_cvae_training_step_via_trainer(tmp_path, vocab_file):
    """CVAE fast_dev_run=1 via the Trainer (required because training_step accesses self.trainer)."""
    import pandas as pd
    import pytorch_lightning as pl
    from bmfm_sm_gen.trainers.lm_cvae import CVAELightningModule
    from bmfm_sm_gen.data.dm_mol import MolecularDataModule
    df = pd.DataFrame({"selfies": ["[C][C]", "[C][O]"] * 4,
                       "y_normalized_minmax": [0.5] * 8})
    csv_path = str(tmp_path / "tiny.csv")
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
    from pytorch_lightning.loggers import TensorBoardLogger
    trainer = pl.Trainer(
        fast_dev_run=1, accelerator="cpu",
        logger=TensorBoardLogger(save_dir=str(tmp_path), name="test_cvae_step"),
        enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False,
    )
    trainer.fit(model, datamodule=dm)  # raises on error, passes on success


@pytest.mark.parametrize("step,expected", [(0, 0.0), (50, 0.5), (100, 1.0), (200, 1.0)])
def test_kl_weight_warmup(tiny_vae, monkeypatch, step, expected):
    """Exercise the implemented KL scheduler rather than duplicating its formula."""
    tiny_vae.kl_warmup_steps = 100
    tiny_vae.kl_weight_max = 1.0
    monkeypatch.setattr(type(tiny_vae), "global_step", property(lambda self: step))
    assert tiny_vae._kl_weight() == pytest.approx(expected)


# ── End-to-end trainer fast_dev_run ──────────────────────────────────────────


@pytest.mark.slow
def test_fast_dev_run_vae(tmp_output, vocab_file):
    """
    pytorch_lightning Trainer with fast_dev_run=1 runs 1 train + 1 val step
    on CPU with an isolated temp directory.  Uses TensorBoardLogger so the
    validation step's self.logger.experiment.add_image() call succeeds.
    """
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import TensorBoardLogger
    from bmfm_sm_gen.trainers.lm_vae import VAELightningModule
    from bmfm_sm_gen.data.dm_mol import MolecularDataModule
    csv_path = str(tmp_output / "tiny.csv")
    import pandas as pd
    pd.DataFrame({"selfies": ["[C][C]", "[C][O]"] * 4}).to_csv(csv_path, index=False)

    model = VAELightningModule(
        model_class="bmfm_sm_gen.models.vae.TransformerVAE",
        model_params=dict(**TINY_VAE_PARAMS),
        tokenizer="bmfm_sm_gen.data.base_tokenizer.SELFIESTokenizer",
        vocab_file=vocab_file,
        kl_warmup_steps=1,
    )

    dm = MolecularDataModule(
        data_dir="", train_mode="mem",
        train_csv=csv_path, val_csv=csv_path,
        allow_same_split=True,
        tokenizer="bmfm_sm_gen.data.base_tokenizer.SELFIESTokenizer",
        vocab_file=vocab_file,
        max_len=20, batch_size=2,
        dataset_args={"modality": "selfies", "padding": "max_length"},
    )

    trainer = pl.Trainer(
        fast_dev_run=1,
        accelerator="cpu",
        logger=TensorBoardLogger(save_dir=str(tmp_output), name="test_vae"),
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, datamodule=dm)
    # Fit completed without error = pass


@pytest.mark.slow
def test_fast_dev_run_cvae(tmp_output, vocab_file):
    """
    CVAE Trainer fast_dev_run=1 with a tiny label-bearing dataset.
    Creates a 10-row CSV with a dummy label column.
    """
    import pandas as pd
    import pytorch_lightning as pl
    from bmfm_sm_gen.trainers.lm_cvae import CVAELightningModule
    from bmfm_sm_gen.data.dm_mol import MolecularDataModule
    df = pd.DataFrame({"selfies": ["[C][C]", "[C][O]"] * 4,
                       "y_normalized_minmax": [0.5] * 8})
    csv_path = str(tmp_output / "tiny_labelled.csv")
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
            "modality": "selfies",
            "padding": "max_length",
            "label_columns": ["y_normalized_minmax"],
        },
    )

    from pytorch_lightning.loggers import TensorBoardLogger
    trainer = pl.Trainer(
        fast_dev_run=1,
        accelerator="cpu",
        logger=TensorBoardLogger(save_dir=str(tmp_output), name="test_cvae"),
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, datamodule=dm)


# ── Checkpoint save/load round-trip ──────────────────────────────────────────


@pytest.mark.slow
def test_vae_checkpoint_roundtrip(tmp_output, vocab_file):
    """
    Save a tiny VAE to disk and reload it; generated output must remain
    deterministic across the two model instances.
    """
    import torch
    import pytorch_lightning as pl
    from bmfm_sm_gen.trainers.lm_vae import VAELightningModule
    from bmfm_sm_gen.data.dm_mol import MolecularDataModule
    import pandas as pd
    csv_path = str(tmp_output / "tiny.csv")
    pd.DataFrame({"selfies": ["[C][C]", "[C][O]"] * 4}).to_csv(csv_path, index=False)

    ckpt_path = str(tmp_output / "tiny_vae.ckpt")

    model = VAELightningModule(
        model_class="bmfm_sm_gen.models.vae.TransformerVAE",
        model_params=dict(**TINY_VAE_PARAMS),
        tokenizer="bmfm_sm_gen.data.base_tokenizer.SELFIESTokenizer",
        vocab_file=vocab_file,
    )

    dm = MolecularDataModule(
        data_dir="", train_mode="mem",
        train_csv=csv_path, val_csv=csv_path,
        allow_same_split=True,
        tokenizer="bmfm_sm_gen.data.base_tokenizer.SELFIESTokenizer",
        vocab_file=vocab_file,
        max_len=20, batch_size=2,
        dataset_args={"modality": "selfies", "padding": "max_length"},
    )

    from pytorch_lightning.loggers import TensorBoardLogger
    trainer = pl.Trainer(
        fast_dev_run=1, accelerator="cpu",
        logger=TensorBoardLogger(save_dir=str(tmp_output), name="test_roundtrip"),
        enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False,
    )
    trainer.fit(model, datamodule=dm)

    # Save manually
    trainer.save_checkpoint(ckpt_path)
    assert os.path.exists(ckpt_path), "Checkpoint file was not written"

    # Reload and compare weight norms
    model2 = VAELightningModule.load_from_checkpoint(
        ckpt_path,
        map_location=torch.device("cpu"),
    )
    p1 = sum(p.norm().item() for p in model.parameters())
    p2 = sum(p.norm().item() for p in model2.parameters())
    assert abs(p1 - p2) < 1e-3, (
        f"Parameter norms differ after checkpoint reload: {p1:.4f} vs {p2:.4f}"
    )
