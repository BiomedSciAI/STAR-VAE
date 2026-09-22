"""
Shared pytest fixtures and helpers for the STAR-VAE test suite.

Default tests are CPU-only and self-contained. They create a test vocabulary
and a micro-scale model, so no dataset or checkpoint bundle is required.
Tests that need real checkpoints are marked with @pytest.mark.requires_ckpt
and use the CKPT_* environment variables documented in .env.example.
"""

import os
import json
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).parent.parent

# Checkpoint env-var names must match .env.example.
CKPT_PRETRAINED = os.environ.get("PRETRAINED_VAE_CKPT", "")
CKPT_SA_SCORE = os.environ.get("CKPT_SA_SCORE", "")
CKPT_1SYH = os.environ.get("CKPT_1SYH", "")
CKPT_4LDE = os.environ.get("CKPT_4LDE", "")
CKPT_6Y2F = os.environ.get("CKPT_6Y2F", "")

# ── Tiny model kwargs (CPU-friendly, no real checkpoint needed) ────────────────
TINY_VAE_PARAMS = dict(
    d_model=64,
    nhead=2,
    num_encoder_layers=2,
    num_decoder_layers=2,
    dim_feedforward=128,
    latent_dim=32,
    dropout=0.0,
    max_len=20,
)

TINY_CVAE_PARAMS = dict(
    d_model=64,
    nhead=2,
    num_encoder_layers=2,
    num_decoder_layers=2,
    dim_feedforward=128,
    latent_dim=32,
    dropout=0.0,
    max_len=20,
    num_tasks=1,
)


def pytest_collection_modifyitems(config, items):
    """Give opted-in checkpoint and artifact tests clear missing-input skips."""
    for item in items:
        if "requires_ckpt" in item.keywords:
            # Determine which ckpt the test needs from the marker argument,
            # defaulting to the pretrained backbone.
            marker = item.get_closest_marker("requires_ckpt")
            ckpt_var = marker.args[0] if marker.args else "PRETRAINED_VAE_CKPT"
            ckpt_path = os.environ.get(ckpt_var, "")
            if not ckpt_path or not Path(ckpt_path).exists():
                item.add_marker(
                    pytest.mark.skip(
                        reason=f"Checkpoint not found: {ckpt_var}={ckpt_path!r}. "
                        "Set the env-var and ensure the file exists."
                    )
                )
        if "external_artifact" in item.keywords:
            marker = item.get_closest_marker("external_artifact")
            paths = [REPO_ROOT / path for path in marker.args]
            missing = [str(path) for path in paths if not path.exists()]
            if missing:
                item.add_marker(
                    pytest.mark.skip(
                        reason=f"External artifact(s) not found: {missing}"
                    )
                )


@pytest.fixture(scope="session")
def vocab_file(tmp_path_factory):
    """Create a minimal SELFIES vocabulary independent of checkpoint assets."""
    path = tmp_path_factory.mktemp("vocab") / "vocab.json"
    vocab = {
        "<pad>": 0,
        "<sos>": 1,
        "<eos>": 2,
        "<unk>": 3,
        "[C]": 4,
        "[O]": 5,
        "[N]": 6,
        "[=C]": 7,
        "[=O]": 8,
        "[F]": 9,
        "[Branch1]": 10,
        "[Ring1]": 11,
        "[=Ring1]": 12,
    }
    path.write_text(json.dumps(vocab), encoding="utf-8")
    return str(path)


@pytest.fixture(scope="session")
def tiny_vae(vocab_file):
    """A tiny, randomly-initialised VAELightningModule on CPU."""
    from bmfm_sm_gen.trainers.lm_vae import VAELightningModule

    return VAELightningModule(
        model_class="bmfm_sm_gen.models.vae.TransformerVAE",
        model_params=dict(**TINY_VAE_PARAMS),
        tokenizer="bmfm_sm_gen.data.base_tokenizer.SELFIESTokenizer",
        vocab_file=vocab_file,
    )


@pytest.fixture(scope="session")
def tiny_cvae(vocab_file):
    """A tiny, randomly-initialised CVAELightningModule on CPU."""
    from bmfm_sm_gen.trainers.lm_cvae import CVAELightningModule

    return CVAELightningModule(
        model_class="bmfm_sm_gen.models.cvae.ConditionalTransformerVAE",
        model_params=dict(**TINY_CVAE_PARAMS),
        tokenizer="bmfm_sm_gen.data.base_tokenizer.SELFIESTokenizer",
        vocab_file=vocab_file,
        lora=None,
        use_conditional_prior=False,
    )


@pytest.fixture()
def tmp_output(tmp_path):
    """Isolated temp directory for a single test's output files."""
    return tmp_path
