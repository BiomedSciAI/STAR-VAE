"""
test_generation.py
==================
Smoke-tests for molecule generation (both unconditional and conditional).

Two tiers:

Tier 1, "tiny model" (always runs, CPU, no checkpoint):
  Verifies that VAELightningModule.generate_molecules() and
  CVAELightningModule.generate_molecules_with_target_label() run end-to-end
  with a randomly-initialised micro model and return non-empty lists.

Tier 2, "real checkpoint" (skipped unless env-vars are set):
  Loads each published checkpoint and generates 1 molecule, asserting that:
  - At least 1 valid SMILES is returned.
  - RDKit can parse the molecule (validity check).
  Requires: PRETRAINED_VAE_CKPT, CKPT_SA_SCORE, CKPT_1SYH, CKPT_4LDE,
            CKPT_6Y2F
"""

import os
from pathlib import Path

import pytest
import torch

from tests.conftest import (
    CKPT_PRETRAINED,
    CKPT_SA_SCORE,
    CKPT_1SYH,
    CKPT_4LDE,
    CKPT_6Y2F,
)

# ── helpers ───────────────────────────────────────────────────────────────────


def _is_valid_smiles(smi: str) -> bool:
    """Return True if RDKit can parse the SMILES string."""
    from rdkit import Chem

    if not smi:
        return False
    return Chem.MolFromSmiles(smi) is not None


def _ckpt_vocab():
    """Return the vocab file that matches the real checkpoints (PubChem-80M)."""
    vocab = os.environ.get("VOCAB_FILE", "")
    if vocab and Path(vocab).exists():
        return vocab
    pytest.skip("VOCAB_FILE must point to the vocabulary matching the checkpoint")


# ── Tier 1: tiny randomly-initialised models ──────────────────────────────────


@pytest.mark.slow
def test_vae_generate_returns_list(tiny_vae):
    """generate_molecules() returns a list of length == num_samples."""
    result = tiny_vae.generate_molecules(num_samples=3)
    assert isinstance(result, list)
    assert len(result) == 3


def test_cvae_generation_passes_requested_batch_to_decoder(tiny_cvae, monkeypatch):
    """Conditional generation creates one latent and label row per request."""
    captured = {}

    def capture(model, tokenizer, z, labels, *args):
        captured["z"] = z
        captured["labels"] = labels
        return ["C"] * z.shape[0]

    monkeypatch.setattr("bmfm_sm_gen.trainers.lm_cvae.generate_molecules_with_labels", capture)
    result = tiny_cvae.generate_molecules_with_target_label(2, target_label=0.5)
    assert result == ["C", "C"]
    assert captured["z"].shape == (2, tiny_cvae.model.latent_dim)
    assert torch.equal(captured["labels"], torch.full((2, 1), 0.5))


@pytest.mark.slow
def test_vae_generate_strings(tiny_vae):
    """Each generated item is a string or None (untrained model may return empty strings)."""
    result = tiny_vae.generate_molecules(num_samples=4)
    for item in result:
        assert item is None or isinstance(item, str), (
            f"Expected str or None, got {type(item)}: {item!r}"
        )


@pytest.mark.slow
def test_vae_temperature_effect(tiny_vae):
    """generate_molecules() accepts temperature parameter without error."""
    tiny_vae.generate_molecules(num_samples=2, temperature=0.8)
    tiny_vae.generate_molecules(num_samples=2, temperature=1.2)


@pytest.mark.slow
def test_vae_top_k_effect(tiny_vae):
    """generate_molecules() accepts top_k parameter without error."""
    tiny_vae.generate_molecules(num_samples=2, top_k=5)


def test_categorical_sampling_respects_top_k_and_seed():
    from bmfm_sm_gen.trainers.lm_vae import _sample_from_logits

    logits = torch.tensor([[0.0, 1.0, 2.0]])
    torch.manual_seed(7)
    samples1 = torch.cat([_sample_from_logits(logits, 1.0, 2) for _ in range(20)]).flatten()
    torch.manual_seed(7)
    samples2 = torch.cat([_sample_from_logits(logits, 1.0, 2) for _ in range(20)]).flatten()
    assert torch.equal(samples1, samples2)
    assert set(samples1.tolist()) <= {1, 2}
    assert len(set(samples1.tolist())) == 2


@pytest.mark.parametrize(
    "temperature,top_k,error",
    [(0.0, None, ValueError), (float("nan"), None, ValueError), (1.0, 0, ValueError),
     (1.0, 10_000, ValueError), (1.0, 1.5, TypeError)],
)
def test_sampling_argument_validation(tiny_vae, temperature, top_k, error):
    with pytest.raises(error):
        tiny_vae.generate_molecules(1, temperature=temperature, top_k=top_k)


def test_conditional_generation_uses_enabled_prior(tiny_cvae, monkeypatch):
    monkeypatch.setattr(tiny_cvae.model, "use_conditional_prior", True)
    expected = torch.full((2, tiny_cvae.model.latent_dim), 4.0)
    monkeypatch.setattr(
        tiny_cvae.model,
        "get_conditional_prior",
        lambda labels: (expected, torch.zeros_like(expected)),
    )
    monkeypatch.setattr(tiny_cvae.model, "reparameterize", lambda mu, logvar: mu)
    captured = {}

    def capture(model, tokenizer, z, labels, *args):
        captured["z"] = z
        return []

    monkeypatch.setattr("bmfm_sm_gen.trainers.lm_cvae.generate_molecules_with_labels", capture)
    tiny_cvae.generate_molecules_with_target_label(2, 0.5)
    assert torch.equal(captured["z"], expected)


def test_conditional_prior_mode_rejects_unavailable_prior(tiny_cvae):
    with pytest.raises(ValueError, match="no conditional prior"):
        tiny_cvae.generate_molecules_with_target_label(1, 0.5, prior_mode="conditional")


def test_conditional_generation_preserves_failed_attempts(monkeypatch):
    from bmfm_sm_gen.trainers import lm_cvae

    class Tokenizer:
        vocab_size = 4
        bos_token_id = 1
        eos_token_id = 2
        pad_token_id = 0

        @staticmethod
        def convert_ids_to_tokens(_):
            return "invalid"

        @staticmethod
        def convert_to_smiles(_):
            raise ValueError("decode failed")

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.d_model = 2
            self.embedding = torch.nn.Embedding(4, 2)
            self.pos_decoder = torch.nn.Identity()
            self.transformer_decoder = lambda tgt, memory, **kwargs: tgt
            self.output_layer = torch.nn.Linear(2, 4)

        @staticmethod
        def generate_square_subsequent_mask(size):
            return torch.zeros(size, size)

    class Model(torch.nn.Module):
        max_len = 2
        latent_dim = 2
        label_dim = 1
        modulation = None

        def __init__(self):
            super().__init__()
            self.decoder = Decoder()
            self.latent_to_hidden = torch.nn.Identity()

    monkeypatch.setattr(lm_cvae, "_sample_from_logits", lambda *args: torch.ones(3, 1, dtype=torch.long))
    result = lm_cvae.generate_molecules_with_labels(
        Model(), Tokenizer(), torch.zeros(3, 2), torch.zeros(3, 1)
    )
    assert result == [None, None, None]


def test_conditional_generation_automatically_uses_model_memory(monkeypatch):
    from bmfm_sm_gen.trainers import lm_cvae

    class Tokenizer:
        vocab_size = 4
        bos_token_id = 1
        eos_token_id = 2
        pad_token_id = 0

        @staticmethod
        def convert_ids_to_tokens(_):
            return ""

        @staticmethod
        def convert_to_smiles(_):
            return None

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.d_model = 2
            self.embedding = torch.nn.Embedding(4, 2)
            self.pos_decoder = torch.nn.Identity()
            self.transformer_decoder = lambda tgt, memory, **kwargs: tgt
            self.output_layer = torch.nn.Linear(2, 4)

        @staticmethod
        def generate_square_subsequent_mask(size):
            return torch.zeros(size, size)

    class Model(torch.nn.Module):
        max_len = 2
        latent_dim = 2
        label_dim = 1

        def __init__(self):
            super().__init__()
            self.decoder = Decoder()
            self.memory_calls = 0

        def _build_memory(self, z, labels):
            self.memory_calls += 1
            return torch.zeros(self.max_len + 1, z.size(0), 2)

    model = Model()
    monkeypatch.setattr(lm_cvae, "_sample_from_logits", lambda *args: torch.ones(2, 1, dtype=torch.long))
    result = lm_cvae.generate_molecules_with_labels(
        model, Tokenizer(), torch.zeros(2, 2), torch.zeros(2, 1)
    )
    assert model.memory_calls == 1
    assert result == [None, None]


# ── Tier 2: real checkpoints ──────────────────────────────────────────────────


@pytest.mark.requires_ckpt("PRETRAINED_VAE_CKPT")
def test_pretrained_vae_generates_valid_smiles():
    """Pretrained VAE checkpoint produces ≥1 valid molecule out of 10 samples."""
    import torch
    from bmfm_sm_gen.trainers.lm_vae import VAELightningModule

    vocab = _ckpt_vocab()
    torch.manual_seed(42)
    model = VAELightningModule.load_from_checkpoint(
        CKPT_PRETRAINED,
        vocab_file=vocab,
        map_location=torch.device("cpu"),
    )
    model.eval()
    smiles = model.generate_molecules(num_samples=10, temperature=1.0)
    smiles = [s for s in smiles if s]
    valid = [s for s in smiles if _is_valid_smiles(s)]
    assert len(valid) >= 1, (
        f"Expected ≥1 valid molecule; got {len(valid)}/{len(smiles)} valid."
    )


@pytest.mark.requires_ckpt("CKPT_SA_SCORE")
def test_sa_score_cvae_generates_valid_smiles():
    """SA-score CVAE checkpoint generates ≥1 valid molecule conditioned on target=0.3."""
    import torch
    from bmfm_sm_gen.trainers.lm_cvae import CVAELightningModule

    vocab = _ckpt_vocab()
    torch.manual_seed(42)
    model = CVAELightningModule.load_from_checkpoint(
        CKPT_SA_SCORE,
        vocab_file=vocab,
        map_location=torch.device("cpu"),
    )
    model.eval()
    smiles = model.generate_molecules_with_target_label(
        num_samples=5, target_label=0.3
    )
    smiles = [s for s in smiles if s]
    valid = [s for s in smiles if _is_valid_smiles(s)]
    assert len(valid) >= 1, (
        f"SA-score CVAE: expected ≥1 valid molecule; got {len(valid)}/{len(smiles)}."
    )


@pytest.mark.parametrize(
    "ckpt_env,target",
    [
        pytest.param("CKPT_1SYH", 0.2, marks=pytest.mark.requires_ckpt("CKPT_1SYH")),
        pytest.param("CKPT_4LDE", 0.2, marks=pytest.mark.requires_ckpt("CKPT_4LDE")),
        pytest.param("CKPT_6Y2F", 0.2, marks=pytest.mark.requires_ckpt("CKPT_6Y2F")),
    ],
)
def test_tartarus_cvae_generates_valid_smiles(ckpt_env, target):
    """Each Tartarus CVAE checkpoint produces ≥1 valid molecule."""
    import torch
    from bmfm_sm_gen.trainers.lm_cvae import CVAELightningModule

    ckpt_path = os.environ.get(ckpt_env, "")
    if not ckpt_path or not Path(ckpt_path).exists():
        pytest.skip(f"{ckpt_env} not set or file not found")

    vocab = _ckpt_vocab()
    torch.manual_seed(42)
    model = CVAELightningModule.load_from_checkpoint(
        ckpt_path,
        vocab_file=vocab,
        map_location=torch.device("cpu"),
    )
    model.eval()
    smiles = model.generate_molecules_with_target_label(
        num_samples=5, target_label=target
    )
    smiles = [s for s in smiles if s]
    valid = [s for s in smiles if _is_valid_smiles(s)]
    assert len(valid) >= 1, (
        f"{ckpt_env}: expected ≥1 valid molecule; got {len(valid)}/{len(smiles)}."
    )


@pytest.mark.parametrize(
    "ckpt_env,target",
    [
        pytest.param("CKPT_1SYH_STRAT", 0.2, marks=pytest.mark.requires_ckpt("CKPT_1SYH_STRAT")),
        pytest.param("CKPT_4LDE_STRAT", 0.2, marks=pytest.mark.requires_ckpt("CKPT_4LDE_STRAT")),
        pytest.param("CKPT_6Y2F_STRAT", 0.2, marks=pytest.mark.requires_ckpt("CKPT_6Y2F_STRAT")),
        pytest.param("CKPT_B3DB", 0.5, marks=pytest.mark.requires_ckpt("CKPT_B3DB")),
    ],
)
def test_additional_cvae_checkpoints_generate_valid_smiles(ckpt_env, target):
    """Each additional distributed CVAE produces at least one valid molecule."""
    from bmfm_sm_gen.trainers.lm_cvae import CVAELightningModule

    ckpt_path = os.environ.get(ckpt_env, "")
    if not ckpt_path or not Path(ckpt_path).exists():
        pytest.skip(f"{ckpt_env} not set or file not found")
    torch.manual_seed(42)
    model = CVAELightningModule.load_from_checkpoint(
        ckpt_path, vocab_file=_ckpt_vocab(), map_location=torch.device("cpu")
    )
    model.eval()
    smiles = model.generate_molecules_with_target_label(
        num_samples=5, target_label=target
    )
    valid = [s for s in smiles if s and _is_valid_smiles(s)]
    assert valid, f"{ckpt_env}: no valid molecules from {len(smiles)} attempts"


@pytest.mark.requires_ckpt("PRETRAINED_VAE_CKPT")
def test_distributed_checkpoint_bundle_loads_expected_models():
    """Load every supported distributed checkpoint with exact state coverage."""
    import gc
    import json
    from bmfm_sm_gen.trainers.lm_vae import VAELightningModule
    from bmfm_sm_gen.trainers.lm_cvae import CVAELightningModule

    root = Path(os.environ["PRETRAINED_VAE_CKPT"]).parents[1]
    vocab = _ckpt_vocab()
    manifest = json.loads((Path(__file__).resolve().parents[1] / "checkpoint_bundle.json").read_text())
    supported = []
    for name, entry in sorted(manifest["models"].items()):
        module_class = VAELightningModule if name.startswith("vae_") else CVAELightningModule
        supported.append((root / entry["checkpoint"], module_class))
    assert len(supported) == 20
    for checkpoint, module_class in supported:
        model = module_class.load_from_checkpoint(
            checkpoint, vocab_file=vocab, map_location=torch.device("cpu")
        )
        assert model.tokenizer.vocab_size == 328
        del model
        gc.collect()
