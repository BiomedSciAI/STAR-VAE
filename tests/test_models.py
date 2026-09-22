"""
test_models.py
==============
Unit tests for TransformerVAE and ConditionalTransformerVAE model architectures.

Tests (all CPU, no checkpoint required):
- Forward pass produces outputs of correct shapes.
- encode() returns (mu, logvar) with correct latent_dim.
- VAE reparameterization: z has same shape as mu.
- CVAE modulation: FiLM conditioning does not crash.
"""

import pytest
import torch

from tests.conftest import TINY_VAE_PARAMS, TINY_CVAE_PARAMS


@pytest.fixture(scope="module")
def vae_model(vocab_file):
    from bmfm_sm_gen.models.vae import TransformerVAE
    from bmfm_sm_gen.data.base_tokenizer import SELFIESTokenizer

    tok = SELFIESTokenizer(vocab=vocab_file)
    params = dict(**TINY_VAE_PARAMS)
    params["vocab_size"] = tok.vocab_size
    params["padding_idx"] = tok.pad_token_id
    params["max_len"] = params["max_len"] + 1  # matches lm_vae.py increment
    return TransformerVAE(**params)


@pytest.fixture(scope="module")
def cvae_model(vocab_file):
    from bmfm_sm_gen.models.cvae import ConditionalTransformerVAE
    from bmfm_sm_gen.data.base_tokenizer import SELFIESTokenizer

    tok = SELFIESTokenizer(vocab=vocab_file)
    params = dict(**TINY_CVAE_PARAMS)
    params["vocab_size"] = tok.vocab_size
    params["padding_idx"] = tok.pad_token_id
    params["max_len"] = params["max_len"] + 1
    return ConditionalTransformerVAE(**params)


def _make_batch(vocab_size, seq_len=10, batch_size=2):
    """Create a random integer token batch."""
    return torch.randint(3, vocab_size, (batch_size, seq_len))


def test_vae_forward_shapes(vae_model):
    tok_size = vae_model.decoder.embedding.num_embeddings
    src = _make_batch(tok_size)
    output, mu, logvar = vae_model(src, src)
    B, S = src.shape
    assert output.shape == (B, S, tok_size), f"output shape mismatch: {output.shape}"
    assert mu.shape == (B, TINY_VAE_PARAMS["latent_dim"])
    assert logvar.shape == (B, TINY_VAE_PARAMS["latent_dim"])


def test_vae_encode(vae_model):
    tok_size = vae_model.decoder.embedding.num_embeddings
    src = _make_batch(tok_size)
    mu, logvar = vae_model.encode(src)
    assert mu.shape[-1] == TINY_VAE_PARAMS["latent_dim"]
    assert logvar.shape == mu.shape


def test_vae_reparameterize(vae_model):
    tok_size = vae_model.decoder.embedding.num_embeddings
    src = _make_batch(tok_size)
    mu, logvar = vae_model.encode(src)
    z = vae_model.reparameterize(mu, logvar)
    assert z.shape == mu.shape


def test_cvae_forward_shapes(cvae_model):
    tok_size = cvae_model.decoder.embedding.num_embeddings
    src = _make_batch(tok_size)
    labels = torch.rand(src.shape[0], 1)
    output, mu, logvar = cvae_model(src, src, labels)
    B, S = src.shape
    assert output.shape == (B, S, tok_size)
    assert mu.shape == (B, TINY_CVAE_PARAMS["latent_dim"])


def test_film_without_lora_uses_labels_and_backpropagates(vocab_file):
    from bmfm_sm_gen.data.base_tokenizer import SELFIESTokenizer
    from bmfm_sm_gen.models.cvae import ConditionalTransformerVAE

    tok = SELFIESTokenizer(vocab=vocab_file)
    params = dict(**TINY_CVAE_PARAMS)
    params.update(
        vocab_size=tok.vocab_size,
        padding_idx=tok.pad_token_id,
        max_len=params["max_len"] + 1,
        modulation_scheme="film",
        lora=None,
        use_vae=False,
    )
    model = ConditionalTransformerVAE(**params).eval()
    src = _make_batch(tok.vocab_size)
    low = torch.zeros(src.shape[0], 1)
    high = torch.ones(src.shape[0], 1)

    low_output, low_mu, _ = model(src, src, low)
    high_output, high_mu, _ = model(src, src, high)
    assert not torch.allclose(low_mu, high_mu)
    assert not torch.allclose(low_output, high_output)

    high_output.sum().backward()
    assert model.label_to_dmodel.weight.grad is not None
    assert model.modulation.mlp[0].weight.grad is not None
    assert model.modulation.mlp[0].weight.grad.abs().sum() > 0


def test_no_conditioning_ablation_ignores_labels(vocab_file):
    from bmfm_sm_gen.data.base_tokenizer import SELFIESTokenizer
    from bmfm_sm_gen.models.cvae import ConditionalTransformerVAE

    tok = SELFIESTokenizer(vocab=vocab_file)
    params = dict(**TINY_CVAE_PARAMS)
    params.update(
        vocab_size=tok.vocab_size,
        padding_idx=tok.pad_token_id,
        max_len=params["max_len"] + 1,
        modulation_scheme="none",
        lora="both",
        use_vae=False,
    )
    model = ConditionalTransformerVAE(**params).eval()
    src = _make_batch(tok.vocab_size)
    output_a, mu_a, _ = model(src, src, torch.zeros(src.shape[0], 1))
    output_b, mu_b, _ = model(src, src, torch.ones(src.shape[0], 1))
    assert torch.allclose(mu_a, mu_b)
    assert torch.allclose(output_a, output_b)


def test_model_is_deterministic_eval(vae_model):
    """In eval mode with same z, decoder output must be identical across calls."""
    vae_model.eval()
    tok_size = vae_model.decoder.embedding.num_embeddings
    src = _make_batch(tok_size)
    with torch.no_grad():
        out1, mu1, _ = vae_model(src, src)
        out2, mu2, _ = vae_model(src, src)
    assert torch.allclose(mu1, mu2), "mu is non-deterministic in eval mode"


def test_vae_masked_pooling_ignores_trailing_padding(vae_model):
    vae_model.eval()
    token = 3
    short = torch.tensor([[token, token]])
    padded = torch.tensor([[token, token, vae_model.padding_idx, vae_model.padding_idx]])
    with torch.no_grad():
        short_mu, short_logvar = vae_model.encode(short)
        padded_mu, padded_logvar = vae_model.encode(padded)
    assert torch.allclose(short_mu, padded_mu, atol=1e-6)
    assert torch.allclose(short_logvar, padded_logvar, atol=1e-6)


def test_cvae_masked_pooling_ignores_trailing_padding(cvae_model):
    cvae_model.eval()
    labels = torch.tensor([[0.5]])
    short = torch.tensor([[3, 3]])
    padded = torch.tensor([[3, 3, cvae_model.padding_idx, cvae_model.padding_idx]])
    with torch.no_grad():
        short_mu, _ = cvae_model.encode_with_labels(short, labels)
        padded_mu, _ = cvae_model.encode_with_labels(padded, labels)
    assert torch.allclose(short_mu, padded_mu, atol=1e-6)


def test_masked_pooling_does_not_change_state_dict_keys(vae_model):
    assert not any("pool" in key for key in vae_model.state_dict())


def test_deep_film_masked_pooling_ignores_trailing_padding(vocab_file):
    from bmfm_sm_gen.data.base_tokenizer import SELFIESTokenizer
    from bmfm_sm_gen.models.cvae_deep_film import ConditionalDeepFiLMVAE

    tokenizer = SELFIESTokenizer(vocab=vocab_file)
    params = dict(**TINY_CVAE_PARAMS)
    params.update(
        vocab_size=tokenizer.vocab_size,
        padding_idx=tokenizer.pad_token_id,
        max_len=params["max_len"] + 1,
    )
    model = ConditionalDeepFiLMVAE(**params).eval()
    labels = torch.tensor([[0.5]])
    short = torch.tensor([[3, 3]])
    padded = torch.tensor([[3, 3, model.padding_idx, model.padding_idx]])
    with torch.no_grad():
        short_mu, short_logvar = model.encode_with_labels(short, labels)
        padded_mu, padded_logvar = model.encode_with_labels(padded, labels)
    assert torch.allclose(short_mu, padded_mu, atol=1e-6)
    assert torch.allclose(short_logvar, padded_logvar, atol=1e-6)
