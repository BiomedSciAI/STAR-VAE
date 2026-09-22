"""
test_tokenizer.py
=================
Unit tests for SELFIESTokenizer and MolecularTokenizer.

Verifies that:
- The MOSES vocabulary file loads correctly (28 tokens incl. special tokens).
- Encoding a known SELFIES string produces a sequence bounded by max_len.
- Decoding the encoded sequence recovers the original string.
- Special token IDs are sane (0 = pad, 1 = bos/sos, 2 = eos).
"""

import builtins
import pytest

def test_vocab_loads(vocab_file):
    from bmfm_sm_gen.data.base_tokenizer import SELFIESTokenizer

    tok = SELFIESTokenizer(vocab=vocab_file)
    assert tok.vocab_size == 13


def test_special_token_ids(vocab_file):
    from bmfm_sm_gen.data.base_tokenizer import SELFIESTokenizer

    tok = SELFIESTokenizer(vocab=vocab_file)
    assert tok.pad_token_id == 0
    assert tok.bos_token_id == 1
    assert tok.eos_token_id == 2


def test_encode_decode_roundtrip(vocab_file):
    """Encoding then decoding a simple SELFIES string should recover the original."""
    from bmfm_sm_gen.data.base_tokenizer import SELFIESTokenizer

    tok = SELFIESTokenizer(vocab=vocab_file)
    # Simple SELFIES for ethane, present in the MOSES vocab
    selfies = "[C][C]"
    enc = tok(selfies, return_tensors="pt", max_length=20, truncation=True)
    ids = enc["input_ids"].squeeze(0).tolist()
    assert tok.decode(ids, end_on_eos=True) == selfies


def test_max_length_truncation(vocab_file):
    from bmfm_sm_gen.data.base_tokenizer import SELFIESTokenizer

    tok = SELFIESTokenizer(vocab=vocab_file)
    long_selfies = "[C]" * 50
    enc = tok(long_selfies, return_tensors="pt", max_length=10, truncation=True)
    assert enc["input_ids"].shape[-1] <= 10
    assert enc["input_ids"][-1].item() == tok.eos_token_id


def test_batch_truncation_preserves_eos(vocab_file):
    from bmfm_sm_gen.data.base_tokenizer import SELFIESTokenizer

    tok = SELFIESTokenizer(vocab=vocab_file)
    enc = tok(["[C]" * 20, "[C]"], max_length=5, truncation=True)
    assert enc["input_ids"][0][-1] == tok.eos_token_id
    assert enc["input_ids"][1][-1] == tok.eos_token_id


def test_safe_dependency_is_lazy_and_clear(monkeypatch):
    from bmfm_sm_gen.data.base_tokenizer import SAFETokenizer

    real_import = builtins.__import__

    def reject_safe(name, *args, **kwargs):
        if name == "safe":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_safe)
    tok = SAFETokenizer(vocab={})
    with pytest.raises(ImportError, match="optional 'safe-mol' package"):
        tok.tokenize("C")


def test_smiles_tokenizer_rejects_unmatched_spans():
    from bmfm_sm_gen.data.base_tokenizer import SMILESTokenizer

    tok = SMILESTokenizer(vocab={"C": 0})
    with pytest.raises(ValueError, match="Unrecognized SMILES span"):
        tok.tokenize("C&C")
