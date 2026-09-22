import json
import os
import re
from pathlib import Path

import selfies as sf
import torch
from transformers import PreTrainedTokenizer


class MolecularTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        vocab: dict[str, int] | str | Path | None = None,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        sos_token: str = "<sos>",
        eos_token: str = "<eos>",
        additional_special_tokens: list[str] | None = None,
    ):
        if isinstance(vocab, str | Path):
            with open(vocab) as f:
                vocab = json.load(f)

        self.vocab = vocab or {}
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token

        # If special tokens are not in vocab, add them
        for token in [unk_token, pad_token, sos_token, eos_token]:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

        # Add other special tokens if provided (needed for HF Compatibility)
        if additional_special_tokens:
            for token in additional_special_tokens:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

        self.id_to_token = {v: k for k, v in self.vocab.items()}

        # Register tokens with Hugging Face tokenizer class
        super().__init__(
            pad_token=pad_token,
            eos_token=eos_token,
            unk_token=unk_token,
            bos_token=sos_token,
            additional_special_tokens=additional_special_tokens or [],
        )

    def add_tokens(self, new_tokens: list[str]) -> int:
        """Adds new tokens to the tokenizer's vocabulary."""
        num_added_tokens = 0
        for token in new_tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                num_added_tokens += 1
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        return num_added_tokens

    # Example for SELFIES tokenizer could be --  sf.split_selfies(selfies_str)
    def _tokenize(self, text: str) -> list[str]:
        raise NotImplementedError("Subclasses must implement this method.")

    def _encode_token_ids(
        self, text: str, add_special_tokens: bool = True
    ) -> list[int]:
        tokens = self.tokenize(text)
        token_ids = self.convert_tokens_to_ids(tokens)
        if add_special_tokens:
            return (
                [self.vocab[self.sos_token]] + token_ids + [self.vocab[self.eos_token]]
            )
        return token_ids

    def _truncate(self, token_ids: list[int], max_length: int) -> list[int]:
        if max_length < 0:
            raise ValueError("max_length must be non-negative")
        if len(token_ids) <= max_length:
            return token_ids
        truncated = token_ids[:max_length]
        if max_length and token_ids[-1] == self.eos_token_id:
            truncated[-1] = self.eos_token_id
        return truncated

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        padding: bool | str = False,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> dict:
        token_ids = self._encode_token_ids(text, add_special_tokens)

        if truncation and max_length is not None:
            token_ids = self._truncate(token_ids, max_length)

        if padding == True or padding == "max_length":
            if max_length is None:
                raise ValueError("max_length must be set when padding is not 'longest'")
            pad_id = self.vocab[self.pad_token]
            token_ids = token_ids + [pad_id] * (max_length - len(token_ids))

        attention_mask = [
            1 if tok != self.vocab[self.pad_token] else 0 for tok in token_ids
        ]
        return {
            "input_ids": token_ids,
            "attention_mask": attention_mask,
            "token_type_ids": [0] * len(token_ids),
        }

    def batch_encode(
        self,
        texts: list[str],
        add_special_tokens: bool = True,
        padding: bool | str = False,  # True/"max_length" or "longest"
        truncation: bool = False,
        max_length: int | None = None,
    ) -> dict:
        batch = [self._encode_token_ids(t, add_special_tokens) for t in texts]

        if truncation and max_length is not None:
            batch = [self._truncate(seq, max_length) for seq in batch]

        if padding is not False:
            pad_id = self.vocab[self.pad_token]
            if padding == "longest":
                pad_len = max(len(seq) for seq in batch)
            else:  # "max_length" or True
                if max_length is None:
                    raise ValueError(
                        "max_length must be set when padding is not 'longest'"
                    )
                pad_len = max_length

            batch = [seq + [pad_id] * (pad_len - len(seq)) for seq in batch]

        attention_masks = [
            [1 if tok != self.vocab[self.pad_token] else 0 for tok in seq]
            for seq in batch
        ]
        token_type_ids = [[0] * len(seq) for seq in batch]

        return {
            "input_ids": batch,
            "attention_mask": attention_masks,
            "token_type_ids": token_type_ids,
        }

    def __call__(
        self,
        text: str | list[str],
        add_special_tokens: bool = True,
        padding: bool | str = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
        **kwargs,
    ):
        if isinstance(text, str):
            output = self.encode(
                text,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
            )
        else:
            output = self.batch_encode(
                text,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
            )

        if return_tensors == "pt":
            return {k: torch.tensor(v) for k, v in output.items()}
        return output

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
        end_on_eos: bool = False,
        **kwargs,
    ) -> str:
        tokens = self.convert_ids_to_tokens(token_ids)
        decoded_tokens = []
        for t in tokens:
            if skip_special_tokens and t in {
                self.pad_token,
                self.sos_token,
                self.unk_token,
            }:
                continue
            if t == self.eos_token:
                if not skip_special_tokens:
                    decoded_tokens.append(t)
                if end_on_eos:
                    break
                continue
            decoded_tokens.append(t)

        return self.detokenize(decoded_tokens)

    def batch_decode(self, token_ids, **kwargs):
        kwargs.setdefault("skip_special_tokens", True)
        return super().batch_decode(token_ids, **kwargs)

    # This would work for SELFIES and SMILES but other tokenizers could override
    def detokenize(self, tokens: list[str]) -> str:
        return "".join(tokens)

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab[self.unk_token])

    def _convert_id_to_token(self, index):
        return self.id_to_token.get(index, self.unk_token)

    def get_vocab(self) -> dict[str, int]:
        return self.vocab

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def save_pretrained(self, save_directory: str):
        """Saves the tokenizer to the specified directory."""
        os.makedirs(save_directory, exist_ok=True)

        # Save vocab as a JSON
        vocab_file = os.path.join(save_directory, "vocab.json")
        with open(vocab_file, "w") as f:
            json.dump(self.vocab, f)

        # Save special tokens
        special_tokens = {
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "sos_token": self.sos_token,
            "eos_token": self.eos_token,
        }
        special_tokens_file = os.path.join(save_directory, "special_tokens.json")
        with open(special_tokens_file, "w") as f:
            json.dump(special_tokens, f)

    @classmethod
    def from_pretrained(cls, pretrained_directory: str):
        """Loads a tokenizer from the specified directory."""
        vocab_file = os.path.join(pretrained_directory, "vocab.json")
        special_tokens_file = os.path.join(pretrained_directory, "special_tokens.json")

        with open(vocab_file) as f:
            vocab = json.load(f)

        with open(special_tokens_file) as f:
            special_tokens = json.load(f)

        tokenizer = cls(
            vocab=vocab,
            unk_token=special_tokens["unk_token"],
            pad_token=special_tokens["pad_token"],
            sos_token=special_tokens["sos_token"],
            eos_token=special_tokens["eos_token"],
        )
        return tokenizer


#########################################################################################################


class SELFIESTokenizer(MolecularTokenizer):
    def _tokenize(self, text: str):
        return list(sf.split_selfies(text))

    def convert_to_smiles(self, selfies_string):
        try:
            smiles = sf.decoder(selfies_string)
        except Exception:
            smiles = None
        return smiles


class SMILESTokenizer(MolecularTokenizer):
    """
    Character-level SMILES tokenizer.

    Atoms in brackets (e.g. ``[NH]``, ``[C@@H]``) are treated as single
    tokens, two-character atoms ``Cl`` / ``Br`` are kept together, and all
    remaining SMILES characters are individual tokens.  This mirrors the
    granularity of ``SELFIESTokenizer`` at the atom/bond level.

    Vocabulary must be built from a training corpus and passed as a
    ``vocab`` dict or path to a JSON file.  Use
    ``bmfm_sm_gen/scripts/exp19_build_smiles_vocab.py`` to generate it.
    """

    # Matches: bracket atoms · Br/Cl (2-char) · B/C/N/O/S/P/F/I/b/c/n/o/s/p
    #          aromatic atoms · bonds · ring closures · stereo / charge symbols
    _ATOM_RE = re.compile(
        r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p"
        r"|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$"
        r"|\%[0-9]{2}|[0-9])"
    )

    def _tokenize(self, text: str) -> list[str]:
        tokens = []
        position = 0
        for match in self._ATOM_RE.finditer(text):
            if match.start() != position:
                unmatched = text[position:match.start()]
                raise ValueError(
                    f"Unrecognized SMILES span {unmatched!r} at position {position}"
                )
            tokens.append(match.group(0))
            position = match.end()
        if position != len(text):
            raise ValueError(
                f"Unrecognized SMILES span {text[position:]!r} at position {position}"
            )
        return tokens

    def convert_to_smiles(self, smiles_string: str) -> str:
        """Identity, the token sequence already IS a SMILES string."""
        return smiles_string


class SAFETokenizer(MolecularTokenizer):
    @staticmethod
    def _safe():
        try:
            import safe
        except ImportError as exc:
            raise ImportError(
                "SAFETokenizer requires the optional 'safe-mol' package. "
                "Install SAFE support before tokenizing or decoding SAFE strings."
            ) from exc
        return safe

    def __init__(
        self,
        vocab: dict[str, int] | str | Path | None = None,
        unk_token: str = "[UNK]",
        pad_token: str = "[PAD]",
        sos_token: str = "[CLS]",
        eos_token: str = "[SEP]",
        additional_special_tokens: list[str] | None = None,
    ):
        super().__init__(
            vocab, unk_token, pad_token, sos_token, eos_token, additional_special_tokens
        )

    def _tokenize(self, text: str):
        return self._safe().split(text)

    def convert_to_smiles(self, safes_string):
        try:
            smiles = self._safe().decode(safes_string)
        except ImportError:
            raise
        except Exception:
            smiles = None
        return smiles
