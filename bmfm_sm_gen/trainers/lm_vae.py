import importlib
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import selfies as sf
import torch
import torch.nn as nn
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw
import random

from bmfm_sm_gen.utils.trainer_utils import import_class


def _validate_sampling_args(temperature, top_k, vocab_size):
    if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a finite number greater than 0")
    if top_k is not None:
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer or None")
        if not 1 <= top_k <= vocab_size:
            raise ValueError(f"top_k must be between 1 and vocab size ({vocab_size})")


def _sample_from_logits(logits, temperature=1.0, top_k=None):
    """Draw one categorical token per row after temperature and top-k filtering."""
    _validate_sampling_args(temperature, top_k, logits.size(-1))
    logits = logits / temperature
    if top_k is not None:
        values, indices = torch.topk(logits, top_k, dim=-1)
        filtered = torch.full_like(logits, -torch.inf)
        logits = filtered.scatter(-1, indices, values)
    return torch.multinomial(torch.softmax(logits, dim=-1), 1)


class VAELightningModule(pl.LightningModule):
    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path,
        map_location=None,
        hparams_file=None,
        strict=True,
        **kwargs,
    ):
        """Load released checkpoints without replaying stale filesystem paths."""
        checkpoint = torch.load(checkpoint_path, map_location=map_location or "cpu")
        state = checkpoint.get("state_dict", checkpoint)
        hparams = dict(checkpoint.get("hyper_parameters") or {})
        if hparams_file:
            from omegaconf import OmegaConf

            hparams.update(OmegaConf.to_container(OmegaConf.load(hparams_file), resolve=True))
        model_params = dict(hparams.get("model_params") or {})

        def shape(key):
            return tuple(state[key].shape)

        embedding_key = "model.encoder.embedding.weight"
        pe_key = "model.encoder.pos_encoder.pe"
        if embedding_key not in state or pe_key not in state:
            raise ValueError(f"Unsupported checkpoint structure: {checkpoint_path}")

        d_model = shape(embedding_key)[1]
        encoder_layers = {
            int(key.split(".layers.", 1)[1].split(".", 1)[0])
            for key in state
            if key.startswith("model.encoder.transformer_encoder.layers.")
        }
        decoder_layers = {
            int(key.split(".layers.", 1)[1].split(".", 1)[0])
            for key in state
            if key.startswith("model.decoder.transformer_decoder.layers.")
        }
        nhead = model_params.get("nhead")
        if nhead is None:
            raise ValueError(
                "Checkpoint does not record nhead. Use a sanitized STAR-VAE "
                "release checkpoint or provide a compatible hparams file."
            )
        inferred = {
            "vocab_size": shape(embedding_key)[0],
            "max_len": shape(pe_key)[1] - 1,
            "d_model": d_model,
            "nhead": nhead,
            "num_encoder_layers": max(encoder_layers) + 1,
            "num_decoder_layers": max(decoder_layers) + 1,
            "dim_feedforward": shape(
                "model.encoder.transformer_encoder.layers.0.linear1.weight"
            )[0],
            "latent_dim": shape("model.encoder_fc_mu.weight")[0],
            "dropout": model_params.get("dropout", 0.1),
            "use_vae": model_params.get("use_vae", True),
        }
        inferred.update({k: v for k, v in model_params.items() if k in inferred})
        # Positional-encoding shape is authoritative for historical checkpoints
        # whose saved model_params were mutated after construction.
        inferred["max_len"] = shape(pe_key)[1] - 1
        inferred.pop("vocab_size", None)
        inferred.pop("padding_idx", None)

        vocab_file = kwargs.pop("vocab_file", None) or hparams.get("vocab_file")
        if not vocab_file:
            raise ValueError("vocab_file is required to load this checkpoint")
        with open(vocab_file) as handle:
            vocab_size = len(json.load(handle))
        expected_vocab_size = shape(embedding_key)[0]
        if vocab_size != expected_vocab_size:
            raise ValueError(
                f"Vocabulary has {vocab_size} tokens but checkpoint expects "
                f"{expected_vocab_size}: {vocab_file}"
            )
        expected_vocab_hash = (checkpoint.get("star_vae_release") or {}).get(
            "vocabulary_sha256"
        )
        if expected_vocab_hash:
            digest = hashlib.sha256(Path(vocab_file).read_bytes()).hexdigest()
            if digest != expected_vocab_hash:
                raise ValueError(
                    f"Vocabulary SHA-256 mismatch for checkpoint: {vocab_file}"
                )

        tokenizer = kwargs.pop("tokenizer", None) or hparams.get("tokenizer")
        if not tokenizer or tokenizer.startswith("bmfm_sm_gen.api."):
            tokenizer = "bmfm_sm_gen.data.base_tokenizer.SELFIESTokenizer"
        model_class = kwargs.pop("model_class", None) or hparams.get(
            "model_class", "bmfm_sm_gen.models.vae.TransformerVAE"
        )
        init_kwargs = {
            key: value
            for key, value in hparams.items()
            if key not in {"model_class", "model_params", "tokenizer", "vocab_file"}
        }
        init_kwargs.update(kwargs)
        module = cls(
            model_class=model_class,
            model_params=inferred,
            tokenizer=tokenizer,
            vocab_file=vocab_file,
            **init_kwargs,
        )
        incompatible = module.load_state_dict(state, strict=False)
        if strict and (incompatible.missing_keys or incompatible.unexpected_keys):
            raise RuntimeError(
                "Checkpoint model state is incompatible: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        return module

    def __init__(
        self,
        model_class: str,
        model_params: dict,
        tokenizer: str,
        vocab_file: str,
        learning_rate: float = 1e-4,
        kl_warmup_steps: int = 10000,
        kl_weight_max: float = 1.0,
        **kwargs,
    ):
        super(VAELightningModule, self).__init__()
        # Snapshot model_params BEFORE any mutation so save_hyperparameters
        # stores the user-specified max_len (e.g. 71), not the post-increment
        # value (72).  load_from_checkpoint replays __init__ with saved hparams,
        # so storing 72 would cause a second +1 → model built with max_len=73 →
        # PE size mismatch against the checkpoint's state_dict (shape [1,72,512]).
        _orig_model_params = {**model_params}

        self.model_params = model_params
        self.learning_rate = learning_rate

        # Load tokenizer from vocab file path or class reference
        if isinstance(tokenizer, str):
            tokenizer = import_class(tokenizer)
        self.tokenizer = tokenizer(vocab=vocab_file)
        
        # Load vocabulary
        with open(vocab_file) as file:
            self.vocab = json.load(file)
        self.idx_to_token = {idx: token for token, idx in self.vocab.items()}
        self.token_to_idx = self.vocab
        self.padding_idx = self.vocab["<pad>"]
        self.sos_token = self.vocab["<sos>"]
        self.eos_token = self.vocab["<eos>"]

        # need to add additional details to model_params before instantiating model
        self.model_params["padding_idx"] = self.tokenizer.pad_token_id
        self.model_params["max_len"] = (
            self.model_params["max_len"] + 1
        )  # +1 to account for the <sos> token prepended during generation
        self.max_len = self.model_params["max_len"]
        self.model_params["vocab_size"] = self.tokenizer.vocab_size
        cls = import_class(model_class)
        self.model = cls(**self.model_params)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        self.validation_step_counter = 0
        self.kl_warmup_steps = kl_warmup_steps
        self.kl_weight_max = kl_weight_max
        # Use the pre-mutation snapshot so hparams["model_params"]["max_len"]
        # always reflects the config value (e.g. 71), not the internal +1 value.
        model_params = _orig_model_params
        self.save_hyperparameters(ignore=["model"])
        self.random_val_batch_id = 0

    def _kl_weight(self) -> float:
        """Linear KL warmup: weight rises from 0 to kl_weight_max over kl_warmup_steps."""
        if self.kl_warmup_steps <= 0:
            return self.kl_weight_max
        return min(self.kl_weight_max, self.global_step / max(1, self.kl_warmup_steps))

    def training_step(self, batch, batch_idx):
        batch = batch["input_ids"]
        src = batch[:, :-1]
        tgt = batch[:, 1:]
        output, mu, logvar = self.model(src, src)
        output = output.view(-1, output.size(-1))
        tgt = tgt.reshape(-1)
        recon_loss = self.loss_fn(output, tgt)
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        kl_weight = self._kl_weight()
        loss = recon_loss + kl_weight * kld_loss
        self.log("train_loss", loss)
        self.log("recon_loss", recon_loss)
        self.log("kl_divergence", kld_loss)
        self.log("kl_weight", kl_weight)
        return loss

    def validation_step(self, batch, batch_idx):
        self.validation_step_counter += 1
        batch = batch["input_ids"]
        src = batch[:, :-1]
        tgt = batch[:, 1:]
        output, mu, logvar = self.model(src, src)
        output = output.view(-1, output.size(-1))
        tgt = tgt.reshape(-1)
        recon_loss = self.loss_fn(output, tgt)
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kld_loss
        self.log("val_loss", loss)

        if batch_idx == self.random_val_batch_id:
            num_samples = 100
            generated_smiles = self.generate_molecules(num_samples)
            valid_smiles = [smi for smi in generated_smiles if smi is not None]
            num_generated = len(generated_smiles)
            num_valid = len(valid_smiles)
            validity = (num_valid / num_generated) * 100 if num_generated > 0 else 0.0

            # Compute diversity
            if num_valid > 1:
                fingerprints = [
                    Chem.RDKFingerprint(Chem.MolFromSmiles(smi)) for smi in valid_smiles
                ]
                total_distance = 0.0
                num_pairs = 0
                for i in range(num_valid):
                    for j in range(i + 1, num_valid):
                        sim = DataStructs.FingerprintSimilarity(
                            fingerprints[i], fingerprints[j]
                        )
                        dist = 1 - sim
                        total_distance += dist
                        num_pairs += 1
                diversity = (total_distance / num_pairs) * 100  # Percentage
            else:
                diversity = 0.0

            self.log("validity", validity)
            self.log("diversity", diversity)

            if valid_smiles:
                mols = [Chem.MolFromSmiles(smi) for smi in valid_smiles[:16]]
                mols = [mol for mol in mols if mol is not None]
                if mols:
                    img = Draw.MolsToGridImage(
                        mols, molsPerRow=4, subImgSize=(200, 200)
                    )
                    img_tensor = self.PIL_to_tensor(img)
                    self.logger.experiment.add_image(
                        "Generated Molecules", img_tensor, self.global_step
                    )
                    self.logger.experiment.flush()
            self.plot_real_data_samples(batch)

    def generate_molecules(self, num_samples, temperature=1.0, top_k=None):
        if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError("num_samples must be a positive integer")
        _validate_sampling_args(temperature, top_k, self.tokenizer.vocab_size)
        self.model.eval()
        generated_smiles = []
        with torch.no_grad():
            z = torch.randn(num_samples, self.model.latent_dim).to(self.device)
            generated_tokens = torch.full(
                (num_samples, 1), self.tokenizer.bos_token_id, dtype=torch.long
            ).to(self.device)
            for _ in range(self.max_len - 1):
                tgt = generated_tokens
                # Use decoder's embedding and positional encoding
                tgt_emb = self.model.decoder.embedding(tgt) * math.sqrt(
                    self.model.decoder.d_model
                )
                tgt_emb = self.model.decoder.pos_decoder(tgt_emb)
                tgt_emb = tgt_emb.transpose(0, 1)  # (tgt_len, batch_size, d_model)
                tgt_mask = self.model.decoder.generate_square_subsequent_mask(
                    tgt_emb.size(0)
                ).to(self.device)
                tgt_key_padding_mask = tgt == self.tokenizer.pad_token_id
                # Prepare memory from z
                z_hidden = self.model.latent_to_hidden(z)  # (batch_size, d_model)
                memory = (
                    z_hidden.unsqueeze(1)
                    .repeat(1, self.model.max_len, 1)
                    .transpose(0, 1)
                )  # (max_len, batch_size, d_model)
                output = self.model.decoder.transformer_decoder(
                    tgt_emb,
                    memory,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                )
                output = output.transpose(0, 1)  # (batch_size, seq_len, d_model)
                output = self.model.decoder.output_layer(
                    output[:, -1, :]
                )  # (batch_size, vocab_size)
                predicted = _sample_from_logits(output, temperature, top_k)
                generated_tokens = torch.cat((generated_tokens, predicted), dim=1)
                if (predicted == self.tokenizer.eos_token_id).all():
                    break
            generated_tokens = generated_tokens[:, 1:]  # Remove <sos> token
            # Decode with the configured tokenizer (SELFIES or SMILES).
            for token_seq in generated_tokens:
                tokens = []
                for idx in token_seq:
                    idx = idx.item()
                    if idx == self.tokenizer.eos_token_id:
                        break
                    if idx != self.tokenizer.pad_token_id:
                        tokens.append(self.tokenizer.convert_ids_to_tokens(idx))
                encoded = "".join(tokens)
                try:
                    smiles = self.tokenizer.convert_to_smiles(encoded)
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        generated_smiles.append(smiles)
                    else:
                        generated_smiles.append(None)
                except Exception:
                    generated_smiles.append(None)
        return generated_smiles

    def PIL_to_tensor(self, img):
        img = img.convert("RGB")
        img = np.array(img)
        img = torch.tensor(img).permute(2, 0, 1).float()  # Convert to (C, H, W)
        return img / 255.0

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def plot_real_data_samples(self, batch):
        mols = []
        input_ids = batch if isinstance(batch, torch.Tensor) else batch["input_ids"]
        for seq in input_ids:
            tokens = []
            for idx in seq:
                idx = idx.item()
                if (
                    idx == self.tokenizer.eos_token_id
                    or idx == self.tokenizer.pad_token_id
                ):
                    continue
                tokens.append(self.tokenizer.convert_ids_to_tokens(idx))
            selfies_str = "".join(tokens)
            try:
                smiles = sf.decoder(selfies_str)
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    mols.append(mol)
            except Exception:
                pass

        if mols:
            mols = mols[:16]  # Limit to 16 molecules
            img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(200, 200))
            img_tensor = self.PIL_to_tensor(img)
            self.logger.experiment.add_image(
                "Real Molecules", img_tensor, self.global_step
            )
