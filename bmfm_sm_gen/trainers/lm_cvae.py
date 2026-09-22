import logging
import hashlib
import json
import os
from pathlib import Path
import torch
import torch.nn as nn
from typing import Tuple
from rdkit import Chem
import math
import selfies as sf

from bmfm_sm_gen.models.layers import add_lora_adapters, freeze_backbone_except_lora
from bmfm_sm_gen.trainers.lm_vae import (
    VAELightningModule,
    _sample_from_logits,
    _validate_sampling_args,
)
from bmfm_sm_gen.utils.conditional_evaluation import (
    prepare_validation_info,
    run_conditional_generation_evaluation_compute_prop,
    run_validation_distribution_match,
)
from bmfm_sm_gen.utils.trainer_utils import draw_labeled_molecule_grid

class CVAELightningModule(VAELightningModule):

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path,
        map_location=None,
        hparams_file=None,
        strict=True,
        **kwargs,
    ):
        """Load released CVAE checkpoints using state shapes as ground truth."""
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
            "num_tasks": shape("model.label_to_dmodel.weight")[1],
            "use_vae": model_params.get("use_vae", True),
        }
        inferred.update({k: v for k, v in model_params.items() if k in inferred})
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
        lora = kwargs.pop("lora", hparams.get("lora"))
        if lora is None and any("lora_" in key for key in state):
            encoder_lora = any(
                key.startswith("model.encoder.") and "lora_" in key for key in state
            )
            decoder_lora = any(
                key.startswith("model.decoder.") and "lora_" in key for key in state
            )
            lora = "both" if encoder_lora and decoder_lora else (
                "encoder_only" if encoder_lora else "decoder_only"
            )
        lora_a = next((value for key, value in state.items() if key.endswith("lora_A")), None)
        use_conditional_prior = any("conditional_prior_mu" in key for key in state)
        modulation_scheme = "film" if any("model.modulation." in key for key in state) else "none"
        use_property_prediction = any("model.property_predictor." in key for key in state)

        model_class = kwargs.pop("model_class", None) or hparams.get(
            "model_class", "bmfm_sm_gen.models.cvae.ConditionalTransformerVAE"
        )
        supported_init_keys = {
            "learning_rate", "lora", "lora_r", "lora_alpha", "lora_dropout",
            "use_conditional_prior", "modulation_scheme", "label_norm",
            "num_bins", "samples_per_bin", "use_cyclical_annealing", "n_cycles",
            "ratio", "free_bits", "beta_schedule", "fixed_beta",
            "use_property_prediction", "lambda_prop", "freeze_backbone",
            "use_lagrangian_vae", "kl_target", "kl_tolerance", "lagrange_lr",
            "use_dimension_wise_kl", "min_kl_per_dim",
        }
        init_kwargs = {
            key: value for key, value in hparams.items() if key in supported_init_keys
        }
        init_kwargs.update(
            {
                "pretrained_path": None,
                "val_csv": None,
                "lora": lora,
                "lora_r": int(lora_a.shape[0]) if lora_a is not None else 8,
                "use_conditional_prior": use_conditional_prior,
                "modulation_scheme": modulation_scheme,
                "use_property_prediction": use_property_prediction,
            }
        )
        init_kwargs.update(kwargs)
        module = cls(
            model_class=model_class,
            model_params=inferred,
            tokenizer=tokenizer,
            vocab_file=vocab_file,
            **init_kwargs,
        )
        incompatible = module.load_state_dict(state, strict=False)
        allowed_unexpected = ("scorer_model.",)
        unexpected = [
            key for key in incompatible.unexpected_keys
            if not key.startswith(allowed_unexpected)
        ]
        if strict and (incompatible.missing_keys or unexpected):
            raise RuntimeError(
                "Checkpoint model state is incompatible: "
                f"missing={incompatible.missing_keys}, unexpected={unexpected}"
            )
        return module

    def __init__(
        self,
        model_class: str,
        model_params: dict,
        tokenizer: str,
        vocab_file: str,
        learning_rate: float = 1e-4,
        pretrained_path: str | None = None,
        lora: str = None,  # both, encoder_only, decoder_only
        lora_r: int | None = 8,  # set None to disable adapters
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        use_conditional_prior: bool = False,
        modulation_scheme: str = "film",
        val_csv: str = None,
        label_norm: str = "minmax",
        num_bins: int = 10,
        samples_per_bin: int = 100,
        # New hyperparameters for Priority 1 improvements
        use_cyclical_annealing: bool = False,
        n_cycles: int = 4,
        ratio: float = 0.5,
        free_bits: float = 0.5,
        # A5 ablation: "cyclical" (default) | "linear" | "fixed"
        beta_schedule: str = "cyclical",
        fixed_beta: float = 1.0,  # used when beta_schedule="fixed"
        use_property_prediction: bool = False,
        lambda_prop: float = 0.1,
        # A4 ablation: freeze entire backbone (no LoRA), only conditioning modules trainable
        freeze_backbone: bool = False,
        # Tier 1: Lagrangian VAE parameters
        use_lagrangian_vae: bool = False,
        kl_target: float = 1.5,
        kl_tolerance: float = 0.3,
        lagrange_lr: float = 0.01,
        # Tier 1: Dimension-wise KLD parameters
        use_dimension_wise_kl: bool = False,
        min_kl_per_dim: float = 0.01,
        run_generation_validation: bool = False,
        validation_property: str = "sa_score",
    ):
        model_params["use_conditional_prior"] = use_conditional_prior
        model_params["modulation_scheme"] = modulation_scheme
        model_params["lora"] = lora  # Pass lora config to model
        model_params["use_property_prediction"] = use_property_prediction  # Priority 1
        # Snapshot max_len BEFORE super().__init__ increments it (+1 in lm_vae.py).
        # save_hyperparameters() below must store the original value so that
        # load_from_checkpoint → __init__ replay doesn't double-increment.
        _max_len_orig = model_params.get("max_len")
        super().__init__(
            model_class=model_class,
            model_params=model_params,
            tokenizer=tokenizer,
            vocab_file=vocab_file,
            learning_rate=learning_rate,
        )
        # Restore the original max_len in model_params so save_hyperparameters
        # (called later in this __init__) stores the pre-increment value.
        if _max_len_orig is not None:
            model_params["max_len"] = _max_len_orig
        self.use_conditional_prior = use_conditional_prior
        self.lora = lora  # Store lora configuration for inference
        
        # Store Priority 1 improvement hyperparameters
        self.use_cyclical_annealing = use_cyclical_annealing
        self.n_cycles = n_cycles
        self.ratio = ratio
        self.free_bits = free_bits
        # A5 ablation: KL annealing schedule
        self.beta_schedule = beta_schedule
        self.fixed_beta = fixed_beta
        self.use_property_prediction = use_property_prediction
        self.lambda_prop = lambda_prop
        # A4 ablation: freeze backbone flag
        self.freeze_backbone = freeze_backbone
        
        # Lagrangian VAE parameters
        self.use_lagrangian_vae = use_lagrangian_vae
        self.kl_target = kl_target
        self.kl_tolerance = kl_tolerance
        self.lagrange_lr = lagrange_lr
        
        # Dimension-wise KLD parameters
        self.use_dimension_wise_kl = use_dimension_wise_kl
        self.min_kl_per_dim = min_kl_per_dim
        self.run_generation_validation = run_generation_validation
        self.validation_property = validation_property

        logging.info(f"Conditional prior {use_conditional_prior}")
        logging.info(f"LoRA configuration: {lora}")
        logging.info(f"Cyclical annealing: {use_cyclical_annealing} (cycles={n_cycles}, ratio={ratio})")
        logging.info(f"Beta schedule (A5): {beta_schedule} (fixed_beta={fixed_beta})")
        logging.info(f"Freeze backbone (A4): {freeze_backbone}")
        logging.info(f"Free bits: {free_bits}")
        logging.info(f"Property prediction: {use_property_prediction} (lambda={lambda_prop})")
        logging.info(f"Lagrangian VAE: {use_lagrangian_vae} (target={kl_target}, tolerance={kl_tolerance})")
        logging.info(f"Dimension-wise KL: {use_dimension_wise_kl} (min_per_dim={min_kl_per_dim})")

        self.mse = nn.MSELoss()

        # Initialize Lagrange multiplier (log scale for stability)
        if self.use_lagrangian_vae:
            self.log_lagrange_mult = nn.Parameter(torch.tensor(0.0))

        if pretrained_path:
            self._load_pretrained(pretrained_path)

        if lora is not None:
            if lora in ["both", "encoder_only"]:
                add_lora_adapters(
                    self.model.encoder, r=lora_r, alpha=lora_alpha, dropout=lora_dropout
                )
            if lora in ["both", "decoder_only"]:
                add_lora_adapters(
                    self.model.decoder, r=lora_r, alpha=lora_alpha, dropout=lora_dropout
                )
            freeze_backbone_except_lora(self.model)
        elif freeze_backbone:
            # A4-frozen: freeze encoder + decoder transformer blocks;
            # keep conditioning modules (FiLM, conditional_prior, label_to_dmodel,
            # encoder_fc_mu/logvar, latent_to_hidden) trainable.
            _FROZEN_PREFIXES = ("encoder.", "decoder.")
            for name, param in self.model.named_parameters():
                if any(name.startswith(p) for p in _FROZEN_PREFIXES):
                    param.requires_grad_(False)
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            logging.info(f"A4-frozen: {trainable:,} / {total:,} parameters trainable")

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                logging.debug("trainable param: %s", name)

        if val_csv is not None:
            (
                self.val_df,
                self.val_fps,
                self.label_points_equal_count,
                self.label_points_uniform,
            ) = prepare_validation_info(
                val_csv_path=val_csv,
                label_key=label_norm,
                num_bins=num_bins,
            )
        else:
            self.val_df = None
            self.val_fps = None
            self.label_points_equal_count = None
            self.label_points_uniform = None

        self.save_hyperparameters(ignore=["model", "tokenizer", "mse", "loss_fn"])

    def get_cyclical_beta(self, step: int, total_steps: float) -> float:
        """
        Compute cyclical β-annealing schedule.
        
        Args:
            step: Current training step
            total_steps: Total number of training steps (can be float from estimate)
            
        Returns:
            Beta value for KL weighting
        """
        cycle_length = total_steps / self.n_cycles
        cycle_position = (step % cycle_length) / cycle_length
        
        if cycle_position < self.ratio:
            # Linear increase phase
            beta = cycle_position / self.ratio
        else:
            # Constant phase
            beta = 1.0
            
        return beta
    
    def apply_free_bits(self, kl_per_dim: torch.Tensor) -> torch.Tensor:
        """
        Apply free bits constraint to KL divergence.
        
        Args:
            kl_per_dim: KL divergence per latent dimension [batch_size, latent_dim]
            
        Returns:
            KL divergence with free bits applied
        """
        # Clamp each dimension's KL to be at least free_bits
        kl_clamped = torch.clamp(kl_per_dim, min=self.free_bits)
        return kl_clamped.mean()
    def update_lagrange_multiplier(self, current_kl: float):
        """Update Lagrange multiplier based on KL divergence"""
        if not self.use_lagrangian_vae:
            return
        
        # Constraint: |KL - target| < tolerance
        kl_error = current_kl - self.kl_target
        
        # Update lambda if outside tolerance
        if abs(kl_error) > self.kl_tolerance:
            with torch.no_grad():
                # Gradient ascent on lambda
                grad = kl_error  # Positive if KL too high, negative if too low
                self.log_lagrange_mult.data += self.lagrange_lr * grad
                # Clamp to reasonable range
                self.log_lagrange_mult.data.clamp_(-5.0, 5.0)

    def get_effective_beta(self, base_beta: float, current_kl: float) -> float:
        """Get effective beta combining cyclical annealing and Lagrangian"""
        if not self.use_lagrangian_vae:
            return base_beta
        
        # Lagrangian beta = exp(log_lambda)
        lagrange_beta = torch.exp(self.log_lagrange_mult).item()
        
        # Combine: use max to ensure minimum regularization
        effective_beta = max(base_beta, lagrange_beta)
        
        return effective_beta

    def compute_dimension_wise_kl(self, mu: torch.Tensor, logvar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute KL divergence per dimension and apply penalties"""
        # KL per dimension: -0.5 * (1 + logvar - mu^2 - exp(logvar))
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        
        # Mean across batch
        kl_per_dim_mean = kl_per_dim.mean(dim=0)  # [latent_dim]
        
        if self.use_dimension_wise_kl:
            # Apply penalty to collapsed dimensions
            collapsed_mask = kl_per_dim_mean < self.min_kl_per_dim
            penalty = torch.where(
                collapsed_mask,
                (self.min_kl_per_dim - kl_per_dim_mean) ** 2,
                torch.zeros_like(kl_per_dim_mean)
            )
            total_penalty = penalty.sum()
        else:
            total_penalty = torch.tensor(0.0, device=mu.device)
        
        return kl_per_dim_mean, total_penalty
    
    def _shared_step(self, batch, tag: str):
        seq = batch["input_ids"]
        y_lbl = batch["labels"].float()
        src, tgt = seq[:, :-1], seq[:, 1:]

        out, mu, logvar = self.model(src, src, y_lbl)

        recon = self.loss_fn(out.view(-1, out.size(-1)), tgt.reshape(-1))

        # Compute dimension-wise KL
        kl_per_dim_mean, dim_penalty = self.compute_dimension_wise_kl(mu, logvar)

        # Compute KL divergence
        if self.use_conditional_prior:  # KL[q(z|x,y) || p(z|y)]
            prior_mu, prior_logvar = self.model.get_conditional_prior(y_lbl)
            # KL per dimension for free bits
            kl_per_dim = -0.5 * (
                1
                + logvar
                - prior_logvar
                - ((mu - prior_mu).pow(2) + logvar.exp()) / prior_logvar.exp()
            )
            
            # Apply free bits if enabled
            if self.use_cyclical_annealing and self.free_bits > 0:
                kld_cond = self.apply_free_bits(kl_per_dim)
            else:
                kld_cond = kl_per_dim.mean()
        else:  # KL[q(z|x,y) || N(0, I)]
            # KL per dimension for free bits
            kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            
            # Apply free bits if enabled
            if self.use_cyclical_annealing and self.free_bits > 0:
                kld_cond = self.apply_free_bits(kl_per_dim)
            else:
                kld_cond = kl_per_dim.mean()

        # Standard KL for logging (without free bits)
        kld_std = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # Get base beta. Respects beta_schedule flag (A5 ablation)
        # beta_schedule: "cyclical" (default) | "linear" | "fixed"
        # use_cyclical_annealing is kept for backwards compat; beta_schedule overrides it.
        effective_schedule = self.beta_schedule if hasattr(self, "beta_schedule") else (
            "cyclical" if self.use_cyclical_annealing else "linear"
        )
        if effective_schedule == "cyclical":
            total_steps = self.trainer.estimated_stepping_batches
            base_beta = self.get_cyclical_beta(self.global_step, total_steps)
        elif effective_schedule == "linear":
            # Monotone linear ramp: 0 → 1 over first half of training, then 1
            total_steps = self.trainer.estimated_stepping_batches
            base_beta = min(1.0, self.global_step / max(1, total_steps * 0.5))
        elif effective_schedule == "fixed":
            # Fixed β without annealing (A5-fixed ablation)
            base_beta = self.fixed_beta if hasattr(self, "fixed_beta") else 1.0
        else:
            base_beta = min(1.0, self.current_epoch / 50)
        
        # Get effective beta with Lagrangian adjustment
        effective_beta = self.get_effective_beta(base_beta, kld_cond.item())
        
        # Compute total loss
        loss = recon + effective_beta * kld_cond + dim_penalty
        
        # Property prediction loss (if enabled)
        prop_loss = torch.tensor(0.0, device=loss.device)
        if self.use_property_prediction and self.model.property_predictor is not None:
            # Sample z from posterior
            z = self.model.reparameterize(mu, logvar)
            # Predict properties from latent
            y_pred = self.model.property_predictor(z)
            # MSE loss between predicted and actual labels
            prop_loss = self.mse(y_pred, y_lbl)
            loss = loss + self.lambda_prop * prop_loss

        # Update Lagrange multiplier (only during training)
        if tag == "train" and self.use_lagrangian_vae:
            self.update_lagrange_multiplier(kld_cond.item())

        # Logging
        log_dict = {
            f"{tag}_loss": loss,
            f"{tag}_recon": recon,
            f"{tag}_kld_std": kld_std,
            f"{tag}_kld_cond": kld_cond,
            f"{tag}_base_beta": base_beta,
            f"{tag}_effective_beta": effective_beta,
            f"{tag}_kl_per_dim_mean": kl_per_dim_mean.mean(),
            f"{tag}_kl_per_dim_min": kl_per_dim_mean.min(),
            f"{tag}_kl_per_dim_max": kl_per_dim_mean.max(),
            f"{tag}_dim_penalty": dim_penalty,
            f"{tag}_active_dims": (kl_per_dim_mean > self.min_kl_per_dim).sum(),
        }
        
        if self.use_property_prediction:
            log_dict[f"{tag}_prop_loss"] = prop_loss

        if self.use_lagrangian_vae:
            log_dict[f"{tag}_lagrange_mult"] = torch.exp(self.log_lagrange_mult).item()

        # Log per-batch values without aggregation to see variation across batches
        # sync_dist=True for distributed training, on_step=True to log each batch
        self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def _load_pretrained(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        clean_state = {
            k[6:] if k.startswith("model.") else k: v for k, v in state.items()
        }

        # Filter out positional encoding tensors that may differ in size
        # (e.g. pretrained backbone trained with max_len=71 → PE [1,72,d],
        #  fine-tuned model built with max_len=72 → PE [1,73,d]).
        # strict=False skips missing/unexpected keys but still raises on size
        # mismatches, so we drop any PE tensors with incompatible shape.
        filtered_state = {}
        for k, v in clean_state.items():
            if "pos_encoder.pe" in k or "pos_decoder.pe" in k:
                # Check shape against current model parameter
                try:
                    current_param = dict(self.model.named_parameters()).get(k)
                    if current_param is None:
                        # Also check buffers
                        current_param = dict(self.model.named_buffers()).get(k)
                    if current_param is not None and current_param.shape != v.shape:
                        logging.info(
                            f"[pretrained] skipping {k}: shape mismatch "
                            f"{v.shape} vs model {current_param.shape}"
                        )
                        continue
                except Exception:
                    pass
            filtered_state[k] = v

        missing, unexpected = self.model.load_state_dict(filtered_state, strict=False)
        logging.info(
            f"[pretrained] loaded {len(filtered_state) - len(unexpected)} "
            f"tensors  |  missing: {len(missing)}  unexpected: {len(unexpected)}"
        )

    def on_after_backward(self):
        if (self.global_step + 1) % 100 != 0:
            return
        if self.model.modulation is None:
            return
        for name, p in self.model.modulation.named_parameters():
            if p.grad is not None:
                self.log(
                    f"grad_{name}",
                    p.grad.detach().norm(),
                    prog_bar=False,
                    on_step=True,
                    on_epoch=False,
                )

    def on_validation_epoch_end(self):
        if not self.run_generation_validation:
            return
        logging.info("on_validation_epoch_end starting analysis...")
        if self.hparams.val_csv is None:
            results, labeled_sets = run_conditional_generation_evaluation_compute_prop(
                model=self.model,
                tokenizer=self.tokenizer,
                label_name=self.validation_property,
                device=self.device,
                num_labels=10,
                samples_per_label=20,
            )
            for k, v in results.items():
                if v is not None:
                    self.log(k, v, prog_bar=True)
            for idx, labeled_molecules in enumerate(labeled_sets):
                img = draw_labeled_molecule_grid(labeled_molecules)
                if img is not None:
                    img_tensor = self.PIL_to_tensor(img)
                    experiment = getattr(self.logger, "experiment", None)
                    if experiment is None or not hasattr(experiment, "add_image"):
                        continue
                    experiment.add_image(
                        f"Generated Molecules Label Group {idx}",
                        img_tensor,
                        self.global_step,
                    )
        else:
            label_col = f"y_normalized_{self.hparams.label_norm}"
            logging.info("on_validation_epoch_end done analysis...for quantile")
            scores_eq = run_validation_distribution_match(
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                val_df=self.val_df,
                val_fps=self.val_fps,
                label_points=self.label_points_equal_count,
                label_col=label_col,
                samples_per_bin=self.hparams.samples_per_bin,
                binning_mode="equal_count",
            )

            logging.info("on_validation_epoch_end done analysis...for distribution")
            scores_un = run_validation_distribution_match(
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                val_df=self.val_df,
                val_fps=self.val_fps,
                label_points=self.label_points_uniform,
                label_col=label_col,
                samples_per_bin=self.hparams.samples_per_bin,
                binning_mode="uniform",
            )

            for tag, scores in [("equal_count", scores_eq), ("uniform", scores_un)]:
                agg = scores["aggregate"]
                self.log(f"val_{tag}_mmd", agg["mmd_mean"], prog_bar=True)
                self.log(f"val_{tag}_ws", agg["ws_mean"], prog_bar=True)
                self.log(f"val_{tag}_jsd", agg["jsd_mean"], prog_bar=True)
                self.log(
                    f"val_{tag}_scaf_jaccard",
                    agg["scaffold_mean_jaccard"],
                    prog_bar=False,
                )
                self.log(
                    f"val_{tag}_scaf_recovery",
                    agg["scaffold_mean_recovery"],
                    prog_bar=False,
                )
                self.log(
                    f"val_{tag}_scaf_jsd", agg["scaffold_dist_jsd_mean"], prog_bar=False
                )
                self.log(
                    f"val_{tag}_scaf_ws", agg["scaffold_dist_ws_mean"], prog_bar=False
                )

            step = self.global_step
            out_dir = self.logger.log_dir
            scaf_df = scores["scaffold_stats"]
            scaf_df.to_csv(
                f"{out_dir}/scaffold_stats_step_{step}_{tag}.csv", index=False
            )
            self.logger.experiment.flush()
    
    def generate_molecules_with_target_label(
        self, num_samples, target_label, temperature=1.0, top_k=None,
        prior_mode="auto",
    ):
        if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError("num_samples must be a positive integer")
        _validate_sampling_args(temperature, top_k, self.tokenizer.vocab_size)
        if prior_mode not in {"auto", "standard", "conditional"}:
            raise ValueError("prior_mode must be 'auto', 'standard', or 'conditional'")
        if prior_mode == "conditional" and not self.model.use_conditional_prior:
            raise ValueError("conditional prior sampling requested but the model has no conditional prior")
        self.model.eval()
        chunk = 1000
        all_smiles = []
        for start in range(0, num_samples, chunk):
            batch_size = min(chunk, num_samples - start)
            labels = torch.full(
                (batch_size, self.model.label_dim),
                target_label,
                dtype=torch.float,
                device=self.device,
            )
            use_conditional = self.model.use_conditional_prior and prior_mode != "standard"
            if use_conditional:
                prior_mu, prior_logvar = self.model.get_conditional_prior(labels)
                z = self.model.reparameterize(prior_mu, prior_logvar)
            else:
                z = torch.randn(batch_size, self.model.latent_dim, device=self.device)
            generated_smiles = generate_molecules_with_labels(self.model, self.tokenizer, z, labels, temperature, top_k, self.lora)
            all_smiles.extend(generated_smiles)
        return all_smiles

def generate_molecules_with_labels(
    model, tokenizer, z, labels,
    temperature=1.0, top_k=None, lora=None,
    use_label_token=None,
):
    """
    Autoregressive generation conditioned on labels.

    Parameters
    ----------
    use_label_token : bool or None
        Models exposing ``_build_memory`` use it automatically. ``False`` can
        explicitly select baseline memory for controlled comparisons.
    lora : str or None
        "both" / "decoder_only" → apply FiLM modulation to z before decoding.
        None / other → skip FiLM (unconditional memory from z directly).
    """
    model.eval()
    device = z.device
    batch_size = z.size(0)
    max_len = model.max_len
    _validate_sampling_args(temperature, top_k, tokenizer.vocab_size)
    if z.ndim != 2 or z.size(1) != model.latent_dim:
        raise ValueError(f"z must have shape (batch, {model.latent_dim})")
    if labels.ndim != 2 or labels.size(0) != batch_size or labels.size(1) != model.label_dim:
        raise ValueError(f"labels must have shape ({batch_size}, {model.label_dim})")

    generated_tokens = torch.full(
        (batch_size, 1), tokenizer.bos_token_id, dtype=torch.long, device=device
    )
    generated_smiles = []

    # Pre-compute memory once. It is the same for every decoding step because
    # z and labels are fixed per call.
    with torch.no_grad():
        if hasattr(model, "_build_memory") and use_label_token is not False:
            # ── Deep FiLM path: memory shape (max_len+1, B, d_model) ──────────
            # Includes FiLM-modulated latent positions (1..max_len) PLUS a
            # dedicated label token at position max_len+1.
            memory = model._build_memory(z, labels)
        else:
            # ── Baseline path: memory shape (max_len, B, d_model) ───────────
            if lora in ["both", "decoder_only"] and model.modulation is not None:
                zc = model.modulation(z, labels)
                z_hidden = model.latent_to_hidden(zc)
            else:
                z_hidden = model.latent_to_hidden(z)
            memory = z_hidden.unsqueeze(1).repeat(1, max_len, 1).transpose(0, 1)

        for _ in range(max_len - 1):
            tgt = generated_tokens
            tgt_emb = model.decoder.embedding(tgt) * math.sqrt(model.decoder.d_model)
            tgt_emb = model.decoder.pos_decoder(tgt_emb)
            tgt_emb = tgt_emb.transpose(0, 1)
            tgt_mask = model.decoder.generate_square_subsequent_mask(
                tgt_emb.size(0)
            ).to(device)
            tgt_key_padding_mask = tgt == tokenizer.pad_token_id

            output = model.decoder.transformer_decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )
            output = output.transpose(0, 1)
            logits = model.decoder.output_layer(output[:, -1, :])
            predicted = _sample_from_logits(logits, temperature, top_k)
            generated_tokens = torch.cat((generated_tokens, predicted), dim=1)

            if (predicted == tokenizer.eos_token_id).all():
                break

        generated_tokens = generated_tokens[:, 1:]  # remove BOS

        for token_seq in generated_tokens:
            tokens = []
            for idx in token_seq:
                idx = idx.item()
                if idx == tokenizer.eos_token_id:
                    break
                if idx != tokenizer.pad_token_id:
                    tokens.append(tokenizer.convert_ids_to_tokens(idx))
            mol_str = "".join(tokens)
            try:
                # Use the tokenizer's own conversion method so that:
                #   SELFIESTokenizer → sf.decoder(mol_str)  (SELFIES → SMILES)
                #   SMILESTokenizer  → mol_str              (already SMILES)
                #   SAFETokenizer    → safe.decode(mol_str)
                if hasattr(tokenizer, "convert_to_smiles"):
                    smiles = tokenizer.convert_to_smiles(mol_str)
                else:
                    smiles = sf.decoder(mol_str)  # fallback: assume SELFIES
                mol = Chem.MolFromSmiles(smiles) if smiles is not None else None
                generated_smiles.append(smiles if mol is not None else None)
            except Exception:
                generated_smiles.append(None)

    return generated_smiles
