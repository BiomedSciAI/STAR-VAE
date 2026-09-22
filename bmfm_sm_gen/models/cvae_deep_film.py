"""
cvae_deep_film.py
==================
Exp17-A: ConditionalDeepFiLMVAE
================================
Extension of ConditionalTransformerVAE that injects the conditioning label y* at
EVERY decoder cross-attention layer via a dedicated "label memory token", rather than
only modulating the latent vector z once before decoding.

Design decision (see README.md):
  - A single extra memory position `label_token = label_token_proj(y*)` is appended
    to the latent-derived memory before every decoder forward pass.
  - Memory shape changes from (max_len, B, d_model) → (max_len+1, B, d_model).
  - Every cross-attention layer in the 12-layer decoder now has one dedicated slot
    carrying pure label information, which each attention head can learn to attend to
    independently and in combination with the latent content.
  - The rest of the architecture is unchanged: same encoder, same FiLM latent
    modulation (kept for backward-compatible weight loading), same conditional prior.

New trainable parameters (all others frozen during fine-tuning):
  - label_token_proj: Linear(label_dim, d_model): 1×512 = 512 weights
  - modulation (FiLM MLP): re-trained from pretrained init
  - conditional_prior_mu / logvar MLPs: re-trained from pretrained init
  - latent_to_hidden: re-trained from pretrained init

Backward compatibility:
  - Can load weights from any existing ConditionalTransformerVAE checkpoint
    with strict=False; label_token_proj initialises to near-zero (near-identity start).
  - The FiLM modulation pathway is retained so the model can also be loaded as a
    standard ConditionalTransformerVAE for baseline comparison.
"""

import math

import torch
import torch.nn as nn

from bmfm_sm_gen.models.encoder import Encoder
from bmfm_sm_gen.models.decoder import Decoder


class LatentFiLM(nn.Module):
    """Affine transformation of z conditioned on y: zc = γ(y) ⊙ z + β(y)."""
    def __init__(self, latent_dim: int, label_dim: int, hidden_mult: int = 4):
        super().__init__()
        h = hidden_mult * latent_dim
        self.mlp = nn.Sequential(
            nn.Linear(label_dim, h),
            nn.SiLU(),
            nn.Linear(h, 2 * latent_dim),
        )

    def forward(self, z, y):
        gamma, beta = self.mlp(y).chunk(2, dim=-1)
        return gamma * z + beta


class ConditionalDeepFiLMVAE(nn.Module):
    """
    CVAE with label injection at every decoder layer via a prefix memory token.

    At decode time the memory tensor is extended by one position:
        memory_aug = [z_hidden × max_len, label_token]   shape: (max_len+1, B, d_model)
    The transformer decoder's cross-attention then attends to both the latent-derived
    positions and the dedicated label position at every layer.

    Encoding and the KL / conditional-prior pathway are identical to the baseline CVAE.
    """

    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        latent_dim: int,
        dropout: float,
        padding_idx: int,
        num_tasks: int,
        use_conditional_prior: bool = True,
        modulation_scheme: str = "film",   # kept for compat; always uses FiLM here
        lora: str = None,                  # kept for compat
        use_property_prediction: bool = False,  # kept for compat
    ):
        super().__init__()

        self.d_model    = d_model
        self.latent_dim = latent_dim
        self.max_len    = max_len
        self.padding_idx = padding_idx
        self.label_dim  = num_tasks
        self.use_conditional_prior = use_conditional_prior
        self.lora       = lora
        self.use_property_prediction = use_property_prediction

        # ── Encoder ──────────────────────────────────────────────────────────
        self.encoder = Encoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            padding_idx=padding_idx,
            max_len=max_len,
        )

        # ── Decoder ──────────────────────────────────────────────────────────
        self.decoder = Decoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            padding_idx=padding_idx,
            max_len=max_len,
        )

        # ── Latent projection ────────────────────────────────────────────────
        self.encoder_fc_mu     = nn.Linear(d_model, latent_dim)
        self.encoder_fc_logvar = nn.Linear(d_model, latent_dim)
        self.latent_to_hidden  = nn.Linear(latent_dim, d_model)

        # ── Label injection (encoder side, for compat with encode_with_labels) ─
        self.label_to_dmodel = nn.Linear(num_tasks, d_model)

        # ── FiLM modulation (latent side, retained for compat + additional signal) ─
        self.modulation = LatentFiLM(latent_dim, num_tasks)

        # ── NEW: Label token projection for deep decoder injection ────────────
        # Projects scalar label y* → d_model-dimensional token appended to memory.
        # Initialised near-zero so model starts at baseline behaviour.
        self.label_token_proj = nn.Linear(num_tasks, d_model, bias=True)
        nn.init.normal_(self.label_token_proj.weight, std=1e-3)
        nn.init.zeros_(self.label_token_proj.bias)

        # ── Conditional prior ─────────────────────────────────────────────────
        if use_conditional_prior:
            hidden_dim = latent_dim // 2
            self.conditional_prior_mu = nn.Sequential(
                nn.Linear(num_tasks, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.conditional_prior_logvar = nn.Sequential(
                nn.Linear(num_tasks, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            for module in [self.conditional_prior_mu, self.conditional_prior_logvar]:
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight, gain=0.01)
                        nn.init.zeros_(layer.bias)
        else:
            self.conditional_prior_mu    = None
            self.conditional_prior_logvar = None

        # property predictor stub (kept for compat)
        self.property_predictor = None

    # ── Encoding ─────────────────────────────────────────────────────────────

    def encode(self, src):
        memory = self.encoder(src)
        pooled = self._masked_pool(memory, src)
        mu     = self.encoder_fc_mu(pooled)
        logvar = self.encoder_fc_logvar(pooled)
        return mu, logvar

    def encode_with_labels(self, src, labels):
        memory    = self.encoder(src)
        pooled    = self._masked_pool(memory, src)
        label_proj = self.label_to_dmodel(labels)
        pooled    = pooled + label_proj
        mu        = self.encoder_fc_mu(pooled)
        logvar    = self.encoder_fc_logvar(pooled)
        return mu, logvar

    def _masked_pool(self, memory, src):
        mask = src.ne(self.padding_idx).unsqueeze(-1)
        return (memory * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

    def get_conditional_prior(self, labels):
        assert self.use_conditional_prior
        return self.conditional_prior_mu(labels), self.conditional_prior_logvar(labels)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    # ── Decoding ─────────────────────────────────────────────────────────────

    def _build_memory(self, z, labels):
        """
        Build the augmented memory tensor used by the decoder.

        Baseline: memory = [z_h, z_h, ..., z_h]          shape (max_len,   B, d_model)
        Deep FiLM: memory = [z_h, ..., z_h, label_token]  shape (max_len+1, B, d_model)

        The FiLM-modulated z is kept so that the original conditioning signal is also
        present in all max_len latent positions.
        """
        # FiLM on latent (existing pathway, retained)
        zc       = self.modulation(z, labels)
        z_hidden = self.latent_to_hidden(zc)                   # (B, d_model)
        z_mem    = z_hidden.unsqueeze(1).repeat(1, self.max_len, 1)  # (B, max_len, d_model)
        z_mem    = z_mem.transpose(0, 1)                       # (max_len, B, d_model)

        # New label token
        label_tok = self.label_token_proj(labels)              # (B, d_model)
        label_mem = label_tok.unsqueeze(0)                     # (1, B, d_model)

        return torch.cat([z_mem, label_mem], dim=0)            # (max_len+1, B, d_model)

    def decode_with_labels(self, z, tgt, labels):
        memory = self._build_memory(z, labels)

        tgt_emb = self.decoder.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.decoder.pos_decoder(tgt_emb)
        tgt_emb = tgt_emb.transpose(0, 1)                       # (tgt_len, B, d_model)
        tgt_mask = self.decoder.generate_square_subsequent_mask(
            tgt_emb.size(0)
        ).to(tgt_emb.device)
        tgt_key_padding_mask = tgt == self.padding_idx

        output = self.decoder.transformer_decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        output = output.transpose(0, 1)                         # (B, tgt_len, d_model)
        return self.decoder.output_layer(output)

    def decode(self, z, tgt):
        """Unconditional decode, uses FiLM only, no label token. For compat."""
        zc       = self.modulation(z, torch.zeros(z.size(0), self.label_dim, device=z.device))
        z_hidden = self.latent_to_hidden(zc)
        memory   = z_hidden.unsqueeze(1).repeat(1, self.max_len, 1).transpose(0, 1)

        tgt_emb = self.decoder.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.decoder.pos_decoder(tgt_emb)
        tgt_emb = tgt_emb.transpose(0, 1)
        tgt_mask = self.decoder.generate_square_subsequent_mask(
            tgt_emb.size(0)
        ).to(tgt_emb.device)
        tgt_key_padding_mask = tgt == self.padding_idx

        output = self.decoder.transformer_decoder(
            tgt_emb, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return self.decoder.output_layer(output.transpose(0, 1))

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, src, tgt, labels):
        # Encode with label bias (same as baseline lora='both' path)
        mu, logvar = self.encode_with_labels(src, labels)
        z          = self.reparameterize(mu, logvar)
        output     = self.decode_with_labels(z, tgt, labels)
        return output, mu, logvar
