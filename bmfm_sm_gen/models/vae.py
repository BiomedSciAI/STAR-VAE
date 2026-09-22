import torch
import torch.nn as nn
from bmfm_sm_gen.models.encoder import Encoder
from bmfm_sm_gen.models.decoder import Decoder


class TransformerVAE(nn.Module):
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
        use_vae: bool = True,  # A2 ablation: set False for deterministic AE
    ):
        super(TransformerVAE, self).__init__()
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
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.use_vae = use_vae

        # Latent space transformations
        self.encoder_fc_mu = nn.Linear(d_model, latent_dim)
        self.encoder_fc_logvar = nn.Linear(d_model, latent_dim)
        self.latent_to_hidden = nn.Linear(latent_dim, d_model)

    def encode(self, src):
        # src: (batch_size, seq_len)
        memory = self.encoder(src)  # (batch_size, seq_len, d_model)
        mask = src.ne(self.padding_idx).unsqueeze(-1)
        pooled = (memory * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        mu = self.encoder_fc_mu(pooled)
        logvar = self.encoder_fc_logvar(pooled)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Reparameterization trick (skipped when use_vae=False, deterministic AE)
        if not self.use_vae:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, tgt):
        # z: (batch_size, latent_dim)
        # Map z to (batch_size, d_model)
        z_hidden = self.latent_to_hidden(z)  # (batch_size, d_model)
        # Expand z_hidden to (batch_size, max_len, d_model)
        memory = z_hidden.unsqueeze(1).repeat(
            1, self.max_len, 1
        )  # (batch_size, max_len, d_model)
        memory = memory.transpose(0, 1)  # (max_len, batch_size, d_model)
        output = self.decoder(tgt, memory)
        return output

    def forward(self, src, tgt):
        mu, logvar = self.encode(src)
        z = self.reparameterize(mu, logvar)
        output = self.decode(z, tgt)
        return output, mu, logvar
