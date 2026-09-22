from bmfm_sm_gen.models.decoder import Decoder
from bmfm_sm_gen.models.encoder import Encoder
import torch
import torch.nn as nn


class LatentFiLM(nn.Module):
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


class PropertyGuidedShift(nn.Module):
    """
    Shift a latent vector using its associated property.

    To make sure we start with almost zero shift, the last layer in the MLP is scaled down
    """

    def __init__(self, latent_dim, num_tasks, init_scale=1e-3):
        super().__init__()
        hidden_dim = latent_dim * 2
        self.shift = nn.Sequential(
            nn.Linear(latent_dim + num_tasks, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Scale down the final layer weights to start with near-zero shift
        nn.init.zeros_(self.shift[-1].bias)
        nn.init.normal_(self.shift[-1].weight, mean=0.0, std=init_scale)

    def forward(self, z, prop):
        # z: [B, latent_dim]
        # prop: [B, num_tasks], e.g., torch.tensor([[logP], [logP], ...])
        x = torch.cat([z, prop], dim=-1)
        delta_z = self.shift(x)
        return z + delta_z  # Apply the shift


class ConditionalTransformerVAE(nn.Module):
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
        num_tasks: int,  # continuous label dim
        use_conditional_prior: bool = False,
        modulation_scheme: str = "film",
        lora: str = None,  # both, encoder_only, decoder_only
        use_property_prediction: bool = False,  # Priority 1: property prediction
        use_vae: bool = True,  # A2 ablation: set False for deterministic AE
    ):
        super().__init__()
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
        self.label_dim = num_tasks
        self.use_conditional_prior = use_conditional_prior
        self.lora = lora  # Store lora configuration
        self.use_property_prediction = use_property_prediction
        self.use_vae = use_vae

        self.encoder_fc_mu = nn.Linear(d_model, latent_dim)
        self.encoder_fc_logvar = nn.Linear(d_model, latent_dim)
        self.latent_to_hidden = nn.Linear(latent_dim, d_model)

        # Label injection into encoder output
        self.label_to_dmodel = nn.Linear(num_tasks, d_model)

        # Decoder conditioning
        self.modulation_scheme = modulation_scheme
        if modulation_scheme == "film":
            self.modulation = LatentFiLM(latent_dim, num_tasks)
        elif modulation_scheme == "pgs":
            self.modulation = PropertyGuidedShift(latent_dim, num_tasks)
        else:
            # "none" or unrecognised, no modulation; identity pass-through
            self.modulation = None

        # Priority 1: Enhanced conditional prior with 2-layer MLPs
        # Only initialize if enabled to avoid DDP unused parameter errors
        if use_conditional_prior:
            hidden_dim = latent_dim // 2
            
            # 2-layer MLP for mu
            self.conditional_prior_mu = nn.Sequential(
                nn.Linear(num_tasks, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim)
            )
            
            # 2-layer MLP for logvar
            self.conditional_prior_logvar = nn.Sequential(
                nn.Linear(num_tasks, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim)
            )
            
            # Xavier initialization with small gain for both MLPs
            for module in [self.conditional_prior_mu, self.conditional_prior_logvar]:
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight, gain=0.01)
                        nn.init.zeros_(layer.bias)
        else:
            # Set to None when disabled to make it explicit
            self.conditional_prior_mu = None
            self.conditional_prior_logvar = None
        
        # Priority 1: Property predictor for auxiliary loss
        # Only initialize if enabled to avoid DDP unused parameter errors
        if use_property_prediction:
            hidden_dim = latent_dim // 2
            self.property_predictor = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_tasks)
            )
        else:
            # Set to None when disabled to make it explicit
            self.property_predictor = None

    def encode(self, src):
        # src: (batch_size, seq_len)
        memory = self.encoder(src)  # (batch_size, seq_len, d_model)
        pooled = self._masked_pool(memory, src)
        mu = self.encoder_fc_mu(pooled)
        logvar = self.encoder_fc_logvar(pooled)
        return mu, logvar

    def encode_with_labels(self, src, labels):
        memory = self.encoder(src)  # (B, L, d_model)
        pooled = self._masked_pool(memory, src)

        label_proj = self.label_to_dmodel(labels)  # (B, d_model)
        pooled = pooled + label_proj

        mu = self.encoder_fc_mu(pooled)
        logvar = self.encoder_fc_logvar(pooled)
        return mu, logvar

    def _masked_pool(self, memory, src):
        mask = src.ne(self.padding_idx).unsqueeze(-1)
        return (memory * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

    def get_conditional_prior(self, labels):
        """
        Get conditional prior distribution p(z|y).
        
        Args:
            labels: Property labels [batch_size, num_tasks]
            
        Returns:
            mu, logvar: Mean and log-variance of conditional prior

        """
        assert self.use_conditional_prior, "Conditional prior not enabled"
        mu = self.conditional_prior_mu(labels)
        logvar = self.conditional_prior_logvar(labels)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Reparameterization trick (skipped when use_vae=False, deterministic AE)
        if not self.use_vae:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_with_labels(self, z, tgt, labels):
        zc = self.modulation(z, labels) if self.modulation is not None else z
        z_hidden = self.latent_to_hidden(zc)  # (B, d_model)
        memory = z_hidden.unsqueeze(1).repeat(1, self.max_len, 1)
        memory = memory.transpose(0, 1)  # (max_len, B, d_model)
        output = self.decoder(tgt, memory)
        return output
    
    def decode(self, z, tgt):
        z_hidden = self.latent_to_hidden(z)
        memory = z_hidden.unsqueeze(1).repeat(1, self.max_len, 1)
        memory = memory.transpose(0, 1)
        output = self.decoder(tgt, memory)
        return output

    def forward(self, src, tgt, labels):
        # Conditioning is a model semantic; LoRA only controls adapter placement.
        if self.modulation is not None:
            mu, logvar = self.encode_with_labels(src, labels)
        else:
            mu, logvar = self.encode(src)
        
        z = self.reparameterize(mu, logvar)
        
        if self.modulation is not None:
            output = self.decode_with_labels(z, tgt, labels)
        else:
            output = self.decode(z, tgt)
        
        return output, mu, logvar
