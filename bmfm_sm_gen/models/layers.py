import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer


class LoRALinear(nn.Module):
    """
    Wraps a frozen nn.Linear and adds a (B @ A) ΔW term with rank=r.
    scaling = alpha / r reproduces the original LoRA paper.
    """

    def __init__(
        self, linear: nn.Linear, r: int = 8, alpha: int = 32, dropout: float = 0.0
    ):
        super().__init__()
        if r <= 0:
            raise ValueError("Rank r must be positive")

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Freeze the base weight and (optionally) bias
        self.weight = linear.weight
        self.weight.requires_grad_(False)
        self.bias = linear.bias  # may be None
        if self.bias is not None:
            self.bias.requires_grad_(False)

        # LoRA parameters (A: r×in, B: out×r)
        self.lora_A = nn.Parameter(torch.zeros(r, linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, r))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        result = F.linear(x, self.weight, self.bias)  # frozen path
        lora_update = (self.lora_B @ self.lora_A) * self.scaling
        result += F.linear(self.dropout(x), lora_update)  # ΔW x
        return result


def _patch_module(module: nn.Module, r: int, alpha: int, dropout: float):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, r, alpha, dropout))
        else:
            _patch_module(child, r, alpha, dropout)


def add_lora_adapters(
    root: nn.Module, r: int = 8, alpha: int = 32, dropout: float = 0.0
):
    for mod in root.modules():  # depth-first walk
        if isinstance(mod, (TransformerEncoderLayer, TransformerDecoderLayer)):
            _patch_module(mod, r, alpha, dropout)
    return root


def freeze_backbone_except_lora(model: nn.Module):
    for n, p in model.named_parameters():
        # if "transformer_encoder" in n or "transformer_decoder" in n:
        if "lora_" not in n and 'modulation' not in n and 'latent_to_hidden' not in n:
            p.requires_grad_(False)
        