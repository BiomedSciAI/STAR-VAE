# encoder.py
import torch
import torch.nn as nn
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(
            1
        )  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        dropout,
        padding_idx,
        max_len,
    ):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

    def forward(self, src):
        src = src.long()  # for Apple Silicon error
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        src_emb = src_emb.transpose(0, 1)  # (seq_len, batch_size, d_model)
        src_key_padding_mask = src == self.padding_idx  # (batch_size, seq_len)
        memory = self.transformer_encoder(
            src_emb, src_key_padding_mask=src_key_padding_mask
        )
        memory = memory.transpose(0, 1)  # (batch_size, seq_len, d_model)
        return memory
