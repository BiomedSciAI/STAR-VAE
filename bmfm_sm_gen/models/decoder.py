# decoder.py
import torch
import torch.nn as nn
import math
from torch.nn import TransformerDecoder, TransformerDecoderLayer


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


class Decoder(nn.Module):
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
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.padding_idx = padding_idx
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_decoder = PositionalEncoding(d_model, dropout, max_len)
        decoder_layers = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout
        )
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_layers)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory):
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_decoder(tgt_emb)
        tgt_emb = tgt_emb.transpose(0, 1)  # (tgt_len, batch_size, d_model)
        tgt_mask = self.generate_square_subsequent_mask(tgt_emb.size(0)).to(
            tgt_emb.device
        )
        tgt_key_padding_mask = tgt == self.padding_idx
        output = self.transformer_decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        output = output.transpose(0, 1)  # (batch_size, tgt_len, d_model)
        output = self.output_layer(output)
        return output

    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence."""
        mask = torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)
        return mask
