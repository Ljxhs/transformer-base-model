import torch
import torch.nn as nn
from modules import MultiHeadAttention, PositionwiseFeedforward, LayerNorm


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout):
        super().__init__()
        
        self.self_attn_layer_norm = LayerNorm(hid_dim)
        self.enc_attn_layer_norm = LayerNorm(hid_dim)
        self.ff_layer_norm = LayerNorm(hid_dim)
        
        self.self_attention = MultiHeadAttention(hid_dim, n_heads, dropout)
        self.encoder_attention = MultiHeadAttention(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforward(hid_dim, pf_dim, dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        
        _trg, _ = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
        
        _trg = self.positionwise_feedforward(trg)
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        
        return trg


class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, max_len): 
        super().__init__()
        
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_len, hid_dim)
        
        self.layers = nn.ModuleList([
            DecoderLayer(hid_dim, n_heads, pf_dim, dropout)
            for _ in range(n_layers)
        ])
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer('scale', torch.sqrt(torch.tensor([hid_dim], dtype=torch.float32)))

    def forward(self, trg, enc_src, trg_mask, src_mask):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        device = trg.device
        
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(device)
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        
        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)
        
        output = self.fc_out(trg)

        return output, None
