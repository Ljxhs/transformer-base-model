import torch
import torch.nn as nn
from modules import MultiHeadAttention, PositionwiseFeedforward, LayerNorm

class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout):
        super().__init__()
        
        self.self_attn_layer_norm = LayerNorm(hid_dim)
        self.ff_layer_norm = LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttention(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforward(hid_dim, pf_dim, dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, 1, 1, src len]
        
        # Multi-head self-attention + residual connection + LayerNorm
        _src, _ = self.self_attention(src, src, src, src_mask)
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        
        # Position-wise FFN + residual connection + LayerNorm
        _src = self.positionwise_feedforward(src)
        src = self.ff_layer_norm(src + self.dropout(_src))
        
        # src = [batch size, src len, hid dim]
        return src

class Encoder(nn.Module):
    # 移除 device 参数
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, max_len): 
        super().__init__()
        
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_len, hid_dim)
        
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('scale', torch.sqrt(torch.tensor([hid_dim], dtype=torch.float32)))

        
        
    def forward(self, src, src_mask):
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        device = src.device 
        
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(device)
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        
        for layer in self.layers:
            src = layer(src, src_mask)
        
        return src