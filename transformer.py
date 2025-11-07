import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from utils import make_src_mask, make_trg_mask


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def forward(self, src, trg):
        # --- 确保 mask 在同一设备 ---
        src_mask = make_src_mask(src, self.src_pad_idx).to(self.device)
        trg_mask = make_trg_mask(trg, self.trg_pad_idx, self.device).to(self.device)

        # --- 编码器 ---
        enc_src = self.encoder(src, src_mask)

        # --- 解码器 ---
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        # output: [batch size, trg len, output dim]
        # attention: [batch size, n heads, trg len, src len] or None
        return output, attention
