import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformer import Transformer
from encoder import Encoder
from decoder import Decoder
from dataset_loader import get_dataloaders
from config import *
from modules import LayerNorm

def train_epoch(model, dataloader, optimizer, criterion, clip, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        src = batch['english'].to(device)
        trg = batch['german'].to(device)
        optimizer.zero_grad()
        output, _ = model(src, trg[:, :-1])
        output_dim = output.shape[-1]
        loss = criterion(output.contiguous().view(-1, output_dim),
                         trg[:, 1:].contiguous().view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            src = batch['english'].to(device)
            trg = batch['german'].to(device)
            output, _ = model(src, trg[:, :-1])
            output_dim = output.shape[-1]
            loss = criterion(output.contiguous().view(-1, output_dim),
                             trg[:, 1:].contiguous().view(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)

def build_model(use_pos_enc=True, use_layer_norm=True, n_heads=ENC_HEADS, dropout=ENC_DROPOUT):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enc = Encoder(INPUT_DIM, HID_DIM, ENC_LAYERS, n_heads, ENC_PF_DIM, dropout, MAX_LEN)
    dec = Decoder(OUTPUT_DIM, HID_DIM, DEC_LAYERS, n_heads, DEC_PF_DIM, dropout, MAX_LEN)
    model = Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device)

    # 🔧 禁用 LayerNorm（避免 OrderedDict mutated 错误）
    if not use_layer_norm:
        from modules import LayerNorm
        # 第一步：收集要替换的层路径
        to_replace = []
        for name_m, module in model.named_modules():
            if isinstance(module, LayerNorm):
                to_replace.append(name_m)

        # 第二步：逐个替换
        for name_m in to_replace:
            parent = model
            *path, last = name_m.split('.')
            for p in path:
                parent = getattr(parent, p)
            setattr(parent, last, nn.Identity())

    model.to(device)
    return model, device
