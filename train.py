import torch
torch.cuda.empty_cache()

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from transformer import Transformer
from encoder import Encoder
from decoder import Decoder
from utils import make_src_mask, make_trg_mask
from dataset_loader import get_dataloaders  
from config import *


import matplotlib.pyplot as plt

# 训练结束后可视化函数
def plot_loss_curve(train_losses, val_losses, save_path="loss_curve.png"):
    plt.figure(figsize=(8,6))
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses)+1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)   # 保存图片
    plt.show()               # 显示图片



# ---------- 单轮训练 ----------
def train_epoch(model, dataloader, optimizer, criterion, clip, device):
    model.train()
    epoch_loss = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        src = batch['english'].to(device)
        trg = batch['german'].to(device)

        optimizer.zero_grad()

        # 模型前向传播
        output, _ = model(src, trg[:, :-1])
        # output: [batch_size, trg_len-1, output_dim]
        # trg: [batch_size, trg_len]

        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output, trg)
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


# ---------- 验证 ----------
def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    total_tokens = 0
    correct_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            src = batch['english'].to(device)
            trg = batch['german'].to(device)

            output, _ = model(src, trg[:, :-1])
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg_flat = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output, trg_flat)
            epoch_loss += loss.item()

            # Accuracy
            preds = output.argmax(dim=1)
            non_pad = trg_flat != TRG_PAD_IDX
            correct_tokens += (preds[non_pad] == trg_flat[non_pad]).sum().item()
            total_tokens += non_pad.sum().item()

    avg_loss = epoch_loss / len(dataloader)
    accuracy = correct_tokens / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    return avg_loss, perplexity.item(), accuracy

# ---------- 主训练流程 ----------
def main():
    print(" 开始加载数据...")
    train_loader, val_loader, test_loader, vocab_en, vocab_de = get_dataloaders()

    torch.cuda.empty_cache()
    
    print(" 数据加载完成。")
    print(f"训练批次数: {len(train_loader)}, 验证批次数: {len(val_loader)}")

    print(" 构建 Transformer 模型...")

    enc = Encoder(
        INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, MAX_LEN
    )
    dec = Decoder(
        OUTPUT_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, MAX_LEN
    )
    model = Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    best_valid_loss = float("inf")

    train_losses = []
    val_losses = []

    print(" 开始训练...\n")
    for epoch in range(1, N_EPOCHS + 1):
        print(f"Epoch {epoch}/{N_EPOCHS}")

        # ----------------- 训练 -----------------
        train_loss = train_epoch(model, train_loader, optimizer, criterion, CLIP, device)
        torch.cuda.empty_cache()  #  训练结束后释放显存

        # ----------------- 验证 -----------------
        valid_loss, valid_ppl, valid_acc = evaluate(model, val_loader, criterion, device)
        print(f"Val Loss: {valid_loss:.3f} | Perplexity: {valid_ppl:.2f} | Accuracy: {valid_acc:.3f}")

        torch.cuda.empty_cache()  #  验证结束后释放显存

        train_losses.append(train_loss)
        val_losses.append(valid_loss)

        print(f"🟢 Epoch {epoch} | Train Loss: {train_loss:.3f} | Val Loss: {valid_loss:.3f}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "best_transformer.pt")
            print(" 保存最佳模型到 best_transformer.pt")
    # 训练完成后可视化
    plot_loss_curve(train_losses, val_losses)

    print(" 训练完成。")
# ----------------- 测试集评估 -----------------
    test_loss, test_ppl, test_acc = evaluate(model, test_loader, criterion, device)
    torch.cuda.empty_cache()  # 测试结束释放显存
    print(f"测试集结果 → Loss: {test_loss:.3f} | Perplexity: {test_ppl:.2f} | Accuracy: {test_acc:.3f}")




if __name__ == "__main__":
    main()
