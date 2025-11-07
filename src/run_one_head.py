import torch
import torch.optim as optim
import torch.nn as nn
from train_utils import train_epoch, evaluate, build_model
from dataset_loader import get_dataloaders
from config import *

print("\n==============================")
print(" 运行实验: OneHead（单头注意力）")
print("==============================\n")

train_loader, val_loader, _, _, _ = get_dataloaders()
model, device = build_model(n_heads=1)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

best_val = float("inf")
for epoch in range(1, N_EPOCHS + 1):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, CLIP, device)
    val_loss = evaluate(model, val_loader, criterion, device)
    print(f"Epoch {epoch}: Train={train_loss:.3f}, Val={val_loss:.3f}")
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "model_OneHead.pt")

print(f"OneHead 最佳验证 Loss={best_val:.3f}")
