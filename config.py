# import torch

# # 模型超参数
# INPUT_DIM = 59616
# OUTPUT_DIM = 81512
# HID_DIM = 256
# ENC_LAYERS = 3
# DEC_LAYERS = 3
# ENC_HEADS = 8
# DEC_HEADS = 8
# ENC_PF_DIM = 512
# DEC_PF_DIM = 512
# ENC_DROPOUT = 0.1
# DEC_DROPOUT = 0.1
# MAX_LEN = 300

# # 训练超参数
# CLIP = 1
# LEARNING_RATE = 1e-4
# N_EPOCHS = 10
# BATCH_SIZE = 128

# # pad 索引
# SRC_PAD_IDX = 1
# TRG_PAD_IDX = 1

# # 设备
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


import torch

# =======================
#      模型超参数
# =======================
INPUT_DIM = 59616       # 英文词表大小
OUTPUT_DIM = 81512      # 德文词表大小
HID_DIM = 32           # 隐藏层维度
ENC_LAYERS = 2          # Encoder 层数
DEC_LAYERS = 2          # Decoder 层数
ENC_HEADS = 2           # Multi-Head Attention 头数
DEC_HEADS = 2
ENC_PF_DIM = 128        # 前馈网络维度
DEC_PF_DIM = 128
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
MAX_LEN = 100           # 限制序列长度

# =======================
#      训练超参数
# =======================
CLIP = 1
LEARNING_RATE = 1e-4
N_EPOCHS = 5
BATCH_SIZE = 8         # 减小 batch size，显存友好

# =======================
#      pad 索引
# =======================
SRC_PAD_IDX = 1
TRG_PAD_IDX = 1

# =======================
#      设备
# =======================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
