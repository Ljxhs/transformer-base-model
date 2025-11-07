
# English-German Transformer Translation Model

## 项目简介

本项目实现了一个基于 **Transformer** 的 **英德（EN→DE）机器翻译模型**，包含多个实验变体（如 `NoLayerNorm`, `NoPosEnc`, `OneHead`），支持训练、验证、测试以及模型预测。

- 作者: Ljxhs
- Python 版本: 3.9+
- 框架: PyTorch 2.0+

---

## 功能与模块

| 文件                            | 功能说明                                           |
| ------------------------------- | -------------------------------------------------- |
| `config.py`                   | 模型与训练超参数配置                               |
| `dataset_loader.py`           | 数据加载与预处理（EN→DE）                         |
| `encoder.py` / `decoder.py` | Transformer 编码器和解码器组件                     |
| `modules.py`                  | Transformer 模块实现（MultiHeadAttention, FFN 等） |
| `transformer.py`              | Transformer 模型组合                               |
| `train.py`                    | 基础训练脚本                                       |
| `train_utils.py`              | 训练工具函数                                       |
| `utils.py`                    | 通用工具函数                                       |
| `run_no_layernorm.py`         | 训练 NoLayerNorm 变体模型                          |
| `run_no_posenc.py`            | 训练 NoPosEnc 变体模型                             |
| `run_one_head.py`             | 训练 OneHead 模型变体                              |
| `best_transformer.pt`         | 最优基础模型权重                                   |
| `model_NoPosEnc.pt`           | NoPosEnc 模型权重                                  |
| `loss_curve.png`              | 训练损失曲线示例                                   |

---

## 环境依赖

请先创建虚拟环境并安装依赖：

```bash
# 创建虚拟环境
conda create -n transformer_env python=3.9 -y
conda activate transformer_env

# 安装依赖
pip install -r requirements.txt
```

`requirements.txt` 已包含项目所需的 PyTorch、Transformers 等库。

---

## 数据说明

* 数据集使用 IWSLT17 英德平行语料
* 数据划分：

  * 训练集（train）
  * 验证集（dev）
  * 测试集（test）
* 数据加载路径需在 `dataset_loader.py` 中配置

---

## 模型训练

### 基础训练

```bash
python train.py --config config.py
```

### 实验变体训练

```bash
# NoLayerNorm
python run_no_layernorm.py

# NoPosEnc
python run_no_posenc.py

# OneHead
python run_one_head.py
```

训练完成后，模型权重将保存在当前目录（例如 `best_transformer.pt`）。

---

## 模型预测

```python
from transformer import Transformer
from dataset_loader import load_test_data
import torch

# 加载模型
model = Transformer(config)
model.load_state_dict(torch.load("best_transformer.pt"))
model.eval()

# 加载测试数据
test_data = load_test_data("data/test/")
predictions = model.predict(test_data)

# 输出前 5 条翻译示例
for src, pred in zip(test_data[:5], predictions[:5]):
    print(f"EN: {src} --> DE: {pred}")
```

---

## 文件夹结构建议

```
Transformer-EN-DE/
│
├── config.py
├── dataset_loader.py
├── encoder.py
├── decoder.py
├── modules.py
├── transformer.py
├── train.py
├── train_utils.py
├── utils.py
├── run_no_layernorm.py
├── run_no_posenc.py
├── run_one_head.py
├── requirements.txt
├── best_transformer.pt
├── model_NoPosEnc.pt
├── loss_curve.png
└── README.md
```

> 所有实验脚本均可在该根目录下直接运行。

---

## 运行环境与硬件

* Python 3.9+
* PyTorch 2.0+
* 建议使用 GPU（CUDA 11+）以加速训练

---

## GitHub 仓库

* 项目地址: [https://github.com/Ljxhs/transformer-base-model](https://github.com/Ljxhs/transformer-base-model)

---
