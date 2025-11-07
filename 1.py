import xml.etree.ElementTree as ET
import os

# --- 1. 定义文件路径 ---
data_dir = 'e:\\研究生\\大模型\\data'
train_en_file = os.path.join(data_dir, 'train.tags.en-de.en')
train_de_file = os.path.join(data_dir, 'train.tags.en-de.de')
dev_en_xml = os.path.join(data_dir, 'IWSLT17.TED.dev2010.en-de.en.xml')
dev_de_xml = os.path.join(data_dir, 'IWSLT17.TED.dev2010.en-de.de.xml')
test_en_xml = os.path.join(data_dir, 'IWSLT17.TED.tst2010.en-de.en.xml')
test_de_xml = os.path.join(data_dir, 'IWSLT17.TED.tst2010.en-de.de.xml')

# --- 2. 定义数据加载函数 ---

# 用于解析XML文件 (dev/test sets)
def parse_xml(file_path):
    """解析IWSLT的XML文件，提取句子"""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        sentences = []
        for seg in root.findall('.//seg'):
            sentences.append(seg.text.strip() if seg.text else "")
        return sentences
    except ET.ParseError as e:
        print(f"Error parsing {file_path}: {e}")
        return []

def read_text_file(file_path):
    """按行读取纯文本文件，并过滤掉XML标签行"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 过滤掉以'<'开头的行，这些是元数据标签
            sentences = [line.strip() for line in f if not line.strip().startswith('<')]
        return sentences
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return []
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

# --- 3. 加载所有数据集 ---
print("开始加载数据...")

# 加载训练数据
train_en = read_text_file(train_en_file)
train_de = read_text_file(train_de_file)

# 加载验证数据
dev_en = parse_xml(dev_en_xml)
dev_de = parse_xml(dev_de_xml)

# 加载测试数据
test_en = parse_xml(test_en_xml)
test_de = parse_xml(test_de_xml)

# --- 4. 检查加载结果 ---
if train_en and dev_en and test_en:
    print("\n数据加载成功!")
    print(f"训练集句子数量: {len(train_en)}")
    print(f"验证集句子数量: {len(dev_en)}")
    print(f"测试集句子数量: {len(test_en)}")

    # 确保每个数据集的源语言和目标语言句子数量匹配
    assert len(train_en) == len(train_de), "训练集语言对数量不匹配!"
    assert len(dev_en) == len(dev_de), "验证集语言对数量不匹配!"
    assert len(test_en) == len(test_de), "测试集语言对数量不匹配!"

    print("\n--- 数据样本 ---")
    print("训练 (EN):", train_en[0])
    print("训练 (DE):", train_de[0])
    print("\n验证 (EN):", dev_en[0])
    print("验证 (DE):", dev_de[0])
else:
    print("\n数据加载失败。请检查文件路径和文件内容是否正确。")


# --- 5. (替代方案) 文本分词 (不使用spaCy) ---
from collections import Counter
from tqdm import tqdm

print("\n开始使用基础方法进行分词...")

# 简单的分词函数：按空格切分并转为小写
def basic_tokenizer(text):
    return text.lower().split()

# 对所有数据进行分词
tokenized_train_en = [basic_tokenizer(sent) for sent in tqdm(train_en, desc="Tokenizing EN Train")]
tokenized_train_de = [basic_tokenizer(sent) for sent in tqdm(train_de, desc="Tokenizing DE Train")]
tokenized_dev_en = [basic_tokenizer(sent) for sent in tqdm(dev_en, desc="Tokenizing EN Dev")]
tokenized_dev_de = [basic_tokenizer(sent) for sent in tqdm(dev_de, desc="Tokenizing DE Dev")]
tokenized_test_en = [basic_tokenizer(sent) for sent in tqdm(test_en, desc="Tokenizing EN Test")]
tokenized_test_de = [basic_tokenizer(sent) for sent in tqdm(test_de, desc="Tokenizing DE Test")]

print("\n分词完成!")
print("--- 分词样本 ---")
print("原始 (EN):", train_en[0])
print("分词后 (EN):", tokenized_train_en[0])


# --- 6. (替代方案) 构建词典 (不使用torchtext) ---

print("\n开始手动构建词典...")

# 定义特殊符号
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

def build_vocab(tokenized_sentences, min_freq):
    word_counts = Counter()
    for sentence in tokenized_sentences:
        word_counts.update(sentence)
    
    # 过滤低频词
    filtered_words = [word for word, count in word_counts.items() if count >= min_freq]
    
    # 创建词典
    vocab = {word: i + len(special_symbols) for i, word in enumerate(filtered_words)}
    
    # 添加特殊符号
    for i, symbol in enumerate(special_symbols):
        vocab[symbol] = i
        
    return vocab, {i: word for word, i in vocab.items()} # 返回 word-to-index 和 index-to-word

# 构建词典
vocab_de, itos_de = build_vocab(tokenized_train_de, min_freq=2)
vocab_en, itos_en = build_vocab(tokenized_train_en, min_freq=2)

UNK_IDX_DE = vocab_de['<unk>']
UNK_IDX_EN = vocab_en['<unk>']

print("\n词典构建完成!")
print(f"德语词典大小: {len(vocab_de)}")
print(f"英语词典大小: {len(vocab_en)}")

# --- 7. 检查词典 ---
print("\n--- 词典映射示例 (DE) ---")
print("'vielen' ->", vocab_de.get('vielen', UNK_IDX_DE))
print("'dank' ->", vocab_de.get('dank', UNK_IDX_DE))
print("'unbekanntes_wort' ->", vocab_de.get('unbekanntes_wort', UNK_IDX_DE)) # 未知词

print("\n--- 索引到词元示例 (EN) ---")
print("Index 0 ->", itos_en[0])
print("Index 1 ->", itos_en[1])
print("Index 100 ->", itos_en[100])

# --- 8. 文本数值化 ---
def numericalize(tokenized_sentence, vocab, unk_idx):
    return [vocab.get(token, unk_idx) for token in tokenized_sentence]

sentence_en = tokenized_train_en[0]
numericalized_sentence = numericalize(sentence_en, vocab_en, UNK_IDX_EN)
print("\n--- 文本数值化示例 ---")
print("原始分词句子 (EN):", sentence_en)
print("数值化后:", numericalized_sentence)

import torch


from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# --- 0. 确保 text_to_numerical 函数已定义 ---
def text_to_numerical(tokens, vocab):
    return [vocab.get(token, vocab['<unk>']) for token in tokens]

# --- 1. 定义 Dataset ---
class TranslationDataset(Dataset):
    def __init__(self, src_data, trg_data):
        self.src_data = src_data
        self.trg_data = trg_data

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        # 将数值化后的列表转换为 Tensor
        src_sample = torch.tensor(self.src_data[idx], dtype=torch.long)
        trg_sample = torch.tensor(self.trg_data[idx], dtype=torch.long)
        return {'english': src_sample, 'german': trg_sample} 

# --- 2. 数值化所有数据 ---
print("开始数值化所有数据...")
numericalized_train_en = [text_to_numerical(tokens, vocab_en) for tokens in tokenized_train_en]
numericalized_train_de = [text_to_numerical(tokens, vocab_de) for tokens in tokenized_train_de]
numericalized_dev_en = [text_to_numerical(tokens, vocab_en) for tokens in tokenized_dev_en]
numericalized_dev_de = [text_to_numerical(tokens, vocab_de) for tokens in tokenized_dev_de]
numericalized_test_en = [text_to_numerical(tokens, vocab_en) for tokens in tokenized_test_en]
numericalized_test_de = [text_to_numerical(tokens, vocab_de) for tokens in tokenized_test_de]
print("数值化完成!")

# --- 3. 创建 Dataset 实例 ---
train_dataset = TranslationDataset(numericalized_train_en, numericalized_train_de) 
val_dataset = TranslationDataset(numericalized_dev_en, numericalized_dev_de)
test_dataset = TranslationDataset(numericalized_test_en, numericalized_test_de)

print(f"训练集样本数: {len(train_dataset)}")
print(f"验证集样本数: {len(val_dataset)}")

# --- 4. 定义 collate_fn 以处理批次数据 ---
PAD_IDX = vocab_de['<pad>'] # 德语和英语的<pad>索引都是1，使用哪个都可以

def collate_fn(batch):
    src_batch, trg_batch = [], []
    for sample in batch:
        src_batch.append(sample['english'])
        trg_batch.append(sample['german'])
    
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    trg_batch = pad_sequence(trg_batch, padding_value=PAD_IDX, batch_first=True)
    return {'english': src_batch, 'german': trg_batch}


# --- 5. 创建 DataLoader 实例 ---
BATCH_SIZE = 128

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

print(f"\n创建了 {len(train_dataloader)} 个训练批次，每个批次大小为 {BATCH_SIZE}")

# --- 6. 检查一个批次的数据 --- 
batch = next(iter(train_dataloader))
src_batch = batch['english'] # 正确获取英文张量
trg_batch = batch['german']  # 正确获取德文张量
print(f"\n--- 单个批次数据形状 ---")
print(f"源语言批次形状: {src_batch.shape}")
print(f"目标语言批次形状: {trg_batch.shape}")

print(f"\n源语言批次第一个样本 (填充后):\n{src_batch[0]}")
print(f"目标语言批次第一个样本 (填充后):\n{trg_batch[0]}")



import torch
import torch.nn as nn
import torch.optim as optim

# 导入之前定义的模块
from modules import MultiHeadAttention, PositionwiseFeedforward, PositionalEncoding, LayerNorm
from encoder import EncoderLayer, Encoder
from decoder import DecoderLayer, Decoder
from transformer import Transformer
from utils import make_src_mask, make_trg_mask


# 定义超参数
INPUT_DIM = len(vocab_en) # 源语言词汇表大小
OUTPUT_DIM = len(vocab_de) # 目标语言词汇表大小
HID_DIM = 256 # 嵌入维度和模型内部维度
ENC_LAYERS = 3 # 编码器层数
DEC_LAYERS = 3 # 解码器层数
ENC_HEADS = 8 # 编码器多头注意力头数
DEC_HEADS = 8 # 解码器多头注意力头数
ENC_PF_DIM = 512 # 编码器前馈网络维度
DEC_PF_DIM = 512 # 解码器前馈网络维度
ENC_DROPOUT = 0.1 # 编码器 dropout
DEC_DROPOUT = 0.1 # 解码器 dropout
MAX_LEN = 300 # 新增：序列的最大长度，用于位置编码

SRC_PAD_IDX = vocab_en['<pad>'] # 源语言填充符索引
TRG_PAD_IDX = vocab_de['<pad>'] # 目标语言填充符索引

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 实例化编码器和解码器
enc = Encoder(
    INPUT_DIM,
    HID_DIM,
    ENC_LAYERS,
    ENC_HEADS,
    ENC_PF_DIM,
    ENC_DROPOUT,
    MAX_LEN, # 新增：传递 MAX_LEN
)

dec = Decoder(
    OUTPUT_DIM,
    HID_DIM,
    DEC_LAYERS,
    DEC_HEADS,
    DEC_PF_DIM,
    DEC_DROPOUT,
    MAX_LEN, # 新增：传递 MAX_LEN
)

# 实例化 Transformer 模型
model = Transformer(
    enc,
    dec,
    SRC_PAD_IDX,
    TRG_PAD_IDX,
    device
).to(device)

# 打印模型参数数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

# 简单的前向传播测试
# 假设一个批次的数据
# src = [batch size, src len]
# trg = [batch size, trg len]

batch = next(iter(train_dataloader))
src = batch['english'].to(device)
trg = batch['german'].to(device)

print(f"Source shape: {src.shape}")
print(f"Target shape: {trg.shape}")

output, attention = model(src, trg[:, :-1]) # 解码器输入不包含 <eos>

print(f"Output shape: {output.shape}") # [batch size, trg len - 1, output dim]
print(f"Attention shape: {attention.shape}") # [batch size, n heads, trg len - 1, src len]

