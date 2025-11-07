import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
from tqdm import tqdm
from collections import Counter
from config import MAX_LEN, BATCH_SIZE  # 直接引用 config.py

# ==========================================================
#               基础分词与词典构建工具
# ==========================================================
special_symbols = ['<unk>', '<pad>', '<sos>', '<eos>']

def basic_tokenizer(text):
    """简单分词器：按空格切分并转为小写"""
    return text.lower().split()

def build_vocab(tokenized_sentences, min_freq=2):
    """构建词典"""
    word_counts = Counter()
    for sentence in tokenized_sentences:
        word_counts.update(sentence)
    filtered_words = [word for word, count in word_counts.items() if count >= min_freq]
    vocab = {word: i + len(special_symbols) for i, word in enumerate(filtered_words)}
    for i, symbol in enumerate(special_symbols):
        vocab[symbol] = i
    itos = {i: word for word, i in vocab.items()}
    return vocab, itos

def text_to_numerical(tokens, vocab):
    """将token序列转为索引序列"""
    return [vocab.get(token, vocab['<unk>']) for token in tokens]


# ==========================================================
#               PyTorch 数据集与批次函数
# ==========================================================
class TranslationDataset(Dataset):
    """英-德翻译数据集"""
    def __init__(self, src_data, trg_data):
        self.src_data = src_data
        self.trg_data = trg_data

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        src = torch.tensor(self.src_data[idx], dtype=torch.long)
        trg = torch.tensor(self.trg_data[idx], dtype=torch.long)
        return {'english': src, 'german': trg}

def collate_fn(batch, pad_idx):
    """按batch组装样本，并补齐序列长度"""
    src_batch = [b['english'][:MAX_LEN] for b in batch]  # 截断到 MAX_LEN
    trg_batch = [b['german'][:MAX_LEN] for b in batch]
    src_batch = pad_sequence(src_batch, padding_value=pad_idx, batch_first=True)
    trg_batch = pad_sequence(trg_batch, padding_value=pad_idx, batch_first=True)
    return {'english': src_batch, 'german': trg_batch}


# ==========================================================
#               数据加载主函数
# ==========================================================
def get_dataloaders(data_dir='e:\\研究生\\大模型\\data', min_freq=2):
    """
    加载 IWSLT2017 英德翻译数据集（本地路径）
    返回：train/val/test dataloader 以及 vocab
    """
    # -------------------- 1. 加载原始文本 --------------------
    def read_text_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if not line.strip().startswith('<')]
        return lines

    def parse_xml(file_path):
        import xml.etree.ElementTree as ET
        tree = ET.parse(file_path)
        root = tree.getroot()
        return [seg.text.strip() for seg in root.findall('.//seg') if seg.text]

    train_en = read_text_file(os.path.join(data_dir, 'train.tags.en-de.en'))
    train_de = read_text_file(os.path.join(data_dir, 'train.tags.en-de.de'))
    dev_en = parse_xml(os.path.join(data_dir, 'IWSLT17.TED.dev2010.en-de.en.xml'))
    dev_de = parse_xml(os.path.join(data_dir, 'IWSLT17.TED.dev2010.en-de.de.xml'))
    test_en = parse_xml(os.path.join(data_dir, 'IWSLT17.TED.tst2010.en-de.en.xml'))
    test_de = parse_xml(os.path.join(data_dir, 'IWSLT17.TED.tst2010.en-de.de.xml'))

    print(f" 数据加载完成: train={len(train_en)}, dev={len(dev_en)}, test={len(test_en)}")

    # -------------------- 2. 分词 --------------------
    print("🧩 分词中...")
    tokenized_train_en = [basic_tokenizer(s) for s in tqdm(train_en, desc='Tokenizing EN')]
    tokenized_train_de = [basic_tokenizer(s) for s in tqdm(train_de, desc='Tokenizing DE')]
    tokenized_dev_en = [basic_tokenizer(s) for s in dev_en]
    tokenized_dev_de = [basic_tokenizer(s) for s in dev_de]
    tokenized_test_en = [basic_tokenizer(s) for s in test_en]
    tokenized_test_de = [basic_tokenizer(s) for s in test_de]

    # -------------------- 3. 构建词典 --------------------
    print(" 构建词典中...")
    vocab_en, itos_en = build_vocab(tokenized_train_en, min_freq)
    vocab_de, itos_de = build_vocab(tokenized_train_de, min_freq)
    print(f"EN vocab size: {len(vocab_en)}, DE vocab size: {len(vocab_de)}")

    # -------------------- 4. 数值化 --------------------
    print(" 数值化中...")
    numericalized_train_en = [text_to_numerical(t, vocab_en) for t in tokenized_train_en]
    numericalized_train_de = [text_to_numerical(t, vocab_de) for t in tokenized_train_de]
    numericalized_dev_en = [text_to_numerical(t, vocab_en) for t in tokenized_dev_en]
    numericalized_dev_de = [text_to_numerical(t, vocab_de) for t in tokenized_dev_de]
    numericalized_test_en = [text_to_numerical(t, vocab_en) for t in tokenized_test_en]
    numericalized_test_de = [text_to_numerical(t, vocab_de) for t in tokenized_test_de]

    # -------------------- 5. 构造 Dataset / DataLoader --------------------
    PAD_IDX = vocab_de['<pad>']

    train_dataset = TranslationDataset(numericalized_train_en, numericalized_train_de)
    val_dataset = TranslationDataset(numericalized_dev_en, numericalized_dev_de)
    test_dataset = TranslationDataset(numericalized_test_en, numericalized_test_de)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_fn(b, PAD_IDX))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b: collate_fn(b, PAD_IDX))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b: collate_fn(b, PAD_IDX))

    print(f" DataLoader 创建完成: train={len(train_loader)} batches")

    return train_loader, val_loader, test_loader, vocab_en, vocab_de
