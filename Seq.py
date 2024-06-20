import os
import re
import jieba
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence


# 数据预处理
def read_inf_file(inf_path):
    with open(inf_path, 'r', encoding='gbk', errors='ignore') as f:
        novels = f.read().strip().split(',')
    return novels


def read_novel(file_path):
    with open(file_path, 'r', encoding='gbk', errors='ignore') as f:
        text = f.read()
    return text


def read_stopwords(stopwords_path):
    with open(stopwords_path, 'r', encoding='gbk', errors='ignore') as f:
        stopwords = f.read().strip().split()
    return set(stopwords)


def clean_text(text):
    text = re.sub(r'[^\u4e00-\u9fa5。！，；：？]', '', text)
    return text


def tokenize_and_remove_stopwords(text, stopwords):
    words = jieba.lcut(text)
    words = [word for word in words if word not in stopwords]
    return words


def preprocess_data(novel_folder, inf_file, stopwords_file):
    novels = read_inf_file(os.path.join(novel_folder, inf_file))
    stopwords = read_stopwords(os.path.join(novel_folder, stopwords_file))
    sentences = []

    for novel in novels:
        novel_path = os.path.join(novel_folder, f'{novel}.txt')
        text = read_novel(novel_path)
        text = clean_text(text)
        words = tokenize_and_remove_stopwords(text, stopwords)
        # 将文本分割成句子
        sentences.extend(re.split(r'。|！|？', ''.join(words)))

    return sentences


# 文件夹和文件路径
novel_folder = 'yuliaoku'
inf_file = 'inf.txt'
stopwords_file = 'cn_stopwords.txt'
corpus = preprocess_data(novel_folder, inf_file, stopwords_file)

# 构建词汇表
counter = Counter(''.join(corpus))
specials = ['<pad>', '<unk>', '<sos>', '<eos>']
vocab = {word: idx for idx, (word, _) in enumerate(counter.most_common(), len(specials))}
vocab.update({special: idx for idx, special in enumerate(specials)})

# 反向词汇表
itos = {idx: word for word, idx in vocab.items()}


def encode_text(text, vocab):
    return [vocab['<sos>']] + [vocab.get(token, vocab['<unk>']) for token in text] + [vocab['<eos>']]


def decode_text(indices, itos):
    tokens = [itos[idx] for idx in indices]
    return ''.join(tokens).replace('<sos>', '').replace('<eos>', '')


encoded_corpus = [encode_text(sentence, vocab) for sentence in corpus if sentence]


# 创建数据加载器
def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, padding_value=vocab['<pad>'], batch_first=True)
    trg_batch = pad_sequence(trg_batch, padding_value=vocab['<pad>'], batch_first=True)
    return src_batch, trg_batch


def create_dataloader(encoded_corpus, batch_size):
    dataset = [(torch.tensor(encoded_corpus[i]), torch.tensor(encoded_corpus[i + 1])) for i in
               range(len(encoded_corpus) - 1)]
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    return dataloader


batch_size = 64
dataloader = create_dataloader(encoded_corpus, batch_size)


# 定义Seq2Seq模型
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        trg_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        hidden, cell = self.encoder(src)

        input = trg[0, :]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            top1 = output.argmax(1)
            input = trg[t] if random.random() < teacher_forcing_ratio else top1

        return outputs


# 定义模型参数
input_dim = len(vocab)
output_dim = len(vocab)
emb_dim = 256
hidden_dim = 512
n_layers = 2
dropout = 0.5

# 创建模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = Encoder(input_dim, emb_dim, hidden_dim, n_layers, dropout)
decoder = Decoder(output_dim, emb_dim, hidden_dim, n_layers, dropout)
seq2seq_model = Seq2Seq(encoder, decoder, device).to(device)

# 定义损失函数和优化器
optimizer = optim.Adam(seq2seq_model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])

# 训练模型
n_epochs = 10

for epoch in range(n_epochs):
    seq2seq_model.train()
    epoch_loss = 0

    for src, trg in dataloader:
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()
        output = seq2seq_model(src, trg)

        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f'Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader)}')


# 文本生成
def generate_text_seq2seq(model, start_text, max_len, vocab, itos, device):
    model.eval()
    tokens = tokenize_and_remove_stopwords(clean_text(start_text), stopwords)
    src_indices = encode_text(tokens, vocab)
    src_tensor = torch.tensor(src_indices).unsqueeze(1).to(device)

    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)

    trg_indices = [vocab['<sos>']]

    for _ in range(max_len):
        trg_tensor = torch.tensor([trg_indices[-1]]).unsqueeze(1).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
            top1 = output.argmax(1).item()

        trg_indices.append(top1)
        if top1 == vocab['<eos>']:
            break

    return decode_text(trg_indices, itos)


start_text = '天色已晚，'
max_len = 100
generated_text = generate_text_seq2seq(seq2seq_model, start_text, max_len, vocab, itos, device)
print("Seq2Seq生成的文本：", generated_text)
