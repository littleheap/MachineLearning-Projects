import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd import Variable
import code
import math
import sys
import numpy as np


# encode English(matrix) -> decode Chinese(matrix)
class EncoderDecoderModel(nn.Module):
    def __init__(self, args):
        super(EncoderDecoderModel, self).__init__()

        # hidden size
        self.nhid = args.hidden_size

        # 转化词向量
        self.embed_en = nn.Embedding(args.en_total_words, args.embedding_size)
        self.embed_cn = nn.Embedding(args.cn_total_words, args.embedding_size)

        # encode & decoder 操作函数
        self.encoder = nn.LSTMCell(args.embedding_size, args.hidden_size)
        self.decoder = nn.LSTM(args.embedding_size, args.hidden_size, batch_first=True)

        # 输出层线性转换，投射到中文词向量上
        self.linear = nn.Linear(self.nhid, args.cn_total_words)
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-0.1, 0.1)

        # W初始化
        self.embed_en.weight.data.uniform_(-0.1, 0.1)
        self.embed_cn.weight.data.uniform_(-0.1, 0.1)

    # hidden vector
    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(bsz, self.nhid).zero_()),
                Variable(weight.new(bsz, self.nhid).zero_()))

    # 前进算法
    def forward(self, x, x_mask, y, hidden):
        # x : B * T tensor
        # x_mask : B * T tensor
        # y : B * J tensor

        # 对x进行embed
        x_embedded = self.embed_en(x)
        # 获取embed矩阵尺寸
        B, T, embedding_size = x_embedded.size()
        # 对y进行embed
        y_embedded = self.embed_cn(y)

        # encoder操作
        hiddens = []
        cells = []
        for i in range(T):
            hidden = self.encoder(x_embedded[:, i, :], hidden)
            hiddens.append(hidden[0].unsqueeze(1))
            cells.append(hidden[1].unsqueeze(1))

        hiddens = torch.cat(hiddens, 1)
        cells = torch.cat(cells, 1)
        x_lengths = x_mask.sum(1).unsqueeze(2).expand(B, 1, embedding_size) - 1
        h = hiddens.gather(1, x_lengths).permute(1, 0, 2)
        c = cells.gather(1, x_lengths).permute(1, 0, 2)

        # decoder操作
        hiddens, (h, c) = self.decoder(y_embedded, hx=(h, c))

        hiddens = hiddens.contiguous()

        # hiddens : B * J * hidden_size vector
        # output输出层：decode输入，线性变化获得输出
        decoded = self.linear(hiddens.view(hiddens.size(0) * hiddens.size(1), hiddens.size(2)))
        # softmax获得概率分布
        decoded = F.log_softmax(decoded)  # (B*J) * cn_total_words

        return decoded.view(hiddens.size(0), hiddens.size(1), decoded.size(1)), hiddens

    # 翻译函数
    def translate(self, x, x_mask, y, hidden, max_length=20):
        x_embedded = self.embed_en(x)
        B, T, embedding_size = x_embedded.size()
        # encoder
        hiddens = []
        cells = []
        for i in range(T):
            hidden = self.encoder(x_embedded[:, i, :], hidden)
            hiddens.append(hidden[0].unsqueeze(1))
            cells.append(hidden[1].unsqueeze(1))

        hiddens = torch.cat(hiddens, 1)
        cells = torch.cat(cells, 1)
        x_lengths = x_mask.sum(1).unsqueeze(2).expand(B, 1, embedding_size) - 1
        h = hiddens.gather(1, x_lengths).permute(1, 0, 2)
        c = cells.gather(1, x_lengths).permute(1, 0, 2)

        pred = [y]
        for i in range(max_length - 1):
            y_embedded = self.embed_cn(y)
            hiddens, (h, c) = self.decoder(y_embedded, hx=(h, c))
            hiddens = hiddens.contiguous()
            # output layer
            decoded = self.linear(hiddens.view(hiddens.size(0) * hiddens.size(1), hiddens.size(2)))
            decoded = F.log_softmax(decoded)
            decoded = decoded.view(hiddens.size(0), decoded.size(1))
            y = torch.max(decoded, 1)[1]
            pred.append(y)
        return torch.cat(pred, 1)
