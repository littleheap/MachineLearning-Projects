import logging
import numpy as np
from collections import Counter
import torch.nn as nn
import torch
import code
import os
import nltk

'''
    辅助性方法
'''


# 加载数据
def load_data(in_file):
    cn = []
    en = []
    num_examples = 0
    with open(in_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split("\t")
            # nltk分词，添加BOS和EOS
            # 英文累加到en中
            en.append(["BOS"] + nltk.word_tokenize(line[0]) + ["EOS"])
            # 中文累加到cn中
            cn.append(["BOS"] + [c for c in line[1]] + ["EOS"])
    return en, cn


# 构建字典
def build_dict(sentences, max_words=50000):
    word_count = Counter()
    for sentence in sentences:
        for s in sentence:
            word_count[s] += 1
    # 选择最平凡出现的单词
    ls = word_count.most_common(max_words)
    total_words = len(ls) + 1
    # 为每个单词建立索引字典
    word_dict = {w[0]: index + 1 for (index, w) in enumerate(ls)}
    # unknow word
    word_dict["UNK"] = 0
    return word_dict, total_words


# 编码单词
def encode(en_sentences, cn_sentences, en_dict, cn_dict, sort_by_len=True):
    '''
        Encode the sequences. 
    '''
    length = len(en_sentences)
    out_en_sentences = []
    out_cn_sentences = []

    for i in range(length):
        en_seq = [en_dict[w] if w in en_dict else 0 for w in en_sentences[i]]
        cn_seq = [cn_dict[w] if w in cn_dict else 0 for w in cn_sentences[i]]
        out_en_sentences.append(en_seq)
        out_cn_sentences.append(cn_seq)

    # 按英文句子从短到长排序
    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    # 确认是否要从短到长排序
    if sort_by_len:
        sorted_index = len_argsort(out_en_sentences)
        out_en_sentences = [out_en_sentences[i] for i in sorted_index]
        out_cn_sentences = [out_cn_sentences[i] for i in sorted_index]

    return out_en_sentences, out_cn_sentences


# 获取minibatch对应内部索引
def get_minibatches(n, minibatch_size, shuffle=False):
    # 计算每一batch开头
    idx_list = np.arange(0, n, minibatch_size)
    # 是否混乱顺序
    if shuffle:
        np.random.shuffle(idx_list)
    minibatches = []
    # 获取每个minibatch序列索引
    for idx in idx_list:
        minibatches.append(np.arange(idx, min(idx + minibatch_size, n)))
    return minibatches


# 将minibatch序列转化为numpy矩阵
def prepare_data(seqs):
    # 获取每行seq的长度
    lengths = [len(seq) for seq in seqs]
    # batch size
    n_samples = len(seqs)
    # 获取该batch中最长序列的长度
    max_len = np.max(lengths)
    # 矩阵初始化为0
    x = np.zeros((n_samples, max_len)).astype('int32')
    x_mask = np.zeros((n_samples, max_len)).astype('float32')
    for idx, seq in enumerate(seqs):
        # x矩阵对应位置赋值seq数值信息
        x[idx, :lengths[idx]] = seq
        # x_mask矩阵记录矩阵中某一位置有没有有效信息
        x_mask[idx, :lengths[idx]] = 1.0
    return x, x_mask


# 打包batch
def gen_examples(en_sentences, cn_sentences, batch_size):
    # 获取每个minibatch及内部应该有的索引
    minibatches = get_minibatches(len(en_sentences), batch_size)
    all_ex = []
    for minibatch in minibatches:
        mb_en_sentences = [en_sentences[t] for t in minibatch]
        mb_cn_sentences = [cn_sentences[t] for t in minibatch]
        mb_x, mb_x_mask = prepare_data(mb_en_sentences)
        mb_y, mb_y_mask = prepare_data(mb_cn_sentences)
        all_ex.append((mb_x, mb_x_mask, mb_y, mb_y_mask))
    return all_ex


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        input = to_contiguous(input).view(-1, input.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        output = -input.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output
