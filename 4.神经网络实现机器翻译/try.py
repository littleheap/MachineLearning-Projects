import logging
import numpy as np
from collections import Counter
import torch.nn as nn
import torch
import code
import os
import nltk


def load_data(in_file):
    cn = []
    en = []
    num_examples = 0
    with open(in_file, 'r') as f:
        for line in f:
            line = line.strip().split("\t")
            en.append(["BOS"] + nltk.word_tokenize(line[0]) + ["EOS"])
            # split chinese sentence into characters
            cn.append(["BOS"] + [c for c in line[1]] + ["EOS"])
    return en, cn
