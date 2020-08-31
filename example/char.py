from mingpt.utils import sample
import os
from mingpt.trainer import Trainer, TrainerConfig
from mingpt.model import GPT, GPTConfig
from torch.utils.data import Dataset
import math
from torch.nn import functional as F
from mingpt.utils import set_seed
import re
import requests
import io
import tarfile
import csv
import torch
import torch.nn as nn
import random
import sys
import concurrent.futures
import time
from collections import Counter
from collections import namedtuple

import torch
import nestedtensor
# set up logging
import logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# make deterministic
set_seed(42)


class CharDataset(Dataset):

    def __init__(self, data, block_size):
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


def get_data(URL):
    r = requests.get(URL)
    file_like_object = io.BytesIO(r.content)
    return io.TextIOWrapper(file_like_object).read()


if __name__ == "__main__":
    block_size = 128  # spatial extent of the model for its context
    URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    # don't worry we won't run out of file handles
    text = get_data(URL)
    # one line of poem is roughly 50 characters
    train_dataset = CharDataset(text, block_size)

    model_file_path = "model.trained"
    if os.path.exists(model_file_path):
        model = torch.load(model_file_path)
    else:
        mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                          n_layer=8, n_head=8, n_embd=512)
        model = GPT(mconf)

        # initialize a trainer instance and kick off training
        tconf = TrainerConfig(max_epochs=2, batch_size=128, learning_rate=6e-4,
                              lr_decay=True, warmup_tokens=128*20, final_tokens=2*len(train_dataset)*block_size,
                              num_workers=4)
        trainer = Trainer(model, train_dataset, None, tconf)
        trainer.train()
        torch.save(model, model_file_path)

    context = "O God, O God!"
    x = torch.tensor([train_dataset.stoi[s] for s in context],
                     dtype=torch.long)[None, ...].to(trainer.device)
    y = sample(model, x, 2000, temperature=1.0, sample=True, top_k=10)[0]
    completion = ''.join([train_dataset.itos[int(i)] for i in y])
    print(completion)
