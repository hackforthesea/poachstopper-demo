#! /usr/bin/env python

import csv
import cPickle as pickle
import numpy as np
import random
import subprocess

import os
import sys
import time

import matplotlib  # if necessary: matplotlib.use('tkagg') or export MPLBACKEND=tkagg
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

LABELS_FILE = "data.csv"
SOUNDS_FILE = "sounds-spectrogram.pkl"
LOW_PASS_FILTER = slice(0, 63)  # or slice(0, None) for no filter
INPUT_SIZE = 13  # number of features
N_CLASSES = 49  # 3 or 49
HIDDEN_SIZE = 128
BATCH_SIZE = 512 #  up to 8000 on `p2` GPU
NUM_EPOCHS = 100

CONTINUE_FROM = None  # (model_state_filename, epoch, batch)

TIME_SINGLE_EPOCH = False

# Use GPU or not
use_gpu = True and torch.cuda.is_available()
print "using GPU?", use_gpu
if use_gpu:
    DTYPE = torch.cuda.FloatTensor
else:
    DTYPE = torch.FloatTensor
DTYPE_CPU = torch.FloatTensor

# Get git hash
git_hash = subprocess.check_output(["git", "rev-parse", "--short", "--verify", "HEAD"]).strip()
print "Git hash of HEAD:", git_hash

# model
class BoatNet(torch.nn.Module):

    def __init__(self, input_size, n_classes=N_CLASSES, hidden_size=HIDDEN_SIZE):
        super(BoatNet, self).__init__()
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.num_layers = 3
        self.gru = torch.nn.GRU(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.dropout = torch.nn.Dropout(0.2)  # shared dropout layer
        self.linear0 = torch.nn.Linear(
            in_features=self.hidden_size,
            out_features=self.hidden_size
        )
        self.linear = torch.nn.Linear(
            in_features=self.hidden_size,
            out_features=self.n_classes
        )
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x, hidden=None):
        x, _ = self.gru(x, hidden)
        x = x[:, -1]  # take last time step
        x = self.dropout(x)
        x = self.linear0(x)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.log_softmax(x)
        return x

    def init_hidden(self, batch=1):
        # regardless of `batch_first`: (num_layers, batch, num_features)
        return Variable(torch.zeros(self.num_layers, batch, self.hidden_size).type(DTYPE))


# data preparation
class BoatDataset(Dataset):

    def __init__(self, csv_file=LABELS_FILE, sounds_file=SOUNDS_FILE, window=20, test_set=False, low_pass_filter=LOW_PASS_FILTER, multiclass=False):
        self.csv_file = csv_file
        self.sounds_file = sounds_file
        self.window = window
        self.low_pass_filter = low_pass_filter
        self.multiclass = multiclass
        self.is_test_set = test_set
        with open(csv_file) as f:
            self.csv_contents = [[x[0], int(x[1]), int(x[2])] for x in csv.reader(f)]
        with open(sounds_file) as f:
            self.sounds = pickle.load(f)
        assert all([x[0] in self.sounds for x in self.csv_contents]), "not all sounds exist"

        # split into random short sequences
        self.data = []
        for i, row in enumerate(self.csv_contents):
            basename = row[0]
            label = row[2] if self.multiclass else row[1]
            n = self.sounds[basename].shape[1]

            # split into test/train
            prng = random.Random(n)  # use length as seed
            in_test = False
            for j in range(n - self.window):

                # split in contiguous blocks of window x 5
                if j % (window * 5) == 0:
                    in_test = prng.random() > 0.80

                # assign to test/train
                if self.is_test_set == in_test:
                    self.data.append((i, j, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        basename_idx, offset, label = self.data[idx]
        basename = self.csv_contents[basename_idx][0]
        clip = self.sounds[basename][self.low_pass_filter, offset:offset + self.window].copy()
        clip = clip.transpose(1, 0)
        tensor = torch.Tensor(clip).type(DTYPE_CPU)
        return (tensor, label)

if __name__ == "__main__":
    # Test state of this file in git
    status = subprocess.check_output(["git", "status", "--porcelain", "demo.py"])
    if status.strip():
        if os.getenv("DEBUG", False):
            print "DEBUGGING mode on... allowing unstaged changes"
        else:
            print "ERROR: Unstaged changes in demo.py; please commit first."
            sys.exit(1)

    dataset = BoatDataset(LABELS_FILE, SOUNDS_FILE, test_set=False, multiclass=True)
    input_size = dataset[0][0].size(1)
    net = BoatNet(input_size)
    # net = torch.nn.DataParallel(net)
    # net.init_hidden = list(net.children())[0].init_hidden
    if use_gpu:
        # move all registered parameter weights and buffers of model to gpu memory.
        # also, do this immediately after instantiating (before referencing while
        # creating optimizer, for instance):
        #   This also makes associated parameters and buffers different objects. So
        #   it should be called before constructing optimizer if the module will
        #   live on GPU while being optimized.
        net.cuda()
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=16 if use_gpu else 2,  # Note: GPU + multiprocessing issue here
        shuffle=True,
        pin_memory=use_gpu   # Only applicable if training on GPU
    )

    # loss
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    # training loop
    with open("training_%s.log" % (git_hash), "a") as f:
        writer = csv.writer(f)
        start_epoch = 0
        start_i = 0
        if CONTINUE_FROM:
            net.load_state_dict(torch.load(CONTINUE_FROM[0]))
            start_epoch = CONTINUE_FROM[1]
            start_i = CONTINUE_FROM[2]
        for epoch in range(start_epoch, NUM_EPOCHS):
            print "starting epoch", epoch
            recent_losses = []
            for i, (x, y) in enumerate(dataloader, start_i):

                if use_gpu:
                    x = x.cuda()
                    y = y.cuda()

                if (i + 1) % 500 == 0:
                    print "saving"
                    torch.save(net.state_dict(), "model_%s_%02d_%09d.state" % (git_hash, epoch, i))

                # Time single epoch (for optimizing GPU utilization)
                if TIME_SINGLE_EPOCH:
                    if i == 2:
                        t0 = time.time()

                # forward pass
                x, y = Variable(x), Variable(y)
                batch_size = x.size(0)  # batch is first dimension
                hidden = net.init_hidden(batch=batch_size)
                y_pred = net(x, hidden)

                # y_pred is output from model (so may be in GPU mem)
                # y must match cuda state
                if use_gpu:
                    y = y.cuda()

                loss = criterion(y_pred, y)

                # zero grad and take step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_v = float(loss.cpu().data[0])  # probably don't need cpu()
                recent_losses.append(loss_v)
                recent_losses = recent_losses[-100:]

                # print loss
                print "%s: epoch: %d, batch: %d, loss: %.5f, rolling average loss: %.5f" % (git_hash, epoch, i, loss_v, sum(recent_losses) / len(recent_losses))

                # log
                writer.writerow([epoch, i, loss_v])
                f.flush()

            # Time single epoch for testing and optimizing GPU utilization
            if TIME_SINGLE_EPOCH:
                t1 = time.time()
                print "Took: %.3s" % (t1 - t0)
                sys.exit()

            print "saving"
            torch.save(net.state_dict(), "model_%s_%02d_%09d.state" % (git_hash, epoch, i))
