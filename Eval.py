
# coding: utf-8

# In[38]:


import demo
from demo import BATCH_SIZE, DTYPE, use_gpu, git_hash

import sys
import glob
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

# optionally override
BATCH_SIZE = 10000 if use_gpu else BATCH_SIZE

dataset = demo.BoatDataset(demo.LABELS_FILE, demo.SOUNDS_FILE, test_set=True)
input_size = dataset[0][0].size(1)
print "input size:", input_size
print "batch size:", BATCH_SIZE


def evaluate(input_size, state_file, dataset, n=None):
    net = demo.BoatNet(input_size)
    if use_gpu:
        net.cuda()
    net.train(False)
    dataloader = demo.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=16 if use_gpu else 2,  # Note: GPU + multiprocessing issue here
        shuffle=True,
        pin_memory=use_gpu   # Only applicable if training on GPU
    )

    # load weights
    state_dict = torch.load(state_file)
    net.load_state_dict(state_dict)

    # results
    all_y = []
    all_y_pred = []
    for i, (x, y) in enumerate(dataloader):

        if use_gpu:
            x = x.cuda()
            y = y.cuda()

        # forward pass
        batch_size = x.size(0)  # batch is first dimension
        hidden = net.init_hidden(batch=batch_size)  # reset hidden layer!
        hidden.volatile=True  # set to volatile since don't care about gradient        
        y_pred = net(Variable(x, volatile=True), hidden).data
        y_pred = y_pred.cpu().numpy().argmax(axis=1)
        all_y.extend(y.tolist())
        all_y_pred.extend(y_pred.tolist())
        
        if i % 5 == 1:
            acc = np.sum(np.array(all_y) == all_y_pred) / float(len(all_y))
            print >> sys.stderr, "\r  evaluating @ %d accuracy: %.2f" % (i, acc * 100),
        
        if n and i == n:
            break

    # only-binary
    def comp(a, b):
        a = np.array(a)
        b = np.array(b)
        return ((a == 1) & (b == 1)) | \
               ((a == 2) & (b == 2)) | \
               (((a < 1) | (a > 2)) & ((b < 1) | (b > 2)))

    # multi-class
    # acc = np.sum(np.array(all_y) == all_y_pred) / float(len(all_y))

    # only-binary
    acc = np.sum(comp(all_y, all_y_pred)) / float(len(all_y))
    
    print >> sys.stderr
    return net, all_y, all_y_pred, acc


# state_files = sorted(glob.glob("model_0?_000028453.state"))
# state_files = sorted(glob.glob("model_0?_000000499.state"))

state_files = sorted(glob.glob("model_%s_?*.state" % (git_hash)))

# state_files = [sorted(glob.glob("model_*.state"))[0]]  # take first checkpoint
# state_files += sorted(glob.glob("model_0?_000007113.state"))  # add final per epoch

state_files.reverse()

print "Will evaluate:", state_files


results = []
n = 10000
with open("eval_%s.log" % (git_hash), "a") as f:
    for state_file in state_files:
        print "Evaluating", state_file
        net, all_y, all_y_pred, acc = evaluate(input_size, state_file, dataset, n=n)
        results.append((state_file, all_y, all_y_pred, acc))
        print "  %s: %.2f" % (state_file, acc * 100)
        print >> f, "%s,%.5f" % (state_file, acc)
        f.flush()

