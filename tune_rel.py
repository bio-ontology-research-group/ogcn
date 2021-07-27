from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler


def load_data():
    train_data = GraphDataset(g, train_df, train_labels, annots, prot_idx)
    test_data = GraphDataset(g, test_df, test_labels, annots, prot_idx)

    train_set_batches = get_batches(train_data, batch_size)
    test_set_batches = get_batches(test_data, batch_size)

    return train_set_batches, test_set_batches


