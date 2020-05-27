import os
import random

import numpy as np
import torch
import torch.cuda
import torch.distributed as dist
import torch.utils.data.distributed

BLANK_SYMBOL = "■"
SPACE = ' '

USE_GPU = torch.cuda.is_available()
HOME = os.environ["HOME"]

def set_seeds(seed):
    torch.manual_seed(seed)
    if USE_GPU:
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def unflatten_targets(targets, target_sizes):
    split_targets = []
    offset = 0
    for size in target_sizes:
        split_targets.append(targets[offset: offset + size])
        offset += size
    return split_targets