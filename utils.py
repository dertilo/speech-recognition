import os
import random

import numpy as np
import torch
import torch.cuda
import torch.utils.data.distributed

BLANK_SYMBOL = "■"
SPACE = ' '

USE_GPU = torch.cuda.is_available()
HOME = os.environ["HOME"]

def set_seeds(seed): # TODO(tilo): unused?
    torch.manual_seed(seed)
    if USE_GPU:
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)