"""This package options includes option modules: training options, test options, and basic options (used in both training and test)."""

import torch
import random
import numpy as np 

seed = 666 

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False