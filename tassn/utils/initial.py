# -*- coding: utf-8 -*-

import torch
import os
import random
import numpy as np

# --Create File Path--
def create_path(path):
    if type(path) != str:
        return False
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
        return True
    else:
        return False


# --Random State--
def set_random_state(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True