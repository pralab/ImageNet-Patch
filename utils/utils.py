import torch
import numpy as np
import random

def set_all_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

_imagenette_classes = [0, 217, 482, 491, 497]
target_transforms = lambda y: _imagenette_classes[y]