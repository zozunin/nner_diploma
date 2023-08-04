import numpy as np
import random
import torch

def set_seed(SEED=42):
    
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)