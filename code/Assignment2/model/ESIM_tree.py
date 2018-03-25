import torch
import numpy as np
import torch.nn as nn

seed = 233
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)


class ESIMTreeClassifier(nn.Module):
    '''
    ESIM with Tree LSTM

    Args:
    '''

    def __init__(self, params):
        super(ESIMTreeClassifier, self).__init__()
