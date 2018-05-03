from __future__ import print_function
import sys
import os


sys.path.append("..")
from model import GridLSTM
from utils import NLIDataloader
from utils import evaluate, combine_dataset, load_param


from sklearn.metrics import accuracy_score
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

seed = 233
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)



