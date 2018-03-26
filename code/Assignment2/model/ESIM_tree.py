import torch
import numpy as np
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.init import xavier_uniform
import torch.nn.functional as F

seed = 233
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)


class TreeLSTMCell(nn.RNNBase):
    '''
    A tree LSTM cell that aggregates information from both children

    Args:
        input_size: size of input at each node
        hidden_size: size of hidden state at each node
        bias: whether the LSTM cell uses biases

    Shape:
         - Input: input at each node (input_size,), hidden states (hidden_size,) * 2, cell states (hidden_size) * 2
         - Output: hidden state (hidden_size,), cell state (hidden_size,)
    '''

    def __init__(self, input_size, hidden_size, bias=True):
        super(TreeLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_lhh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.weight_rhh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(4 * hidden_size))
            self.bias_lhh = Parameter(torch.Tensor(4 * hidden_size))
            self.bias_rhh = Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_lhh', None)
            self.register_parameter('bias_rhh', None)
        self.reset_parameters()

    def reset_parameters(self):
        '''
        Apply the classic initialization of LSTM weights and bias
        '''
        xavier_uniform(self.weight_ih)
        xavier_uniform(self.weight_lhh)
        xavier_uniform(self.weight_rhh)

        # biases should be zeros except for forget gates'
        self.bias_ih.data.fill_(0)
        self.bias_lhh.data.fill_(0)
        self.bias_rhh.data.fill_(0)
        self.bias_ih.data[self.hidden_size:self.hidden_size * 2] = 1
        self.bias_lhh.data[self.hidden_size:self.hidden_size * 2] = 1
        self.bias_rhh.data[self.hidden_size:self.hidden_size * 2] = 1

    def forward(self, input, left_state, right_state):
        lh, lc = left_state
        rh, rc = right_state
        
        gates = F.linear(input, self.weight_ih, self.bias_ih) + F.linear(lh, self.weight_lhh, self.bias_lhh) + F.linear(rh, self.weight_rhh, self.bias_rhh)
        

    


class TreeLSTM(nn.Module):
    '''
    A tree LSTM that unfortunately does not support batching

    Args:
        in_features: size of each input at each time step
        hidden_size: size of hidden states of LSTM

    Shape:
         - Input: a list containing embeddings and REDUCE tokens
         - Output: final hidden state at root node
    '''

    def __init__(self, in_features, hidden_size):
        super(TreeLSTM, self).__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size



class ESIMTreeClassifier(nn.Module):
    '''
    ESIM with Tree LSTM

    Args:
    '''

    def __init__(self, params):
        super(ESIMTreeClassifier, self).__init__()
