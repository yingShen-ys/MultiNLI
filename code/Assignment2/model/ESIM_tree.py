import torch
import numpy as np
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.init import xavier_uniform
from torch.nn.modules.rnn import RNNCellBase
import torch.nn.functional as F
from torch.autograd import Variable

seed = 233
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)


class TreeLSTMCell(RNNCellBase):
    '''
    A tree LSTM cell that aggregates information from both children
    Theoretically this cell can be used on batched data, but it is
    generally very hard to mask trees and batch them

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
        self.weight_ih = Parameter(torch.Tensor(5 * hidden_size, input_size))
        self.weight_lhh = Parameter(torch.Tensor(5 * hidden_size, hidden_size))
        self.weight_rhh = Parameter(torch.Tensor(5 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(5 * hidden_size))
            self.bias_lhh = Parameter(torch.Tensor(5 * hidden_size))
            self.bias_rhh = Parameter(torch.Tensor(5 * hidden_size))
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
        self.bias_ih.data[self.hidden_size:self.hidden_size * 3] = 1
        self.bias_lhh.data[self.hidden_size:self.hidden_size * 3] = 1
        self.bias_rhh.data[self.hidden_size:self.hidden_size * 3] = 1

    def forward(self, input, left_state, right_state):
        lh, lc = left_state # pylint: disable=C0103
        rh, rc = right_state # pylint: disable=C0103

        gates = (F.linear(input, self.weight_ih, self.bias_ih) + F.linear(lh, self.weight_lhh, self.bias_lhh) + F.linear(rh, self.weight_rhh, self.bias_rhh))
        ingate, leftforgetgate, rightforgetgate, cellgate, outgate = gates.chunk(5, 1)

        # gates: shape (hidden_size,)
        ingate = F.sigmoid(ingate)
        leftforgetgate = F.sigmoid(leftforgetgate)
        rightforgetgate = F.sigmoid(rightforgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        # output
        yc = (leftforgetgate * lc) + (rightforgetgate * rc) + (ingate * cellgate) # pylint: disable=C0103
        yh = outgate * F.tanh(yc) # pylint: disable=C0103
        return yh, yc


class TreeLSTM(nn.Module):
    '''
    A binary tree LSTM that unfortunately does not support batching

    Args:
        in_features: size of each input at each time step
        hidden_size: size of hidden states of LSTM

    Shape:
         - Input: input: a list containing embeddings, transfored
                  from the bracketed form of tree representations
                  
                  tree: a list of idx and <-REDUCE-> tokens, used
                  to guide how the network computes along the tree.
                  
         - Output: final hidden state at root node
    '''

    def __init__(self, in_features, hidden_size):
        super(TreeLSTM, self).__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.stack = [] # the processing of the tree is stack-based
        self.cell = TreeLSTMCell(in_features, hidden_size)

    def forward(self, input, tree):
        is_cuda = next(self.cell.parameters()).is_cuda()
        states = [] # a place holder for storing all node states

        if is_cuda:
            FloatTensor = torch.cuda.FloatTensor
        else:
            FloatTensor = torch.FloatTensor

        for node, val in zip(tree, input):
            if node != "<-REDUCE->":
                init_left_state = [Variable(torch.zeros(self.hidden_size).type(FloatTensor), requires_grad=False) for _ in range(2)]
                init_right_state = [Variable(torch.zeros(self.hidden_size).type(FloatTensor), requires_grad=False) for _ in range(2)]
                node_state = self.cell(val, init_left_state, init_right_state)
                self.stack.append(node_state)
            else:
                right_state = self.stack.pop()
                left_state = self.stack.pop()
                node_state = self.cell(val, left_state, right_state)
                self.stack.append(node_state)
            states.append(node_state)

        assert len(self.stack) == 1 # if this is wrong then something must be off
        return self.stack[0][0], states


class ESIMTreeClassifier(nn.Module):
    '''
    ESIM with Tree LSTM

    Args:
    '''

    def __init__(self, params):
        super(ESIMTreeClassifier, self).__init__()
