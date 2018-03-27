import torch
import numpy as np
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.init import xavier_uniform, orthogonal
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
        orthogonal(self.weight_lhh)
        orthogonal(self.weight_rhh)

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
        self.cell = TreeLSTMCell(in_features, hidden_size)

    def forward(self, input, tree):
        '''
        input is all the embedding vectors (hence variables), tree is the list of all token indices
        '''
        stack = []
        is_cuda = next(self.cell.parameters()).is_cuda
        states = [] # a place holder for storing all node states
        right_bracket_token = tree[0, -1] # the last token must be right_bracket

        if is_cuda:
            FloatTensor = torch.cuda.FloatTensor
        else:
            FloatTensor = torch.FloatTensor

        for node, val in zip(tree.permute(1, 0), input.permute(1, 0, 2)):
            if (node != right_bracket_token).all(): # the condition results in a ByteTensor, had to use .all() on it to get bool
                init_left_state = [Variable(torch.zeros(self.hidden_size).type(FloatTensor), requires_grad=False) for _ in range(2)]
                init_right_state = [Variable(torch.zeros(self.hidden_size).type(FloatTensor), requires_grad=False) for _ in range(2)]
                node_state = self.cell(val, init_left_state, init_right_state)
                stack.append(node_state)
            else:
                right_state = stack.pop()
                left_state = stack.pop()
                node_state = self.cell(val, left_state, right_state)
                stack.append(node_state)
            print(len(stack))
            states.append(node_state[0]) # only append hidden states
        states = torch.cat(states).unsqueeze(0) # (1, num_nodes, hidden_size)

        print("\n")
        # print(len(stack))
        assert len(stack) == 1 # if this is wrong then something must be off
        return stack[0][0], states


class ESIMTreeClassifier(nn.Module):
    '''
    ESIM with Tree LSTM

    Args:
        params: a dictionary that contains all the neccessary hyperparameters

    Shapes:
         - Input: 
    '''

    def __init__(self, params):
        super(ESIMTreeClassifier, self).__init__()
        self.embedding = nn.Embedding(params["vocab_size"], embedding_dim=params["embed_dim"])
        self.encoder = TreeLSTM(params["embed_dim"], params["lstm_h"])
        self.compressor = nn.Sequential(nn.Linear(params["lstm_h"] * 4, params['F_h']), nn.ReLU(), nn.Dropout(params['mlp_dr']))
        self.inferer = TreeLSTM(params["F_h"], params["lstm_h"])
        self.classifier = nn.Sequential(nn.Linear(params["lstm_h"] * 6, params['lstm_h']), nn.Tanh(), nn.Dropout(params['mlp_dr']))
        self.softmax_layer = nn.Sequential(nn.Linear(params['lstm_h'], params['num_class']), nn.Softmax())

    def init_weight(self, pretrained_embedding):
        self.embedding.weight = Parameter(pretrained_embedding)


    def forward(self, premise, hypothesis):
        left_bracket_token = premise[0, 0] # the first token must be left_bracket
        premise_mask = premise != left_bracket_token
        hypothesis_mask = hypothesis != left_bracket_token
        print(premise.shape)
        print(hypothesis.shape)

        # mask out the left brackets in the parse
        premise = premise[premise_mask].unsqueeze(0)
        hypothesis = hypothesis[hypothesis_mask].unsqueeze(0)

        premise_embedding = self.embedding(premise)
        hypothesis_embedding = self.embedding(hypothesis)

        _, premise_node_states = self.encoder(premise_embedding, premise)
        _, hypothesis_node_states = self.encoder(hypothesis_embedding, hypothesis)

        # Local Inference Modeling and Enhancement
        print(premise_node_states.shape)
        input()
        corr_matrix = torch.exp(torch.matmul(premise_node_states, torch.transpose(hypothesis_node_states, 1, 2)))
        premise_w = torch.div(corr_matrix, torch.sum(corr_matrix, 2, True))
        hypothesis_w = torch.div(corr_matrix, torch.sum(corr_matrix, 1, True))

        premise_tilde = torch.matmul(premise_w, hypothesis_node_states)
        hypothesis_tilde = torch.matmul(torch.transpose(hypothesis_w, 1, 2), premise_node_states)

        premise_m = torch.cat([premise_node_states,
                               premise_tilde,
                               torch.abs(premise_node_states - premise_tilde),
                               torch.mul(premise_node_states, premise_tilde)], 2)
        hypothesis_m = torch.cat([hypothesis_node_states,
                                  hypothesis_tilde,
                                  torch.abs(hypothesis_node_states - hypothesis_tilde),
                                  torch.mul(hypothesis_node_states, hypothesis_tilde)], 2)

        # Reducing the dimensions
        premise_c = self.compressor(premise_m)
        hypothesis_c = self.compressor(hypothesis_m)

        # Inference composition
        premise_root, premise_node_inference = self.inferer(premise_c, premise)
        hypothesis_root, hypothesis_node_inference = self.inferer(hypothesis_c, hypothesis)

        # Pooling
        premise_max, _ = torch.max(premise_node_inference, 1)
        hypothesis_max, _ = torch.max(hypothesis_node_inference, 1)
        premise_v = torch.cat([torch.mean(premise_node_inference, 1), premise_max], 1)
        hypothesis_v = torch.cat([torch.mean(hypothesis_node_inference, 1), hypothesis_max], 1)

        v = torch.cat([premise_v, hypothesis_v, premise_root, hypothesis_root], 1)

        # final classifier
        prediction = self.classifier(v)
        prediction = self.softmax_layer(prediction)

        return prediction
