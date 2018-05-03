import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable as V

SEED = 233
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)


class GridLSTM(nn.Module):
    """
    The Grid LSTM model for Natural Language Inference.

    Args:
        vocab_size: the size of the total vocabulary.
        embedding_size: the dimensions of the input word embeddings.
        hidden_size: the dimensions of the hidden representation.

    Input:
         - premise: sequences of premise sentences, converted to LongTensors
         - hypothesis: the same as premise
    
    Output:
         - rep: a final representation for the sentence
    """

    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(GridLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.cell = nn.LSTMCell(embedding_size * 2, hidden_size * 2)
        self.condenser_h = nn.Linear(hidden_size * 2, hidden_size) # the layer that shrinks the hidden states
        self.condenser_c = nn.Linear(hidden_size * 2, hidden_size) # the layer that shrinks the cell states
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.linear_3 = nn.Linear(hidden_size, 3)
        self.softmax = nn.Softmax(dim=1)

    def init_weight(self, pretrained_embedding):
        """
        Initialize the weight for the embedding layer using pretrained_embedding
        :param pretrained_embedding:
        :return:
        """
        self.embedding.weight = nn.Parameter(pretrained_embedding)

    def forward(self, premise, hypothesis):
        premise_emb = self.embedding(premise) # (batch_size, max_len, embedding_size)
        hypothesis_emb = self.embedding(hypothesis) # (batch_size, max_len, embedding_size)
        DTYPE = eval(hypothesis_emb.data.type())
        pseq_len = premise_emb.shape[1]
        hseq_len = hypothesis_emb.shape[1]
        batch_size = premise_emb.shape[0]
        hidden_states = [[None] * (hseq_len+1)] * (pseq_len+1)
        cell_states = [[None] * (hseq_len+1)] * (pseq_len+1)
        for i in range(1, pseq_len+1):
            for j in range(1, hseq_len+1):
                # print("Now pulling out the states from: ({}, {}) out of a {} * {} table".format(i-1, j, pseq_len+1, hseq_len+1))
                left_hidden_state = hidden_states[i-1][j]
                if left_hidden_state is None:
                    left_hidden_state = V(torch.zeros(batch_size, self.hidden_size).type(DTYPE))
                lower_hidden_state = hidden_states[i][j-1]
                if lower_hidden_state is None:
                    lower_hidden_state = V(torch.zeros(batch_size, self.hidden_size).type(DTYPE))
                hidden = torch.cat((left_hidden_state, lower_hidden_state), dim=1)
                left_cell_state = cell_states[i-1][j]
                if left_cell_state is None:
                    left_cell_state = V(torch.zeros(batch_size, self.hidden_size).type(DTYPE))
                lower_cell_state = cell_states[i][j-1]
                if lower_cell_state is None:
                    lower_cell_state = V(torch.zeros(batch_size, self.hidden_size).type(DTYPE))
                cell = torch.cat((left_cell_state, lower_cell_state), dim=1)
                premise_input = premise_emb[:, i-1, :]
                hypothesis_input = hypothesis_emb[:, j-1, :]
                input = torch.cat((premise_input, hypothesis_input), dim=1)
                out_hidden, out_cell = self.cell(input, (hidden, cell)) # (batch_size, 2 * hidden_size)
                hidden_states[i][j] = self.condenser_h(out_hidden)
                cell_states[i][j] = self.condenser_c(out_cell)
        # return (hidden_states[-1][-1], cell_states[-1][-1]), (hidden_states, cell_states)
        h = hidden_states[-1][-1]
        h = self.relu1(self.linear_1(h))
        h = self.relu2(self.linear_2(h))
        y = self.softmax(self.linear_3(h))
        return y

def test():
    model = GridLSTM(30, 15, 15)
    premise = torch.randint(0, 29, (16, 8)).type(torch.LongTensor)
    hypothesis = torch.randint(0, 29, (16, 12)).type(torch.LongTensor)
    y = model(premise, hypothesis)
    print(y)

if __name__ == "__main__":
    test()

