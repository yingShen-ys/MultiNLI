import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, config_enumerate
from pyro.optim import Adam

SEED = 233
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

class GenreAgnosticInference(nn.Module):
    """
    Genre Agnostic Inference model for NLI, a structure that is similar to semi-supervised VAE

    Args:
        (zg) --> (g) (partially observed)
          |
          v
         (x)
          ^
          |
        (zy) --> (y)

    Input:
         - premise: sequences of premise sentences, converted to LongTensors
         - hypothesis: the same as premise
    
    Output:
         - rep: a final representation for the sentence
    """
    def __init__(self, x_dim, y_dim, g_dim,
                 zy_dim, zg_dim, vocab_size,
                 embedding_size, hidden_size,
                 label_loss_multiplier,
                 genre_loss_multiplier):
        super(GenreAgnosticInference, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.g_dim = g_dim
        self.zy_dim = zy_dim
        self.zg_dim = zg_dim
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.label_loss_multiplier = label_loss_multiplier
        self.genre_loss_multiplier = genre_loss_multiplier

        # init the components of the model
        self.initialize_model()

    def initialize_model(self):
        # initialize a feature-extraction network
        self.sentence_encoder = SentenceEncoder(self.vocab_size,
                                                self.embedding_size,
                                                self.x_dim)

        # define the probabilistic decoders (generative distributions)
        self.decoder_x = DiagonalGaussianEncoder(self.zy_dim+self.zg_dim,
                                                 [self.hidden_size, self.hidden_size],
                                                 self.x_dim)
        self.decoder_y = CategoricalEncoder(self.zy_dim,
                                            [self.hidden_size, self.hidden_size],
                                            self.y_dim)
        self.decoder_g = CategoricalEncoder(self.zg_dim,
                                            [self.hidden_size, self.hidden_size],
                                            self.g_dim)
        # define the probabilistic encoders (variational distributions in model)
        self.encoder_zg = DiagonalGaussianEncoder(self.x_dim+self.g_dim,
                                                  [self.hidden_size, self.hidden_size],
                                                  self.zg_dim)
        self.encoder_zy = DiagonalGaussianEncoder(self.x_dim+self.y_dim,
                                                  [self.hidden_size, self.hidden_size],
                                                  self.zy_dim)
        self.encoder_g = CategoricalEncoder(self.kz_dim,
                                            [self.hidden_size, self.hidden_size],
                                            self.z_dim)
        self.encoder_y = CategoricalEncoder(self.ky_dim,
                                            [self.hidden_size, self.hidden_size],
                                            self.y_dim)

    def init_weight(self, pretrained_embedding):
        """
        Initialize the weight for the embedding layer using pretrained_embedding
        :param pretrained_embedding:
        :return:
        """
        self.sentence_encoder.init_weight(pretrained_embedding)

    def generative_model(self, xh, xp, ys, gs=None):
        """
        This model have the following generative process:
        p(zg) = normal(0, I)
        p(zy) = normal(0, I)
        p(z|zg) = categorical(theta(zg))
        p(y|zy) = categorical(theta(zy))
        p(x|zg, zy) = normal(mu(zg, zy), sigma(zg, zy))

        It contains the following variational distributions:
        q(g|x) = categorical(theta(x))
        q(y|x) = categorical(theta(x))
        q(zg|x, g) = normal(mu(x, g), sigma(x, g))
        q(zy|x, g) = normal(mu(x, g), sigma(x, g))
        """
        pyro.module("generative", self)
        xs = self.sentence_encoder(xh, xp) # from (batch_sz, seq_len, embedding_sz) to (batch_sz, x_dim)
        batch_size = xs.size(0)
        with pyro.iarange("data"):
            zg_prior_loc = xs.new_zeros([batch_size, self.zg_dim])
            zg_prior_scale = xs.new_ones([batch_size, self.zg_dim])
            zy_prior_loc = xs.new_zeros([batch_size, self.zy_dim])
            zy_prior_scale = xs.new_ones([batch_size, self.zy_dim])
            zg = pyro.sample("zG", dist.Normal(zg_prior_loc, zg_prior_scale).independent(1))
            zy = pyro.sample("zY", dist.Normal(zy_prior_loc, zy_prior_scale).independent(1))

            x_loc, x_scale = self.decoder_x(torch.cat((zy, zg), dim=1))
            theta_g = self.decoder_z(zg)
            theta_y = self.decoder_y(zy)
            
            # observe the data
            pyro.sample("X", dist.Normal(x_loc, x_scale).independence(1), obs=xs)
            pyro.sample("G", dist.OneHotCategorical(theta_g).independence(1), obs=gs)
            pyro.sample("Y", dist.OneHotCategorical(theta_y).independence(1), obs=ys)

    def generative_guide(self, xh, xp, ys, gs=None):
        xs = self.sentence_encoder(xh, xp) # from (batch_sz, seq_len, embedding_sz) to (batch_sz, x_dim)
        with pyro.iarange("data"):
            if gs is None:
                theta_g = self.encoder_g(xs)
                gs = pyro.sample("G", dist.OneHotCategorical(theta_g).independent(1))

            zg_loc, zg_scale = self.encoder_zg(torch.cat((xs, gs), dim=1))
            zy_loc, zy_scale = self.encoder_zy(torch.cat((ys, gs), dim=1))
            
            # observe the data
            pyro.sample("zG", dist.Normal(zg_loc, zg_scale).independent(1))
            pyro.sample("zY", dist.Normal(zy_loc, zy_scale).independent(1))

    def discriminative_model(self, xh, xp, ys, gs=None):
        pyro.module("discriminative", self)
        xs = self.sentence_encoder(xh, xp) # from (batch_sz, seq_len, embedding_sz) to (batch_sz, x_dim)
        with pyro.iarange("data"):
            # only include loss term for genre if gs is provided
            if gs is not None:
                theta_g = self.encoder_g(xs)
                with pyro.poutine.scale(self.genre_loss_multiplier):
                    pyro.sample("G", dist.OneHotCategorical(theta_g).independent(1), obs=gs)

            # loss term is always present for label
            theta_y = self.encoder_y(xs)
            with pyro.poutine.scale(self.label_loss_multiplier):
                pyro.sample("Y", dist.OneHotCategorical(theta_y).independent(1), obs=ys)

    def discriminative_guide(self, xh, xp, ys, gs=None):
        """A dummy guide for discriminative loss"""
        pass

    def forward(self, xh, xp):
        """
        returns the categorical distribution over y and g
        """
        xs = self.sentence_encoder(xh, xp) # from (batch_sz, seq_len, embedding_sz) to (batch_sz, x_dim)
        y_dist = self.encoder_y(xs)
        g_dist = self.encoder_g(xs)
        return y_dist, g_dist


class SentenceEncoder(nn.Module):
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
        super(SentenceEncoder, self).__init__()
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
                    left_hidden_state = torch.zeros(batch_size, self.hidden_size).type(DTYPE)
                lower_hidden_state = hidden_states[i][j-1]
                if lower_hidden_state is None:
                    lower_hidden_state = torch.zeros(batch_size, self.hidden_size).type(DTYPE)
                hidden = torch.cat((left_hidden_state, lower_hidden_state), dim=1)
                left_cell_state = cell_states[i-1][j]
                if left_cell_state is None:
                    left_cell_state = torch.zeros(batch_size, self.hidden_size).type(DTYPE)
                lower_cell_state = cell_states[i][j-1]
                if lower_cell_state is None:
                    lower_cell_state = torch.zeros(batch_size, self.hidden_size).type(DTYPE)
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

class DiagonalGaussianEncoder(nn.Module):
    """
    An MLP that produces the mean and cov for a diagonal Gaussian
    
    Args:
        input_size: int defining the layer sizes
        hidden_size: [int, int], two hidden layers
        output_size: the actual output will be twice as this, one for
                     mean, one for covariance

    Input:
         - samples: samples from the RV this distribution conditions on
    
    Output:
         - mu: mean of the Gaussian
         - sigma: covariance of the Gaussian. Only diagonal values.
    """
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DiagonalGaussianEncoder, self).__init__()
        self.layer_in = nn.Linear(input_size, hidden_sizes[0])
        self.layer_h1 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.layer_out = nn.Linear(hidden_sizes[-1], 2*output_size)

    def forward(self, input):
        h = F.relu(self.layer_in(input))
        h = F.relu(self.layer_h1(h))
        o = self.layer_out(h)
        mu, sigma = torch.chunk(o, 2, dim=1)
        sigma = torch.exp(sigma)
        return mu, sigma

class CategoricalEncoder(nn.Module):
    """
    An MLP that produces the mean and cov for a diagonal Gaussian
    
    Args:
        input_size: int defining the layer sizes
        hidden_size: [int, int], two hidden layers
        output_size: the actual output will be twice as this, one for
                     mean, one for covariance

    Input:
         - samples: samples from the RV this distribution conditions on
    
    Output:
         - theta: the vector representing a categorical distribution
    """
    def __init__(self, input_size, hidden_sizes, output_size):
        super(CategoricalEncoder, self).__init__()
        self.layer_in = nn.Linear(input_size, hidden_sizes[0])
        self.layer_h1 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.layer_out = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, input):
        h = F.relu(self.layer_in(input))
        h = F.relu(self.layer_h1(h))
        theta = F.softmax(self.layer_out(h), dim=1)
        return theta

def test():
    model = SentenceEncoder(30, 15, 15)
    premise = torch.randint(0, 29, (16, 8)).type(torch.LongTensor)
    hypothesis = torch.randint(0, 29, (16, 12)).type(torch.LongTensor)
    y = model(premise, hypothesis)
    print(y)

if __name__ == "__main__":
    test()

