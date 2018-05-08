import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.nn.init import xavier_uniform_, orthogonal_
except:
    from torch.nn.init import xavier_normal as xavier_uniform_
    from torch.nn.init import orthogonal as orthogonal_
try:
    import pyro
    import pyro.distributions as dist
    from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, config_enumerate
    from pyro.optim import Adam
except:
    print("Cannot import Pyro! Probably using older version of PyTorch.")

SEED = 233
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

def long2onehot(long_tensors, num_classes):
    """
    Convert a torch.LongTensor of shape (batch_sz,) to (batch_sz, num_classes)
    """
    if long_tensors is None:
        return None
    long_tensors = long_tensors.view(-1, 1)
    onehot_tensors = torch.zeros([long_tensors.size()[0], num_classes])
    if long_tensors.is_cuda:
        onehot_tensors = onehot_tensors.cuda()
    onehot_tensors.scatter_(1, long_tensors, 1)
    return onehot_tensors

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
                 zy_dim, zg_dim, rnn_type, vocab_size,
                 embedding_size, hidden_size,
                 label_loss_multiplier,
                 genre_loss_multiplier):
        super(GenreAgnosticInference, self).__init__()
        raise NotImplementedError("Softmax removed but not added later")
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.g_dim = g_dim
        self.zy_dim = zy_dim
        self.zg_dim = zg_dim
        self.rnn_type = rnn_type
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.label_loss_multiplier = label_loss_multiplier
        self.genre_loss_multiplier = genre_loss_multiplier

        # init the components of the model
        self.initialize_model()

    def initialize_model(self):
        # initialize a feature-extraction network
        self.sentence_encoder = NaiveSentenceEncoder(self.vocab_size,
                                                self.embedding_size, self.hidden_size * 2,
                                                self.x_dim, rnn_type=self.rnn_type)

        # define the probabilistic decoders (generative distributions)
        self.decoder_x = DiagonalGaussianEncoder([self.zy_dim, self.zg_dim],
                                                 [self.hidden_size],
                                                 self.x_dim)
        self.decoder_y = CategoricalEncoder(self.zy_dim,
                                            [self.hidden_size],
                                            self.y_dim)
        self.decoder_g = CategoricalEncoder(self.zg_dim,
                                            [self.hidden_size],
                                            self.g_dim)
        # define the probabilistic encoders (variational distributions in model)
        self.encoder_zg = DiagonalGaussianEncoder([self.x_dim, self.g_dim],
                                                  [self.hidden_size],
                                                  self.zg_dim)
        self.encoder_zy = DiagonalGaussianEncoder([self.x_dim, self.y_dim],
                                                  [self.hidden_size],
                                                  self.zy_dim)
        self.encoder_g = CategoricalEncoder(self.x_dim,
                                            [self.hidden_size],
                                            self.g_dim)
        self.encoder_y = CategoricalEncoder(self.x_dim,
                                            [self.hidden_size],
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
        ys = long2onehot(ys, self.y_dim)
        gs = long2onehot(gs, self.g_dim)
        with pyro.iarange("data"):
            zg_prior_loc = xs.new_zeros([batch_size, self.zg_dim])
            zg_prior_scale = xs.new_ones([batch_size, self.zg_dim])
            zy_prior_loc = xs.new_zeros([batch_size, self.zy_dim])
            zy_prior_scale = xs.new_ones([batch_size, self.zy_dim])
            zg = pyro.sample("zG", dist.Normal(zg_prior_loc, zg_prior_scale).independent(1))
            zy = pyro.sample("zY", dist.Normal(zy_prior_loc, zy_prior_scale).independent(1))

            x_loc, x_scale = self.decoder_x(zy, zg)
            theta_g = self.decoder_g(zg)
            theta_y = self.decoder_y(zy)

            # observe the data
            pyro.sample("X", dist.Normal(x_loc, x_scale).independent(1), obs=xs)
            pyro.sample("G", dist.OneHotCategorical(theta_g), obs=gs)
            pyro.sample("Y", dist.OneHotCategorical(theta_y), obs=ys)

    def generative_guide(self, xh, xp, ys, gs=None):
        xs = self.sentence_encoder(xh, xp) # from (batch_sz, seq_len, embedding_sz) to (batch_sz, x_dim)
        ys = long2onehot(ys, self.y_dim)
        gs = long2onehot(gs, self.g_dim)
        with pyro.iarange("data"):
            if gs is None:
                theta_g = self.encoder_g(xs)
                gs = pyro.sample("G", dist.OneHotCategorical(theta_g))

            # print(xs.size())
            # print(gs.size())
            # print(torch.cat((xs, gs), dim=1).size())
            # input("Please press any key to continue!")
            zg_loc, zg_scale = self.encoder_zg(xs, gs)
            zy_loc, zy_scale = self.encoder_zy(xs, ys)
            
            # observe the data
            pyro.sample("zG", dist.Normal(zg_loc, zg_scale).independent(1))
            pyro.sample("zY", dist.Normal(zy_loc, zy_scale).independent(1))

    def discriminative_model(self, xh, xp, ys, gs=None):
        pyro.module("discriminative", self)
        xs = self.sentence_encoder(xh, xp) # from (batch_sz, seq_len, embedding_sz) to (batch_sz, x_dim)
        ys = long2onehot(ys, self.y_dim)
        gs = long2onehot(gs, self.g_dim)        
        with pyro.iarange("data"):
            # only include loss term for genre if gs is provided
            if gs is not None:
                theta_g = self.encoder_g(xs)
                with pyro.poutine.scale(scale=self.genre_loss_multiplier):
                    pyro.sample("G", dist.OneHotCategorical(theta_g), obs=gs)

            # loss term is always present for label
            theta_y = self.encoder_y(xs)
            with pyro.poutine.scale(scale=self.label_loss_multiplier):
                pyro.sample("Y", dist.OneHotCategorical(theta_y), obs=ys)

    def discriminative_guide(self, xh, xp, ys, gs=None):
        """A dummy guide for discriminative loss"""
        # ys = long2onehot(ys, self.y_dim)
        # gs = long2onehot(gs, self.g_dim)
        pass

    def forward(self, xh, xp):
        """
        returns the categorical distribution over y and g
        """
        xs = self.sentence_encoder(xh, xp) # from (batch_sz, seq_len, embedding_sz) to (batch_sz, x_dim)
        y_dist = self.encoder_y(xs)
        g_dist = self.encoder_g(xs)
        return y_dist, g_dist


class NaiveSentenceEncoder(nn.Module):
    """
    An bi-directional LSTM to encoder sentence pairs
    """
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size,
                 rnn_type='lstm', num_layers=2, dropout=0.15):
        super(NaiveSentenceEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        rnn_model = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}[rnn_type]
        self.rnn = rnn_model(input_size=embedding_size, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=dropout, bidirectional=True,
                            batch_first=True)
        self.fc_layer = nn.Linear(hidden_size*num_layers*2*2*3, output_size)
        self.batchnorm_in = nn.BatchNorm1d(hidden_size*num_layers*2*2*3)
        self.batchnorm_out = nn.BatchNorm1d(output_size)
        self.reset_parameters()

    def reset_parameters(self):
        xavier_uniform_(self.fc_layer.weight.data)
        self.fc_layer.bias.data.normal_(0, 0.001)

        # what pytorch should've done themselves
        if self.rnn_type == "lstm":
            for l in range(self.num_layers):
                xavier_uniform_(getattr(self.rnn, "weight_ih_l{}".format(l)).data)
                orthogonal_(getattr(self.rnn, "weight_hh_l{}".format(l)).data)
                xavier_uniform_(getattr(self.rnn, "weight_ih_l{}{}".format(l, '_reverse')).data)
                orthogonal_(getattr(self.rnn, "weight_hh_l{}{}".format(l, '_reverse')).data)
                getattr(self.rnn, "bias_ih_l{}".format(l)).data.fill_(0)
                getattr(self.rnn, "bias_ih_l{}".format(l)).data[self.hidden_size: 2*self.hidden_size] = 1./2
                getattr(self.rnn, "bias_ih_l{}{}".format(l, '_reverse')).data.fill_(0)
                getattr(self.rnn, "bias_ih_l{}{}".format(l, '_reverse')).data[self.hidden_size: 2*self.hidden_size] = 1./2
                getattr(self.rnn, "bias_hh_l{}".format(l)).data.fill_(0)
                getattr(self.rnn, "bias_hh_l{}".format(l)).data[self.hidden_size: 2*self.hidden_size] = 1./2
                getattr(self.rnn, "bias_hh_l{}{}".format(l, '_reverse')).data.fill_(0)
                getattr(self.rnn, "bias_hh_l{}{}".format(l, '_reverse')).data[self.hidden_size: 2*self.hidden_size] = 1./2

    def init_weight(self, pretrained_embedding):
        """
        Initialize the weight for the embedding layer using pretrained_embedding
        :param pretrained_embedding:
        :return:
        """
        self.embedding.weight = nn.Parameter(pretrained_embedding)

    def forward(self, premise, hypothesis):
        batch_sz = premise.size()[0]
        premise_emb = self.embedding(premise)
        hypothesis_emb = self.embedding(hypothesis)

        _, premise_rep = self.rnn(premise_emb) # (batch, num_layers * num_directions, hidden)
        _, hypothesis_rep = self.rnn(hypothesis_emb)

        if isinstance(premise_rep, (tuple, list)): # allows different use of GRU and LSTM
            premise_rep = premise_rep[0]
            hypothesis_rep = hypothesis_rep[0]

        premise_rep = premise_rep.view(batch_sz, -1) # (batch, *)
        hypothesis_rep = hypothesis_rep.view(batch_sz, -1)

        total_rep = torch.cat((premise_rep, hypothesis_rep, torch.abs(premise_rep, hypothesis_rep, torch.mul(premise_rep, hypothesis_rep))), dim=1)
        total_rep = self.batchnorm_in(total_rep)
        final_rep = F.relu(self.fc_layer(total_rep)) #
        return self.batchnorm_out(final_rep)


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
        if isinstance(input_size, (list, tuple)):
            self.layer_in = Parallel([nn.Linear(in_sz, hidden_sizes[0]) for in_sz in input_size])
        else:
            self.layer_in = nn.Linear(input_size, hidden_sizes[0])
            xavier_uniform_(self.layer_in.weight.data)
            self.layer_in.bias.data.normal_(0, 0.001)

        hidden_layers = []
        for t in range(len(hidden_sizes)-1):
            hidden_layers += [nn.Linear(hidden_sizes[t], hidden_sizes[t+1]), nn.BatchNorm1d(hidden_sizes[t+1]), nn.Softplus()]
        self.layer_hidden = nn.Sequential(*hidden_layers)
        self.layer_out = nn.Linear(hidden_sizes[-1], 2*output_size)
        self.reset_parameters()

    def reset_parameters(self):
        for mod in self.layer_hidden:
            if isinstance(mod, nn.Linear):
                xavier_uniform_(mod.weight.data)
                mod.bias.data.normal_(0, 0.001)
        self.layer_out.bias.data.normal_(0, 0.001)
        xavier_uniform_(self.layer_out.weight.data)

    def forward(self, *input):
        h = F.softplus(self.layer_in(*input))
        h = F.softplus(self.layer_hidden(h))
        o = self.layer_out(h)
        mu, sigma = torch.chunk(o, 2, dim=-1)
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
        if isinstance(input_size, (list, tuple)):
            self.layer_in = Parallel([nn.Linear(in_sz, hidden_sizes[0]) for in_sz in input_size])
        else:
            self.layer_in = nn.Linear(input_size, hidden_sizes[0])
            xavier_uniform_(self.layer_in.weight.data)
            self.layer_in.bias.data.normal_(0, 0.001)
        
        hidden_layers = []
        for t in range(len(hidden_sizes)-1):
            hidden_layers += [nn.Linear(hidden_sizes[t], hidden_sizes[t+1]), nn.BatchNorm1d(hidden_sizes[t+1]), nn.Softplus()]
        self.layer_hidden = nn.Sequential(*hidden_layers)
        self.layer_out = nn.Linear(hidden_sizes[-1], output_size)
        self.reset_parameters()

    def reset_parameters(self):
        for mod in self.layer_hidden:
            if isinstance(mod, nn.Linear):
                xavier_uniform_(mod.weight.data)
                mod.bias.data.normal_(0, 0.001)
        xavier_uniform_(self.layer_out.weight.data)
        self.layer_out.bias.data.normal_(0, 0.001)

    def forward(self, *input):
        h = F.softplus(self.layer_in(*input))
        h = F.softplus(self.layer_hidden(h))
        theta = self.layer_out(h)
        return theta

class Parallel(nn.Module):
    """
    An object that takes in a bunch of layers and concat them in parallel
    """
    def __init__(self, modules):
        super(Parallel, self).__init__()
        self.parallel_modules = nn.ModuleList(modules)
        self.len = len(modules)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.parallel_modules:
            xavier_uniform_(module.weight.data)
            module.bias.data.normal_(0, 0.001)

    def forward(self, *inputs):
        assert len(inputs) == self.len, "inputs ({}) not equal to number of modules ({})".format(len(inputs), self.len)
        # for input in inputs:
        #     print(input.size())
        # for module in self.parallel_modules:
        #     print(module)
        outputs = [module(input) for module, input in zip(self.parallel_modules, inputs)]
        return sum(outputs)

def test():
    model = SentenceEncoder(30, 15, 15)
    premise = torch.randint(0, 29, (16, 8)).type(torch.LongTensor)
    hypothesis = torch.randint(0, 29, (16, 12)).type(torch.LongTensor)
    y = model(premise, hypothesis)
    print(y)

if __name__ == "__main__":
    test()

