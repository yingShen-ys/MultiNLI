from .ga import NaiveSentenceEncoder, DiagonalGaussianEncoder, CategoricalEncoder, long2onehot

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.nn.init import xavier_uniform_, orthogonal_
except:
    from torch.nn.init import xavier_normal as xavier_uniform_
    from torch.nn.init import orthogonal as orthogonal_
# import pyro
import torch.distributions as distributions
from torch.optim import Adam

SEED = 233
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)


class GAIA(nn.Module):
    """
    Implements the GAI manually.
    """
    def __init__(self, x_dim, y_dim, g_dim,
                 zy_dim, zg_dim, rnn_type, vocab_size,
                 embedding_size, hidden_size,
                 label_loss_multiplier=None,
                 genre_loss_multiplier=None):
        super(GAIA, self).__init__()
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
                                                self.embedding_size, self.hidden_size*2,
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

    def encode(self, xp, xh, ys, gs=None):
        xs = self.sentence_encoder(xp, xh)
        ys = long2onehot(ys, self.y_dim)
        gs = long2onehot(gs, self.g_dim)
        y_loc = self.encoder_y(xs) # (batch_sz, num_classes)
        g_loc = self.encoder_g(xs)

        zy_loc, zy_scale = self.encoder_zy(xs, ys)
        # enumeration of discrete variables, get size (num_classes, batch_sz, num_classes)
        if gs is None:
            g_enums = torch.stack([g_loc.new_zeros(g_loc.size()) for _ in range(g_loc.size()[-1])], dim=0)
            for j in range(g_loc.size()[-1]):
                g_enums[j][:, j] = 1
            zg_loc, zg_scale = self.encoder_zg(xs, g_enums) # (num_classes, batch_sz, hidden_dim)

            # compute expectation
            zg_loc = torch.bmm(zg_loc.permute(1, 2, 0), g_loc.unsqueeze(-1)).squeeze() # (b, h, c) X (b, c, 1) -> (b, h)
            zg_scale = torch.bmm(zg_scale.permute(1, 2, 0), g_loc.unsqueeze(-1)).squeeze()
        else:
            zg_loc, zg_scale = self.encoder_zg(xs, gs)
        return y_loc, g_loc, zg_loc, zy_loc, zg_scale, zy_scale, xs

    def decode(self, zy, zg):
        # from zg, zy to x, y and g
        x_loc, x_scale = self.decoder_x(zy, zg)
        y_dist = self.decoder_y(zy)
        g_dist = self.decoder_g(zg)
        return x_loc, x_scale, y_dist, g_dist

    def classify(self, xp, xh):
        xs = self.sentence_encoder(xp, xh)
        y = self.encoder_y(xs)
        g = self.encoder_g(xs)
        return y, g

    def forward(self, xp, xh, ys, gs=None):
        # encode
        y_loc, g_loc, zg_loc, zy_loc, zg_scale, zy_scale, xs = self.encode(xp, xh, ys, gs)

        # sampling with reparametrization
        zg = distributions.Normal(zg_loc, zg_scale).rsample() # (b, h)
        zy = distributions.Normal(zy_loc, zy_scale).rsample() # (b, h')

        # decode
        x_loc, x_scale, y_dist, g_dist = self.decode(zy, zg)
        return y_loc, g_loc, zg_loc, zy_loc, zg_scale, zy_scale, x_loc, x_scale, y_dist, g_dist, xs

def ELBO(info, true_y, true_g=None, llm=0.0, glm=0.0,
         generative=True, discriminative=True, variational=True):
    """
    Not just ELBO, but also auxiliary posterior loss
    """
    if not (generative or discriminative or variational):
        raise ValueError("All types of losses are turned off!")
    y_loc, g_loc, zg_loc, zy_loc, zg_scale, zy_scale, x_loc, x_scale, y_dist, g_dist, xs = info # unpack
    losses = []
    # generative losses
    # score the xs w.r.t to the distribution
    if generative:
        nll_x = -torch.mean(torch.sum(distributions.Normal(x_loc, x_scale).log_prob(xs), dim=-1), dim=-1)
        losses.append(nll_x)
        nll_labels = F.cross_entropy(y_dist, true_y)
        if true_g is not None:
            nll_labels = nll_labels + F.cross_entropy(g_dist, true_g)
        losses.append(nll_labels)

    if variational:
        # variational losses
        # compute KL w.r.t standard Gaussian
        kl_y = torch.mean(-0.5 * torch.sum(1 + torch.log(zy_scale**2) - zy_loc**2 - zy_scale**2, dim=-1), dim=-1)
        kl_g = torch.mean(-0.5 * torch.sum(1 + torch.log(zg_scale**2) - zg_loc**2 - zg_scale**2, dim=-1), dim=-1)
        losses.append(kl_y)
        losses.append(kl_g)

    if discriminative:
        # maximize the posterior of labels
        nl_posterior_labels = F.cross_entropy(y_loc, true_y) * llm
        if true_g is not None:
            nl_posterior_labels = nl_posterior_labels + F.cross_entropy(g_loc, true_g) * glm
        losses.append(nl_posterior_labels)

    # minimize negative variational lower bound
    return sum(losses)


class GAIAx(nn.Module):
    """
    A fake network which is actually just an LSTM
    """
    def __init__(self, x_dim, y_dim, g_dim,
                 zy_dim, zg_dim, rnn_type, vocab_size,
                 embedding_size, hidden_size,
                 label_loss_multiplier,
                 genre_loss_multiplier):
        super(GAIAx, self).__init__()
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
                                                self.embedding_size, self.hidden_size*2,
                                                self.x_dim, rnn_type=self.rnn_type)

        # # define the probabilistic decoders (generative distributions)
        # self.decoder_x = DiagonalGaussianEncoder([self.zy_dim, self.zg_dim],
        #                                          [self.hidden_size],
        #                                          self.x_dim)
        # self.decoder_y = CategoricalEncoder(self.zy_dim,
        #                                     [self.hidden_size],
        #                                     self.y_dim)
        # self.decoder_g = CategoricalEncoder(self.zg_dim,
        #                                     [self.hidden_size],
        #                                     self.g_dim)
        # define the probabilistic encoders (variational distributions in model)
        # self.encoder_zg = DiagonalGaussianEncoder([self.x_dim, self.g_dim],
        #                                           [self.hidden_size],
        #                                           self.zg_dim)
        # self.encoder_zy = DiagonalGaussianEncoder([self.x_dim, self.y_dim],
        #                                           [self.hidden_size],
        #                                           self.zy_dim)
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

    # def encode(self, xp, xh, ys, gs=None):
    #     xs = self.sentence_encoder(xp, xh)
    #     ys = long2onehot(ys, self.y_dim)
    #     gs = long2onehot(gs, self.g_dim)
    #     y_loc = self.encoder_y(xs) # (batch_sz, num_classes)
    #     g_loc = self.encoder_g(xs)

    #     zy_loc, zy_scale = self.encoder_zy(xs, ys)
    #     # enumeration of discrete variables, get size (num_classes, batch_sz, num_classes)
    #     if gs is None:
    #         g_enums = torch.stack([g_loc.new_zeros(g_loc.size()) for _ in range(g_loc.size()[-1])], dim=0)
    #         for j in range(g_loc.size()[-1]):
    #             g_enums[j][:, j] = 1
    #         zg_loc, zg_scale = self.encoder_zg(xs, g_enums) # (num_classes, batch_sz, hidden_dim)

    #         # compute expectation
    #         # print(zg_loc.permute(1, 2, 0).size())
    #         # print(g_loc.size())
    #         # input("inspect enumeration sizes")
    #         zg_loc = torch.bmm(zg_loc.permute(1, 2, 0), g_loc.unsqueeze(-1)).squeeze() # (b, h, c) X (b, c, 1) -> (b, h)
    #         zg_scale = torch.bmm(zg_scale.permute(1, 2, 0), g_loc.unsqueeze(-1)).squeeze()
    #     else:
    #         zg_loc, zg_scale = self.encoder_zg(xs, gs)
    #     return y_loc, g_loc, zg_loc, zy_loc, zg_scale, zy_scale, xs

    # def decode(self, zy, zg):
    #     # from zg, zy to x, y and g
    #     x_loc, x_scale = self.decoder_x(zy, zg)
    #     y_dist = self.decoder_y(zy)
    #     g_dist = self.decoder_g(zg)
    #     return x_loc, x_scale, y_dist, g_dist

    def classify(self, xp, xh):
        # print(xp.size())
        # print(xh.size())
        # input("inspect input")
        xs = self.sentence_encoder(xp, xh)
        y = self.encoder_y(xs)
        g = self.encoder_g(xs)
        return y, g

    def forward(self, xp, xh, ys, gs=None):
        return self.classify(xp, xh)
        # # encode
        # y_loc, g_loc, zg_loc, zy_loc, zg_scale, zy_scale, xs = self.encode(xp, xh, ys, gs)

        # # sampling with reparametrization
        # zg = distributions.Normal(zg_loc, zg_scale).rsample() # (b, h)
        # zy = distributions.Normal(zy_loc, zy_scale).rsample() # (b, h')

        # # decode
        # x_loc, x_scale, y_dist, g_dist = self.decode(zy, zg)
        # return y_loc, g_loc, zg_loc, zy_loc, zg_scale, zy_scale, x_loc, x_scale, y_dist, g_dist, xs


class GAIAxlarge(nn.Module):
    """
    Even simpler LSTM with same API
    """
    def __init__(self, x_dim, y_dim, g_dim,
                 zy_dim, zg_dim, rnn_type, vocab_size,
                 embedding_size, hidden_size,
                 label_loss_multiplier,
                 genre_loss_multiplier, pretrained=None):
        super(GAIAxlarge, self).__init__()
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
        if pretrained is not None:
            self.embeddings = pretrained.embeddings
            self.sentence_encoder = pretrained.ssencoder
            self.encoder_y = nn.Sequential(pretrained.mlps, pretrained.classifer)
        # init the components of the model
        self.initialize_model()

    def initialize_model(self):

        # # initialize a feature-extraction network
        # self.sentence_encoder = NaiveSentenceEncoder(self.vocab_size,
        #                                         self.embedding_size, self.hidden_size*2,
        #                                         self.x_dim, rnn_type=self.rnn_type)
    
        # # define the probabilistic decoders (generative distributions)
        # self.decoder_x = DiagonalGaussianEncoder([self.zy_dim, self.zg_dim],
        #                                          [self.hidden_size],
        #                                          self.x_dim)
        # self.decoder_y = CategoricalEncoder(self.zy_dim,
        #                                     [self.hidden_size],
        #                                     self.y_dim)
        # self.decoder_g = CategoricalEncoder(self.zg_dim,
        #                                     [self.hidden_size],
        #                                     self.g_dim)
        # define the probabilistic encoders (variational distributions in model)
        # self.encoder_zg = DiagonalGaussianEncoder([self.x_dim, self.g_dim],
        #                                           [self.hidden_size],
        #                                           self.zg_dim)
        # self.encoder_zy = DiagonalGaussianEncoder([self.x_dim, self.y_dim],
        #                                           [self.hidden_size],
        #                                           self.zy_dim)
        # self.encoder_g = CategoricalEncoder(self.x_dim,
        #                                     [self.hidden_size],
        #                                     self.g_dim)
        # self.encoder_y = CategoricalEncoder(self.x_dim,
        #                                     [self.hidden_size],
        #                                     self.y_dim)
        pass

    def init_weight(self, pretrained_embedding):
        """
        Initialize the weight for the embedding layer using pretrained_embedding
        :param pretrained_embedding:
        :return:
        """
        # self.sentence_encoder.init_weight(pretrained_embedding)
        pass

    # def encode(self, xp, xh, ys, gs=None):
    #     xs = self.sentence_encoder(xp, xh)
    #     ys = long2onehot(ys, self.y_dim)
    #     gs = long2onehot(gs, self.g_dim)
    #     y_loc = self.encoder_y(xs) # (batch_sz, num_classes)
    #     g_loc = self.encoder_g(xs)

    #     zy_loc, zy_scale = self.encoder_zy(xs, ys)
    #     # enumeration of discrete variables, get size (num_classes, batch_sz, num_classes)
    #     if gs is None:
    #         g_enums = torch.stack([g_loc.new_zeros(g_loc.size()) for _ in range(g_loc.size()[-1])], dim=0)
    #         for j in range(g_loc.size()[-1]):
    #             g_enums[j][:, j] = 1
    #         zg_loc, zg_scale = self.encoder_zg(xs, g_enums) # (num_classes, batch_sz, hidden_dim)

    #         # compute expectation
    #         # print(zg_loc.permute(1, 2, 0).size())
    #         # print(g_loc.size())
    #         # input("inspect enumeration sizes")
    #         zg_loc = torch.bmm(zg_loc.permute(1, 2, 0), g_loc.unsqueeze(-1)).squeeze() # (b, h, c) X (b, c, 1) -> (b, h)
    #         zg_scale = torch.bmm(zg_scale.permute(1, 2, 0), g_loc.unsqueeze(-1)).squeeze()
    #     else:
    #         zg_loc, zg_scale = self.encoder_zg(xs, gs)
    #     return y_loc, g_loc, zg_loc, zy_loc, zg_scale, zy_scale, xs

    # def decode(self, zy, zg):
    #     # from zg, zy to x, y and g
    #     x_loc, x_scale = self.decoder_x(zy, zg)
    #     y_dist = self.decoder_y(zy)
    #     g_dist = self.decoder_g(zg)
    #     return x_loc, x_scale, y_dist, g_dist

    def classify(self, xp, xh):
        # print(xp.size())
        # print(xh.size())
        # input("inspect input")
        xs = self.sentence_encoder(xp, xh)
        y = self.encoder_y(xs)
        g = self.encoder_g(xs)
        return y, g

    def forward(self, xp, xh, ys, gs=None):
        ep = self.embeddings(xp)
        eh = self.embeddings(xh)
        hp = self.sentence_encoder(ep)
        hh = self.sentence_encoder(eh)

        xs = torch.cat([hp, hh, torch.abs(hp-hh), torch.mul(hp, hh)], 1)
        y = self.encoder_y(xs)
        return y, y # dummy for g
        # # encode
        # y_loc, g_loc, zg_loc, zy_loc, zg_scale, zy_scale, xs = self.encode(xp, xh, ys, gs)

        # # sampling with reparametrization
        # zg = distributions.Normal(zg_loc, zg_scale).rsample() # (b, h)
        # zy = distributions.Normal(zy_loc, zy_scale).rsample() # (b, h')

        # # decode
        # x_loc, x_scale, y_dist, g_dist = self.decode(zy, zg)
        # return y_loc, g_loc, zg_loc, zy_loc, zg_scale, zy_scale, x_loc, x_scale, y_dist, g_dist, xs

