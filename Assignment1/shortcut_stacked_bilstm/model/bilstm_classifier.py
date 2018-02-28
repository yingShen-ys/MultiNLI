from __future__ import print_function
import logging
# logging.basicConfig(level=logging.DEBUG)
import torch
import numpy as np
import torch.nn as nn

seed = 233
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)


class BiLstmClassifier(nn.Module):

    def __init__(self, params):
        super(BiLstmClassifier, self).__init__()

        self.embeddings = nn.Embedding(params["vocab_size"], embedding_dim=params["embed_dim"])
        self.bilstm = nn.LSTM(input_size=params["embed_dim"], hidden_size=params["lstm_h"], batch_first=True, bidirectional=True)

        self.num_mlp_layer = len(params['mlp_h'])
        mlps = []
        for i in range(self.num_mlp_layer):
            if i != 0:
                input_size = params['mlp_h'][i - 1]
            else:
                input_size = params['lstm_h'] * 8  # vp:vh:vp-vh:vp*vh
            mlp = nn.Sequential(nn.Linear(input_size, params['mlp_h'][i]),
                                nn.ReLU(),
                                nn.Dropout(params['mlp_dr']))
            mlps.append(mlp)
        self.mlps = nn.Sequential(*mlps)

        self.classifer = nn.Linear(params['mlp_h'][-1], params["num_class"])

    def init_weight(self, pretrained_embedding):
        self.embeddings.weight = nn.Parameter(pretrained_embedding)

    def forward(self, premise, hypothesis):

        logging.debug("{}Embedding Layer{}".format("-" * 10, "-" * 10))
        premise_embed = self.embeddings(premise)
        hypothesis_embed = self.embeddings(hypothesis)
        logging.debug("premise_embed: {}".format(premise_embed.size()))
        logging.debug("hypothesis_embed: {}".format(hypothesis_embed.size()))
        logging.debug("{}Embedding Layer{}".format("-" * 10, "-" * 10))

        logging.debug("{}Encoder stage{}".format("-" * 10, "-" * 10))
        premise_out, _ = self.bilstm(premise_embed)
        premise_hidden, _ = torch.max(premise_out, 1)  # reduce seq_len dim
        hypothesis_out, _ = self.bilstm(hypothesis_embed)
        hypothesis_hidden, _ = torch.max(hypothesis_out, 1)  # reduce seq_len dim

        logging.debug("After BiLSTM:")
        logging.debug("premise_hidden: {}".format(premise_hidden.size()))
        logging.debug("hypothesis_hidden: {}".format(hypothesis_hidden.size()))

        m = torch.cat([premise_hidden,
                       hypothesis_hidden,
                       torch.abs(premise_hidden-hypothesis_hidden),
                       torch.mul(premise_hidden, hypothesis_hidden)], 1)

        logging.debug("After concat: {}".format(m.size()))
        logging.debug("{}Encoder stage{}".format("-" * 10, "-" * 10))

        logging.debug("{}Classifier stage{}".format("-" * 10, "-" * 10))
        linear = self.mlps(m)
        prediction = self.classifer(linear)
        logging.debug("{}Classifier stage{}".format("-" * 10, "-" * 10))

        return prediction
