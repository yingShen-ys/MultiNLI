from __future__ import print_function
import torch
import numpy as np
import torch.nn as nn

seed = 233
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

class SSEncoder(nn.Module):

    def __init__(self, input_size, hidden_sizes):
        super(SSEncoder, self).__init__()

        self.num_lstm_layers = len(hidden_sizes)
        self.lstms = nn.ModuleList([])
        for i in range(self.num_lstm_layers):
            lstm_in = input_size + sum(hidden_sizes[:i]) * 2
            lstm = nn.LSTM(input_size=lstm_in, hidden_size=hidden_sizes[i], batch_first=True, bidirectional=True)
            self.lstms.append(lstm)

    def detach_hidden(self):
        pass

    def forward(self, input_sent):
        print("-"*10,"LSTM forward", "-"*10)
        final_hidden = None
        for i in range(self.num_lstm_layers):
            output, _ = self.lstms[i](input_sent)
            print("input_sent:", input_sent.size())
            print("output:", output.size())
            input_sent = torch.cat([input_sent, output], 2) # word dim
            print("After concat:", input_sent.size())
            final_hidden = output

        print("Final hidden layer:", final_hidden.size())
        max_pooling, _ = torch.max(final_hidden, 1) # reduce seq_len dim
        print("After max pooling:", max_pooling.size())
        return max_pooling


class SSClassifier(nn.Module):

    def __init__(self, params):
        super(SSClassifier, self).__init__()

        self.embeddings = nn.Embedding(params["vocab_size"], embedding_dim=params["embed_dim"])

        self.ssencoder = SSEncoder(params["embed_dim"], params['lstm_h'])

        self.num_mlp_layer = len(params['mlp_h'])
        mlps = []
        for i in range(self.num_mlp_layer):
            if i != 0:
                input_size = params['mlp_h'][i - 1]
            else:
                input_size = params['lstm_h'][-1] * 8  # vp:vh:vp-vh:vp*vh
            mlp = nn.Sequential(nn.Linear(input_size, params['mlp_h'][i]),
                                nn.ReLU(),
                                nn.Dropout(params['mlp_dr']))
            mlps.append(mlp)
        self.mlps = nn.Sequential(*mlps)


        self.classifer = nn.Linear(params['mlp_h'][-1], params["num_class"])

    def init_weight(self, pretrained_embedding):
        self.embeddings.weight = nn.Parameter(pretrained_embedding)


    def forward(self, premise, hypothesis):
        premise_embed = self.embeddings(premise)
        hypothesis_embed = self.embeddings(hypothesis)
        print("premise_embed:",premise_embed.size())
        print("hypothesis_embed:",hypothesis_embed.size())

        premise_hidden = self.ssencoder(premise_embed)
        hypothesis_hidden = self.ssencoder(hypothesis_embed)

        print("-"*10,"After encoder","-"*10,)
        print("premise_hidden:", premise_hidden.size())
        print("hypothesis_hidden:", hypothesis_hidden.size())

        m = torch.cat([premise_hidden,
                       hypothesis_hidden,
                       torch.abs(premise_hidden-hypothesis_hidden),
                       torch.mul(premise_hidden, hypothesis_hidden)], 2)

        print("")

        linear = self.mlps(m)
        prediction = self.classifer(linear)

        return prediction
