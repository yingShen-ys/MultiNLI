import torch
import numpy as np
import torch.nn as nn

seed = 233
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)


class ESIMClassifier(nn.Module):
	"""
	The ESIM Classifier
	Input: premise, hypothesis
	Output: probabilities of Classes
	"""
	def __init__(self, params):

		super(ESIMClassifier, self).__init__()

		self.embeddings = nn.Sequential(nn.Embedding(params["vocab_size"], embedding_dim=params["embed_dim"]),
										nn.Dropout(params['mlp_dr']))
		self.bilstm_encoding = nn.LSTM(input_size=params["embed_dim"], hidden_size=params["lstm_h"], batch_first=True, bidirectional=True)
		input_size = params['lstm_h'] * 8
		self.mapping = nn.Sequential(nn.Linear(input_size, params['F_h']),
									 nn.ReLU(),
									 nn.Dropout(params['mlp_dr']))
		self.bilstm_infer = nn.LSTM(input_size=params['F_h'], hidden_size=params['lstm_h'], batch_first=True, bidirectional=True)
		self.final_mlp = nn.Sequential(nn.Dropout(params['mlp_dr']),
									   nn.Linear(input_size, params['lstm_h']),
									   nn.Tanh(),
									   nn.Dropout(params['mlp_dr']))
		self.softmax_layer = nn.Sequential(nn.Linear(params['lstm_h'], params['num_class']),
										   nn.Softmax())

	def init_weight(self, pretrained_embedding):
		"""
		Initialize the weight for the embedding layer using pretrained_embedding
		:param pretrained_embedding:
		:return:
		"""
		self.embeddings.weight = nn.Parameter(pretrained_embedding)

	
	def forward(self, premise, hypothesis):
		"""
		Forward function
		:param premise:
		:param hypothesis:
		:return:
		"""
		# Input Encoding
		premise_embed = self.embeddings(premise)
		hypothesis_embed = self.embeddings(hypothesis)
		premise_out, _ = self.bilstm_encoding(premise_embed)
		hypothesis_out, _ = self.bilstm_encoding(hypothesis_embed)

		# Local Inference Modeling
		e_matrix = torch.matmul(premise_out, torch.transpose(hypothesis_out, 1, 2))

		e_max_1, _ = torch.max(e_matrix, 1, keepdim=True)
		e_max_2, _ = torch.max(e_matrix, 2, keepdim=True)
		e_matrix_1 = torch.exp(e_matrix - e_max_1)
		e_matrix_2 = torch.exp(e_matrix - e_max_2)

		w_2 = torch.div(e_matrix_2, torch.sum(e_matrix_2, 2, True)) # premise_w
		w_1 = torch.div(e_matrix_1, torch.sum(e_matrix_1, 1, True)) # hypothesis_w

		premise_tilde = torch.matmul(w_2, hypothesis_out)
		hypothesis_tilde = torch.matmul(torch.transpose(w_1, 1, 2), premise_out)

		premise_m = torch.cat([premise_out,
							   premise_tilde,
							   premise_out - premise_tilde,
							   torch.mul(premise_out, premise_tilde)], 2)
		hypothesis_m = torch.cat([hypothesis_out,
								  hypothesis_tilde,
								  hypothesis_out - hypothesis_tilde,
								  torch.mul(hypothesis_out, hypothesis_tilde)], 2)

		# ReLU mapping to reduce complexity
		premise_m_mapping = self.mapping(premise_m)
		hypothesis_m_mapping = self.mapping(hypothesis_m)

		# Inference Composition
		premise_infer, _ = self.bilstm_infer(premise_m_mapping)
		hypothesis_infer, _ = self.bilstm_infer(hypothesis_m_mapping)

		# Pooling
		premise_max, _ = torch.max(premise_infer, 1)
		hypothesis_max, _ = torch.max(hypothesis_infer, 1)
		premise_v = torch.cat([torch.mean(premise_infer, 1), premise_max], 1)
		hypothesis_v = torch.cat([torch.mean(hypothesis_infer, 1), hypothesis_max], 1)
		
		v = torch.cat([premise_v, hypothesis_v], 1)

		# final classifier
		prediction = self.final_mlp(v)
		softmax_pred = self.softmax_layer(prediction)

		return softmax_pred
