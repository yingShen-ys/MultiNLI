import csv
import sys
import os

from torch.autograd import Variable

from .model import ESIMClassifier
from .model import ESIMTreeClassifier
from ..utils import NLIDataloader
from ..utils import evaluate, combine_dataset

from sklearn.metrics import accuracy_score
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import math

seed = 233
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)


def switch_pre_hypo(options):
    # parse the input args
    run_id = options['run_id']
    pretained = options['pretained']
    model_path = options['model_path']
    multinli_path = options['multinli_data_path']
    snli_path = options['snli_data_path']
    gpu_option = options['gpu']

    if gpu_option >= 0:
        device = None
        print("CUDA available, running on gpu ", gpu_option)
    else:
        device = -1
        print("CUDA not available, running on cpu.")

    # prepare the paths for loading the models
    model_path = os.path.join(model_path, "model_{}.pt".format(run_id))

    params = dict()
    params['batch_sz'] = random.choice([32])

    # prepare the datasets
    (_snli_train_iter, _snli_val_iter, snli_test_iter), \
    (_multinli_train_iter, _multinli_match_iter, _multinli_mis_match_iter), \
    TEXT_FIELD, LABEL_FIELD \
        = NLIDataloader(multinli_path, snli_path, pretained).load_nlidata(batch_size=params["batch_sz"],
                                                                          gpu_option=device)

    criterion = nn.CrossEntropyLoss(size_average=False)
    best_model = torch.load(model_path)
    best_model.init_weight(TEXT_FIELD.vocab.vectors)
    best_model.cuda()
    best_model.eval()
    test_loss = 0.0
    predictions = []
    labels = []
    label_dict = {i:l for i, l in enumerate(LABEL_FIELD.vocab.itos)}
    # reverse_label_dict = {l:i for i, l in enumerate(LABEL_FIELD.vocab.itos)}

    print("Loaded the model from {}".format(model_path))

    hypothesis_list = []
    premise_list =[]
    for _, batch in enumerate(snli_test_iter):
        premise, _premis_lens = batch.premise
        hypothesis, _hypothesis_lens = batch.hypothesis
        label = batch.label
        # switch_label = batch.label.cpu().data.numpy()
        for ex_id in range(hypothesis.size()[0]):
            hypo = hypothesis[ex_id].cpu().data.numpy()
            prem = premise[ex_id].cpu().data.numpy()
            hypothesis_list.append(hypo)
            premise_list.append(prem)

        # switch_label = [l if label_dict[l] != "entailment" else reverse_label_dict["neutral"] for l in label]
        # switch_label = Variable(torch.from_numpy(np.array(true_label)) , requires_grad=False).cuda()

        output = best_model(premise=hypothesis, hypothesis=premise)
        loss = criterion(output, label)
        test_loss += loss.data[0] / len(snli_test_iter)

        predictions.append(output.cpu().data.numpy())
        labels.append(label.cpu().data.numpy())

    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)

    error_cases(predictions, labels, hypothesis_list, premise_list, TEXT_FIELD, label_dict, "temp.txt")


    f1, acc_score = evaluate(predictions, labels, LABEL_FIELD.vocab, "temp.jpg")

    print("Test F1:", f1)
    print("Binary Acc:", acc_score)

def snli_eval(options):
    # parse the input args
    run_id = options['run_id']
    pretained = options['pretained']
    model_path = options['model_path']
    multinli_path = options['multinli_data_path']
    snli_path = options['snli_data_path']
    gpu_option = options['gpu']

    if gpu_option >= 0:
        device = None
        print("CUDA available, running on gpu ", gpu_option)
    else:
        device = -1
        print("CUDA not available, running on cpu.")

    # prepare the paths for loading the models
    model_path = os.path.join(model_path, "model_{}.pt".format(run_id))

    params = dict()
    params['batch_sz'] = random.choice([32])

    # prepare the datasets
    (_snli_train_iter, _snli_val_iter, snli_test_iter), \
    (_multinli_train_iter, _multinli_match_iter, _multinli_mis_match_iter), \
    TEXT_FIELD, LABEL_FIELD \
        = NLIDataloader(multinli_path, snli_path, pretained).load_nlidata(batch_size=params["batch_sz"],
                                                                          gpu_option=device)

    criterion = nn.CrossEntropyLoss(size_average=False)
    best_model = torch.load(model_path)
    best_model.init_weight(TEXT_FIELD.vocab.vectors)
    best_model.cuda()
    best_model.eval()
    test_loss = 0.0
    predictions = []
    labels = []
    label_dict = {i:l for i, l in enumerate(LABEL_FIELD.vocab.itos)}
    # reverse_label_dict = {l:i for i, l in enumerate(LABEL_FIELD.vocab.itos)}

    print("Loaded the model from {}".format(model_path))

    hypothesis_list = []
    premise_list =[]
    for _, batch in enumerate(snli_test_iter):
        premise, _premis_lens = batch.premise
        hypothesis, _hypothesis_lens = batch.hypothesis
        label = batch.label
        for ex_id in range(hypothesis.size()[0]):
            hypo = hypothesis[ex_id].cpu().data.numpy()
            prem = premise[ex_id].cpu().data.numpy()
            hypothesis_list.append(hypo)
            premise_list.append(prem)

        output = best_model(premise=premise, hypothesis=hypothesis)
        loss = criterion(output, label)
        test_loss += loss.data[0] / len(snli_test_iter)

        predictions.append(output.cpu().data.numpy())
        labels.append(label.cpu().data.numpy())

    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)

    error_cases(predictions, labels, hypothesis_list, premise_list, TEXT_FIELD, label_dict, "snli.csv")


    f1, acc_score = evaluate(predictions, labels, LABEL_FIELD.vocab, "snli.jpg")

    print("Test F1:", f1)
    print("Binary Acc:", acc_score)


def multi_eval(options):
    # parse the input args
    run_id = options['run_id']
    signature = options['signature']
    pretained = options['pretained']
    model_path = options['model_path']
    multinli_path = options['multinli_data_path']
    snli_path = options['snli_data_path']
    gpu_option = options['gpu']

    if gpu_option >= 0:
        device = None
        print("CUDA available, running on gpu ", gpu_option)
    else:
        device = -1
        print("CUDA not available, running on cpu.")

    # prepare the paths for loading the models
    model_path = os.path.join(model_path, "model_{}_{}.pt".format(signature, run_id))

    params = dict()
    params['batch_sz'] = random.choice([32])

    # prepare the datasets
    (_snli_train_iter, _snli_val_iter, _snli_test_iter), \
    (_multinli_train_iter, multinli_match_iter, multinli_mis_match_iter), \
    TEXT_FIELD, LABEL_FIELD \
        = NLIDataloader(multinli_path, snli_path, pretained).load_nlidata(batch_size=params["batch_sz"],
                                                                          gpu_option=device)

    criterion = nn.CrossEntropyLoss(size_average=False)
    best_model = torch.load(model_path)
    best_model.eval()
    test_loss = 0.0

    label_dict = {i:l for i, l in enumerate(LABEL_FIELD.vocab.itos)}
    # reverse_label_dict = {l:i for i, l in enumerate(LABEL_FIELD.vocab.itos)}

    print("Loaded the model from {}".format(model_path))

    predictions = []
    labels = []
    hypothesis_list = []
    premise_list =[]
    for _, batch in enumerate(multinli_match_iter):
        premise, _premis_lens = batch.premise
        hypothesis, _hypothesis_lens = batch.hypothesis
        label = batch.label
        # switch_label = batch.label.cpu().data.numpy()
        for ex_id in range(hypothesis.size()[0]):
            hypo = hypothesis[ex_id].cpu().data.numpy()
            prem = premise[ex_id].cpu().data.numpy()
            hypothesis_list.append(hypo)
            premise_list.append(prem)

        # switch_label = [l if label_dict[l] != "entailment" else reverse_label_dict["neutral"] for l in label]
        # switch_label = Variable(torch.from_numpy(np.array(true_label)) , requires_grad=False).cuda()

        output = best_model(premise=premise, hypothesis=hypothesis)
        loss = criterion(output, label)
        test_loss += loss.data[0] / len(multinli_match_iter)

        predictions.append(output.cpu().data.numpy())
        labels.append(label.cpu().data.numpy())

    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)

    error_cases(predictions, labels, hypothesis_list, premise_list, TEXT_FIELD, label_dict, "match.csv")


    f1, acc_score = evaluate(predictions, labels, LABEL_FIELD.vocab, "multi_match.jpg")

    print("Test match F1:", f1)
    print("Binary match Acc:", acc_score)

    hypothesis_list = []
    premise_list = []
    predictions = []
    labels = []
    for _, batch in enumerate(multinli_mis_match_iter):
        premise, _premis_lens = batch.premise
        hypothesis, _hypothesis_lens = batch.hypothesis
        label = batch.label
        # switch_label = batch.label.cpu().data.numpy()
        for ex_id in range(hypothesis.size()[0]):
            hypo = hypothesis[ex_id].cpu().data.numpy()
            prem = premise[ex_id].cpu().data.numpy()
            hypothesis_list.append(hypo)
            premise_list.append(prem)

        # switch_label = [l if label_dict[l] != "entailment" else reverse_label_dict["neutral"] for l in label]
        # switch_label = Variable(torch.from_numpy(np.array(true_label)) , requires_grad=False).cuda()

        output = best_model(premise=premise, hypothesis=hypothesis)
        loss = criterion(output, label)
        test_loss += loss.data[0] / len(multinli_mis_match_iter)

        predictions.append(output.cpu().data.numpy())
        labels.append(label.cpu().data.numpy())

    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)

    error_cases(predictions, labels, hypothesis_list, premise_list, TEXT_FIELD, label_dict, "mismatch.csv")

    f1, acc_score = evaluate(predictions, labels, LABEL_FIELD.vocab, "multi_mismatch.jpg")

    print("Test mismatch F1:", f1)
    print("Binary mismatch Acc:", acc_score)



def error_cases(predictions, labels, hypothesis_list, premise_list, TEXT_FIELD, label_dict, error_file):
    predictions = np.argmax(predictions, axis=1)
    error_idx = np.where(predictions != labels)[0]

    with open(error_file, 'w+') as out:
        writer = csv.writer(out)
        writer.writerow(["Hypothesis", "Premise", 'Predictions', 'label'])

    for i in error_idx:
        hypo = [TEXT_FIELD.vocab.itos[w] for w in hypothesis_list[i].tolist()]
        prem = [TEXT_FIELD.vocab.itos[w] for w in premise_list[i].tolist()]
        predict = label_dict[predictions[i]]
        label = label_dict[labels[i]]
        # print("Hypothesis:", hypo)
        # print("Premise:", prem)
        # print("Predictions:", predict)
        # print("True label:", label)

        with open(error_file, 'a+') as out:
            writer = csv.writer(out)
            writer.writerow([hypo, prem, predict, label])


if __name__ == "__main__":
    OPTIONS = argparse.ArgumentParser()
    OPTIONS.add_argument('--run_id', dest='run_id', type=int, default=1)
    OPTIONS.add_argument('--signature', dest='signature', type=str, default="")
    OPTIONS.add_argument('--pretained', dest='pretained', type=str, default="glove.840B.300d")
    OPTIONS.add_argument('--embedding_dim', dest='embedding_dim', type=int, default=300)
    OPTIONS.add_argument('--multinli_data_path', dest='multinli_data_path',
                         type=str, default='../../data/multinli_1.0/')
    OPTIONS.add_argument('--snli_data_path', dest='snli_data_path',
                         type=str, default='../../data/snli_1.0/')
    OPTIONS.add_argument('--data', dest='data', default='snli')
    OPTIONS.add_argument('--model_path', dest='model_path',
                         type=str, default='saved_model/')
    OPTIONS.add_argument('--gpu', dest='gpu', type=int, default=1)

    PARAMS = vars(OPTIONS.parse_args())
    # multi_eval(PARAMS)
    snli_eval(PARAMS)