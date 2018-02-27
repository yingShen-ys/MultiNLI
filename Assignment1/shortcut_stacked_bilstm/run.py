from __future__ import print_function

import sys

from sklearn.metrics import f1_score, accuracy_score

sys.path.append('../../')
print(sys.path[0])

from Assignment1.shortcut_stacked_bilstm.model.ssclassifer import SSClassifier
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim


seed = 233
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

import os

from Assignment1.utils.data_loader import NLIDataloader



def main(options):
    # parse the input args
    run_id = options['run_id']
    epochs = options['epochs']
    patience = options['patience']
    pretained = options['pretained']
    embed_dim = options['embedding_dim']

    model_path = options['model_path']
    output_path = options['output_path']
    multinli_path = options['multinli_data_path']
    snli_path = options['snli_data_path']
    gpu_option = options['gpu']
    if gpu_option >= 0:
        USE_GPU = True
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        print("CUDA available, running on gpu ", gpu_option)
    else:
        USE_GPU = False
        print("CUDA not available, running on cpu.")

    print("Training initializing... Run ID is: {}".format(run_id))

    # prepare the paths for storing models and outputs
    model_path = os.path.join(model_path, "model_{}.pt".format(run_id))
    output_path = os.path.join(output_path, "results_{}.csv".format(run_id))
    print("Location for savedl models: {}".format(model_path))
    print("Grid search results are in: {}".format(output_path))



    # Set hyperparamters
    # fixed according to shortcut stacked encoder paper
    params = dict()
    params['batch_sz'] = random.choice([32])
    params['lr'] = random.choice([0.0002])
    params['lr_decay'] = random.choice([0.5])
    params['lstm_h'] = random.choice([[512, 1024, 2048]])
    params['mlp_h'] = random.choice([[1600]])
    params['mlp_dr'] = random.choice([0.1])

    # prepare the datasets
    snli_train_iter, snli_val_iter, snli_test_iter, TEXT_FIELD, LABEL_FIELD \
        = NLIDataloader(multinli_path, snli_path, pretained).load_nlidata(batch_size=params["batch_sz"],
                                                                          gpu_option=gpu_option)


    # build model
    params["vocab_size"] = len(TEXT_FIELD.vocab)
    params["num_class"] = len(LABEL_FIELD.vocab)
    params["embed_dim"] = embed_dim

    model = SSClassifier(params)
    model.init_weight(TEXT_FIELD.vocab.vectors)
    print("Model initialized")
    criterion = nn.CrossEntropyLoss(size_average=False)
    if USE_GPU:
        model.cuda()
    optimizer = optim.Adam(params=model.parameters(), lr=params['lr'])
    lr_schedular = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=params['lr_decay'])
    curr_patience = patience
    min_valid_loss = float('Inf')

    print(len(snli_train_iter))
    print(len(snli_val_iter))
    print(len(snli_test_iter))
    complete = True
    for e in range(epochs):
        lr_schedular.step()
        model.train()
        model.zero_grad()
        snli_train_iter.init_epoch()
        print("Epoch {}: learning rate decay to {}".format(e, lr_schedular.get_lr()[0]))

        train_loss = 0.0
        for _, batch in enumerate(snli_train_iter):
            model.zero_grad()

            premise, premis_lens = batch.premise
            hypothesis, hypothesis_lens = batch.hypothesis
            label = batch.label

            output = model(premise=premise, hypothesis=hypothesis)
            loss = criterion(output, label)
            train_loss += loss.data[0] / len(snli_train_iter)
            loss.backward()
            optimizer.step()

        if np.isnan(train_loss):
            print("Training: NaN values happened, rebooting...\n\n")
            complete = False
            break

        model.eval()
        valid_loss = 0.0
        for _, batch in enumerate(snli_val_iter):
            premise, premis_lens = batch.premise
            hypothesis, hypothesis_lens = batch.hypothesis
            label = batch.label
            output = model(premise=premise, hypothesis=hypothesis)
            loss = criterion(output, label)
            valid_loss += loss.data[0] / len(snli_val_iter)

        if np.isnan(valid_loss):
            print("Training: NaN values happened, rebooting...\n\n")
            complete = False
            break

        if (valid_loss < min_valid_loss):
            curr_patience = patience
            min_valid_loss = valid_loss
            torch.save(model, model_path)
            print("Found new best model, saving to disk...")
        else:
            curr_patience -= 1

    if complete:
        best_model = torch.load(model_path)
        best_model.eval()
        test_loss = 0.0
        predictions = []
        labels = []
        for _, batch in enumerate(snli_test_iter):
            premise, premis_lens = batch.premise
            hypothesis, hypothesis_lens = batch.hypothesis
            label = batch.label

            output = best_model(premise=premise, hypothesis=hypothesis)
            loss = criterion(output, label)
            test_loss += loss.data[0] / len(snli_test_iter)

            predictions.append(output.cpu().data.numpy().reshape(-1))
            labels.append(label.cpu().data.numpy().reshape(-1))

        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)

        predictions = np.argmax(predictions, axis=1)

        f1 = f1_score(labels, predictions, average='weighted')
        acc_score = accuracy_score(labels, predictions)

        print("Test F1:", f1)
        print("Binary Acc:", acc_score)


if __name__ == "__main__":
    OPTIONS = argparse.ArgumentParser()
    OPTIONS.add_argument('--run_id', dest='run_id', type=int, default=1)
    OPTIONS.add_argument('--epochs', dest='epochs', type=int, default=500)
    OPTIONS.add_argument('--patience', dest='patience', type=int, default=20)
    OPTIONS.add_argument('--pretained', dest='pretained', type=str, default="glove.840B.300d")
    OPTIONS.add_argument('--embedding_dim', dest='embedding_dim', type=int, default=300)
    OPTIONS.add_argument('--multinli_data_path', dest='multinli_data_path',
                         type=str, default='../../data/multinli_1.0/')
    OPTIONS.add_argument('--snli_data_path', dest='snli_data_path',
                         type=str, default='../../data/snli_1.0/')
    OPTIONS.add_argument('--model_path', dest='model_path',
                         type=str, default='models')
    OPTIONS.add_argument('--output_path', dest='output_path',
                         type=str, default='results')
    OPTIONS.add_argument('--gpu', dest='gpu', type=int, default=1)

    PARAMS = vars(OPTIONS.parse_args())
    main(PARAMS)