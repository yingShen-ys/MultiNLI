from __future__ import print_function
import sys
import os


sys.path.append("..")
from model.ESIM_classifier import ESIMClassifier
from model.ESIM_tree import ESIMTreeClassifier
from model.ssclassifier import SSClassifier
from utils import NLIDataloader
from utils import evaluate, combine_dataset, load_param


from sklearn.metrics import accuracy_score
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


def main(options):
    # parse the input args
    run_id = options['run_id']
    signature = options['signature']
    epochs = options['epochs']
    patience = options['patience']

    model_path = options['model_path']
    output_path = options['output_path']
    multinli_path = options['multinli_data_path']
    snli_path = options['snli_data_path']
    gpu_option = options['gpu']

    if gpu_option >= 0:
        USE_GPU = True
        device = None
        print("CUDA available, running on gpu ", gpu_option)
    else:
        USE_GPU = False
        device = -1
        print("CUDA not available, running on cpu.")

    print("Training initializing... Run ID is: {}".format(run_id))

    # prepare the paths for storing models and outputs
    model_path = os.path.join(model_path, "model_{}_{}.pt".format(signature, run_id))
    output_path = os.path.join(output_path, "results_{}_{}.csv".format(signature, run_id))
    print("Location for savedl models: {}".format(model_path))
    print("Grid search results are in: {}".format(output_path))

    # load hyperparamters
    config = load_param(options["model"])

    # pick tokenizer method
    if options["model"] == "ssbilstm":
        tokenizer_method = 'spacy'
    else:
        tokenizer_method =' '

    # prepare the datasets
    (snli_train_iter, snli_val_iter, snli_test_iter), \
    (multinli_train_iter, multinli_match_iter, multinli_mis_match_iter),\
    TEXT_FIELD, LABEL_FIELD \
        = NLIDataloader(multinli_path, snli_path, config["pretained"]).load_nlidata(batch_size=config["batch_sz"],
                                                                                    gpu_option=device, tokenizer=tokenizer_method)

    # pick the training, validation, testing sets
    if options["data"] == "snli":
        train_iter = snli_train_iter
        val_iter = snli_val_iter
        test_iter = snli_test_iter
    else: # multinli
        # train_iter = combine_dataset(snli_train_iter, multinli_train_iter)
        train_iter = multinli_train_iter
        val_iter = snli_val_iter
        test_match_iter = multinli_match_iter
        test_mismatch_iter = multinli_mis_match_iter

    # build model
    config["vocab_size"] = len(TEXT_FIELD.vocab)
    config["num_class"] = len(LABEL_FIELD.vocab)

    # pick the classification model
    if options["model"] == "esim":
        print("using ESIM classifier")
        model = ESIMClassifier(config)
    elif options["model"] == "ssbilstm":
        print("using shortcut stack classifier")
        model = SSClassifier(config)
    else:
        print("using ESIM-tree classifier")
        options["model"] = "esim_tree"
        model = ESIMTreeClassifier(config)

    model.init_weight(TEXT_FIELD.vocab.vectors)
    print("Model initialized")
    criterion = nn.CrossEntropyLoss(size_average=False)
    if USE_GPU:
        model.cuda()

    # model = torch.load(model_path)
    config["lr"] = 2.5e-05
    epochs = 13
    optimizer = optim.Adam(params=model.parameters(), lr=config['lr'])
    lr_schedular = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=config['lr_decay'])
    curr_patience = patience
    min_valid_loss = float('Inf')

    # training & validation
    complete = True
    for e in range(epochs):
        lr_schedular.step()
        model.train()
        model.zero_grad()

        snli_train_iter.init_epoch()
        snli_val_iter.init_epoch()

        print("Epoch {}: learning rate decay to {}".format(e, lr_schedular.get_lr()[0]))

        train_loss = 0.0
        predictions = []
        labels = []
        skip_cnt = 0
        for batch_idx, batch in enumerate(train_iter):
            model.zero_grad()
            
            if options["model"] != "esim_tree":
                premise, _premis_lens = batch.premise
                hypothesis, _hypothesis_lens = batch.hypothesis
            else:
                premise, _ = batch.premise_parse
                hypothesis, _ = batch.hypothesis_parse
            label = batch.label
            pairID = batch.pairID

            try:
                output = model(premise=premise, hypothesis=hypothesis)
                loss = criterion(output, label)
                train_loss += loss.data[0] / len(train_iter)

                loss.backward()
                nn.utils.clip_grad_norm(model.parameters(), config['clip_c'])
                optimizer.step()

                predictions.append(output.cpu().data.numpy())
                labels.append(label.cpu().data.numpy())
            except:
                skip_cnt += 1
                print("skip {} examples: {}".format(skip_cnt, pairID))

            if batch_idx % 100 == 0:
                print("Batch {}/{} complete! Average training loss {}".format(batch_idx, len(train_iter), loss.data[0]/ batch.batch_size))

        if np.isnan(train_loss):
            print("Training: NaN values happened, rebooting...\n\n")
            complete = False
            break

        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)

        predictions = np.argmax(predictions, axis=1)
        acc_score = accuracy_score(labels, predictions)
        print("Epoch {} complete! Average Training loss: {}".format(e, train_loss))
        print("Epoch {} complete! Training Accuracy: {}".format(e, acc_score))


        model.eval()
        valid_loss = 0.0
        predictions = []
        labels = []
        for _, batch in enumerate(val_iter):
            if options["model"] != "esim_tree":
                premise, _premis_lens = batch.premise
                hypothesis, _hypothesis_lens = batch.hypothesis
            else:
                premise, _ = batch.premise_parse
                hypothesis, _ = batch.hypothesis_parse
            label = batch.label
            pairID = batch.pairID
            try:
                output = model(premise=premise, hypothesis=hypothesis)
                loss = criterion(output, label)
                valid_loss += loss.data[0] / len(val_iter)

                predictions.append(output.cpu().data.numpy())
                labels.append(label.cpu().data.numpy())

        if np.isnan(valid_loss):
            print("Training: NaN values happened, rebooting...\n\n")
            complete = False
            break

        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)

        predictions = np.argmax(predictions, axis=1)
        acc_score = accuracy_score(labels, predictions)
        print("Epoch {} complete! Average Validation loss: {}".format(e, valid_loss))
        print("Epoch {} complete! Validation Accuracy: {}".format(e, acc_score))

        if (valid_loss < min_valid_loss):
            curr_patience = patience
            min_valid_loss = valid_loss
            torch.save(model, model_path)
            print("Found new best model, saving to disk...")
        else:
            curr_patience -= 1

        if curr_patience <= 0:
            break

    # testing
    if complete:
        best_model = torch.load(model_path)
        best_model.eval()
        test_loss = 0.0
        predictions = []
        labels = []
        if options["data"] == "snli":
            skip_cnt = 0
            for _, batch in enumerate(test_iter):
                if options["model"] != "esim_tree":
                    premise, _premis_lens = batch.premise
                    hypothesis, _hypothesis_lens = batch.hypothesis
                else:
                    premise, _ = batch.premise_parse
                    hypothesis, _ = batch.hypothesis_parse
                label = batch.label
                pairID = batch.pairID

                try:
                    output = best_model(premise=premise, hypothesis=hypothesis)
                    loss = criterion(output, label)
                    test_loss += loss.data[0] / len(snli_test_iter)

                    predictions.append(output.cpu().data.numpy())
                    labels.append(label.cpu().data.numpy())

                    predictions = np.concatenate(predictions)
                    labels = np.concatenate(labels)

                    f1, acc_score = evaluate(predictions, labels, LABEL_FIELD.vocab, "snli_cm.jpg")

                    print("Test F1:", f1)
                    print("Binary Acc:", acc_score)
                except:
                    skip_cnt += 1
                    print("test skip {} examples: {}".format(skip_cnt, pairID))
        else:  # multinli
            predictions_match = []
            labels_match = []
            predictions_mismatch = []
            labels_mismatch = []
            for _, batch in enumerate(test_match_iter):
                if options["model"] != "esim_tree":
                    premise, _premis_lens = batch.premise
                    hypothesis, _hypothesis_lens = batch.hypothesis
                else:
                    premise, _ = batch.premise_parse
                    hypothesis, _ = batch.hypothesis_parse
                label = batch.label

                output = best_model(premise=premise, hypothesis=hypothesis)
                loss = criterion(output, label)
                test_loss += loss.data[0] / len(test_match_iter)

                predictions_match.append(output.cpu().data.numpy())
                labels_match.append(label.cpu().data.numpy())

            for _, batch in enumerate(test_mismatch_iter):
                if options["model"] != "esim_tree":
                    premise, _premis_lens = batch.premise
                    hypothesis, _hypothesis_lens = batch.hypothesis
                else:
                    premise, _ = batch.premise_parse
                    hypothesis, _ = batch.hypothesis_parse
                label = batch.label

                output = best_model(premise=premise, hypothesis=hypothesis)
                loss = criterion(output, label)
                test_loss += loss.data[0] / len(test_mismatch_iter)

                predictions_mismatch.append(output.cpu().data.numpy())
                labels_mismatch.append(label.cpu().data.numpy())

            predictions_match = np.concatenate(predictions_match)
            labels_match = np.concatenate(labels_match)
            predictions_mismatch = np.concatenate(predictions_mismatch)
            labels_mismatch = np.concatenate(labels_mismatch)

            f1_match, acc_score_match = evaluate(predictions_match, labels_match, LABEL_FIELD.vocab, "match_cm.jpg")
            f1_mismatch, acc_score_mismatch = evaluate(predictions_mismatch, labels_mismatch, LABEL_FIELD.vocab, "mismatch_cm.jpg")

            print("Test match F1:", f1_match)
            print("Binary match Acc:", acc_score_match)

            print("Test mismatch F1:", f1_mismatch)
            print("Binary mismatch Acc:", acc_score_mismatch)


if __name__ == "__main__":
    OPTIONS = argparse.ArgumentParser()
    OPTIONS.add_argument('--run_id', dest='run_id', type=int, default=1)
    OPTIONS.add_argument('--signature', dest='signature', type=str, default="") # e.g. {model}_{data}

    OPTIONS.add_argument('--epochs', dest='epochs', type=int, default=500)
    OPTIONS.add_argument('--patience', dest='patience', type=int, default=20)
    OPTIONS.add_argument('--multinli_data_path', dest='multinli_data_path',
                         type=str, default='../../data/multinli_1.0/')
    OPTIONS.add_argument('--snli_data_path', dest='snli_data_path',
                         type=str, default='../../data/snli_1.0/')
    # OPTIONS.add_argument('--data', dest='data', default='snli')
    OPTIONS.add_argument('--data', dest='data', default='snli')
    OPTIONS.add_argument('--model', dest='model', default='esim')
    OPTIONS.add_argument('--model_path', dest='model_path',
                         type=str, default='saved_model/')
    OPTIONS.add_argument('--output_path', dest='output_path',
                         type=str, default='results/')
    OPTIONS.add_argument('--gpu', dest='gpu', type=int, default=-1)

    PARAMS = vars(OPTIONS.parse_args())
    main(PARAMS)