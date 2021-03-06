from __future__ import print_function
import sys
import os

sys.path.append("..")
from model import SSVAE # pylint: disable=E0611, C0413
from utils import NLIDataloader # pylint: disable=E0611, C0413
from utils import evaluate, combine_dataset, load_param # pylint: disable=E0611, C0413

from itertools import chain
from sklearn.metrics import accuracy_score
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.contrib.examples.util import print_and_log, set_seed
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, config_enumerate
from pyro.optim import Adam # pylint: disable=E0611, C0413, C0411

seed = 233
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)


# class RandomChain(object):
#     """
#     Chains two genrators randomly
#     """

#     def __init__(self, genA, genB):
#         self.genA = genA
#         self.genB = genB
#         self.probs = np.array([len(genA), len(genB)])
#         self.probs = self.probs / self.probs.sum() # normalize into probabilities

#     def __iter__(self):
#         self.lenA = 0
#         self.lenB = 0
#         return self
    
#     def __next__(self):
        



def main(args, config=None):
    run_id = args['run_id']
    signature = args['signature']
    epochs = args['epochs']
    patience = args['patience']
    model_name = args['model']
    model_path = args['model_path']
    output_path = args['output_path']
    multinli_path = args['multinli_data_path']
    snli_path = args['snli_data_path']
    gpu_option = args['gpu']
    resume = args['resume']
    hypertune = args['hypertune']
    print("Testing on SSVAE")
    if args['data'] == 'snli':
        input("SNLI dataset's results recording is not implemented correctly, still continue?")

    if gpu_option >= 0:
        USE_GPU = True
        device = None
        print("CUDA available, running on gpu ", gpu_option)
    else:
        USE_GPU = False
        device = -1
        print("CUDA not available, running on cpu.")

    print("Training initializing, experiment ID is {}.".format(run_id))

    # prepare the paths for storing models and outputs
    model_path = os.path.join(model_path, "model_{}_{}.pt".format(signature, run_id))
    output_path = os.path.join(output_path, "results_{}_{}.csv".format(signature, run_id))
    print("Location for saved models: {}".format(model_path))
    print("Grid search results are in: {}".format(output_path))

    # load a set of hyperparams
    if config is None and not hypertune:
        config = load_param("{}_{}_{}".format(model_name, run_id, signature))

    # define tokenizer
    tokenizer_method = 'spacy'

    # prepare the datasets
    (snli_train_iter, snli_val_iter, snli_test_iter), \
    (multinli_train_iter, multinli_match_iter, multinli_mis_match_iter),\
    TEXT_FIELD, LABEL_FIELD, GENRE_FIELD \
        = NLIDataloader(multinli_path, snli_path, config["pretrained"]).load_nlidata(batch_size=config["batch_sz"],
                                                                                    gpu_option=device, tokenizer=tokenizer_method)
    print("Dataset loaded.")
    vocab_size = len(TEXT_FIELD.vocab)
    num_class = len(LABEL_FIELD.vocab)
    num_genre = len(GENRE_FIELD.vocab)
    print("There are {} many genres in the data.".format(num_genre))

    # pick the training, validation, testing sets
    if args["data"] == "snli":
        train_iter = snli_train_iter
        val_iter = snli_val_iter
        test_iter = snli_test_iter
        num_samples = len(snli_train_iter)
    elif args['data'] == "multi": # multinli
        train_iter = multinli_train_iter
        num_samples = len(multinli_train_iter)
        val_iter = snli_val_iter
        test_match_iter = multinli_match_iter
        test_mismatch_iter = multinli_mis_match_iter
    else: # all
        # train_iter = combine_dataset(snli_train_iter, multinli_train_iter)
        train_iter = chain(multinli_train_iter, snli_train_iter)
        num_samples = len(multinli_train_iter) + len(snli_train_iter)
        val_iter = snli_val_iter
        test_match_iter = multinli_match_iter
        test_mismatch_iter = multinli_mis_match_iter

    # build model
    if not resume:
        model = SSVAE(vocab_size, output_size=num_class, config_enum='parallel', aux_loss_multiplier=config['llm'])
    else:
        model = torch.load(model_path)

    model.init_weight(TEXT_FIELD.vocab.vectors)
    print("Model initialized")
    optimizer = Adam({'lr': config['lr'], 'betas': (0.9, 0.999)})
    guide = config_enumerate(model.guide, 'parallel')
    generative_loss = SVI(model.model, guide, optimizer,
                          loss=TraceEnum_ELBO(max_iarange_nesting=1))
    discriminative_loss = SVI(model.model_classify,
                              model.guide_classify, optimizer, loss=Trace_ELBO())
    losses = [generative_loss, discriminative_loss]
    if USE_GPU:
        model.cuda()

    # lr_schedular = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=config['lr_decay'])
    curr_patience = patience
    min_valid_loss = float('Inf')
    best_valid_acc = - float('Inf')

    # training & validation
    complete = True
    for e in range(epochs):
    #     lr_schedular.step()
        model.train()
        model.zero_grad()
        snli_train_iter.init_epoch()
        snli_val_iter.init_epoch()
        epoch_losses_sup = np.array([0.] * len(losses))
        epoch_losses_unsup = np.array([0.] * len(losses))
        # print("Epoch {}: learning rate decay to {}".format(e, lr_schedular.get_lr()[0]))
        predictions_y = []
        labels = []
        # genres = []
        for batch_idx, batch in enumerate(train_iter):
            model.zero_grad()

            premise, _premis_lens = batch.premise
            hypothesis, _hypothesis_lens = batch.hypothesis
            label = batch.label
            # try:
            #     genre = batch.genre
            # except AttributeError:
            #     genre = None

            for loss_id, loss in enumerate(losses):
                new_loss = loss.step(premise, hypothesis, label)
                epoch_losses_unsup[loss_id] += new_loss

            output_y = model.classifier(premise, hypothesis)
            predictions_y.append(output_y.cpu().data.numpy())
            labels.append(label.cpu().data.numpy())

            if (batch_idx+1) % 500 == 0:
                print("Batch {}/{} complete! Supervised training loss {}, unsupervised {}".format(batch_idx+1, num_samples, epoch_losses_sup/(batch_idx+1), epoch_losses_unsup/(batch_idx+1)))

        # if np.isnan(train_loss):
        #     print("Training: NaN values happened, rebooting...\n\n")
        #     complete = False
        #     break

        predictions_y = np.concatenate(predictions_y)
        labels = np.concatenate(labels)

        predictions_y = np.argmax(predictions_y, axis=1)
        label_acc_score = accuracy_score(labels, predictions_y)
        print("Epoch {}/{} complete! Label accuracy: {}".format(e, epochs, label_acc_score))

        model.eval()
        valid_losses_sup = [0.] * len(losses)
        valid_losses_unsup = [0.] * len(losses)
        predictions_y = []
        labels = []
        for _, batch in enumerate(val_iter):
            premise, _premis_lens = batch.premise
            hypothesis, _hypothesis_lens = batch.hypothesis
            label = batch.label

            for loss_id, loss in enumerate(losses):
                new_loss = loss.evaluate_loss(premise, hypothesis, label)
                valid_losses_unsup[loss_id] += new_loss

            output_y = model.classifier(premise, hypothesis)
            predictions_y.append(output_y.cpu().data.numpy())
            labels.append(label.cpu().data.numpy())

        # if np.isnan(train_loss):
        #     print("Training: NaN values happened, rebooting...\n\n")
        #     complete = False
        #     break
        valid_loss = sum(valid_losses_sup) + sum(valid_losses_unsup)
        print("Validation loss: supervised: {}; unsupervised loss: {}; total: {}".format(valid_losses_sup, valid_losses_unsup, valid_loss))

        predictions_y = np.concatenate(predictions_y)
        # predictions_g = np.concatenate(predictions_g)
        labels = np.concatenate(labels)
        # genres = np.concatenate(genres)

        predictions_y = np.argmax(predictions_y, axis=1)
        label_acc_score = accuracy_score(labels, predictions_y)
        # predictions_g = np.argmax(predictions_g, axis=1)
        # genre_acc_score = accuracy_score(genres, predictions_g)
        print("Validation label accuracy: {}".format(label_acc_score))

        best_valid_acc = max(label_acc_score, best_valid_acc)
        if valid_loss < min_valid_loss:
            valid_acc_at_minimal_loss = label_acc_score
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
        predictions = []
        labels = []
        if args["data"] == "snli":
            for _, batch in enumerate(test_iter):
                premise, _premis_lens = batch.premise
                hypothesis, _hypothesis_lens = batch.hypothesis
                label = batch.label

                output = best_model.classifier(premise, hypothesis)
                predictions.append(output.cpu().data.numpy())
                labels.append(label.cpu().data.numpy())

                predictions = np.concatenate(predictions)
                labels = np.concatenate(labels)

                f1, acc_score = evaluate(predictions, labels, LABEL_FIELD.vocab, "snli_cm.jpg")

                print("Test F1:", f1)
                print("Entailment Classification Acc:", acc_score)
        else:  # multinli
            predictions_match = []
            labels_match = []
            predictions_mismatch = []
            labels_mismatch = []
            for _, batch in enumerate(test_match_iter):
                premise, _premis_lens = batch.premise
                hypothesis, _hypothesis_lens = batch.hypothesis
                label = batch.label

                output = best_model.classifier(premise, hypothesis)
                predictions_match.append(output.cpu().data.numpy())
                labels_match.append(label.cpu().data.numpy())

            for _, batch in enumerate(test_mismatch_iter):
                premise, _premis_lens = batch.premise
                hypothesis, _hypothesis_lens = batch.hypothesis
                label = batch.label

                output = best_model.classifer(premise, hypothesis)
                predictions_mismatch.append(output.cpu().data.numpy())
                labels_mismatch.append(label.cpu().data.numpy())

            predictions_match = np.concatenate(predictions_match)
            labels_match = np.concatenate(labels_match)
            predictions_mismatch = np.concatenate(predictions_mismatch)
            labels_mismatch = np.concatenate(labels_mismatch)

            f1_match, acc_score_match = evaluate(predictions_match, labels_match, LABEL_FIELD.vocab, "match_cm.jpg")
            f1_mismatch, acc_score_mismatch = evaluate(predictions_mismatch, labels_mismatch, LABEL_FIELD.vocab, "mismatch_cm.jpg")

            print("Test match F1:", f1_match)
            print("Entailment classification (match) Acc:", acc_score_match)

            print("Test mismatch F1:", f1_mismatch)
            print("Entailment classification (mismatch) Acc:", acc_score_mismatch)
    return np.array([valid_acc_at_minimal_loss, min_valid_loss, f1_match, acc_score_match, f1_mismatch, acc_score_mismatch])

if __name__ == "__main__":
    OPTIONS = argparse.ArgumentParser()
    OPTIONS.add_argument('--run_id', dest='run_id', type=int, default=1)
    OPTIONS.add_argument('--cell', dest='cell', type=str, default='lstm')
    OPTIONS.add_argument('--hypertune', dest='hypertune', action='store_true')
    OPTIONS.add_argument('--signature', dest='signature', type=str, default="") # e.g. {model}_{data}
    OPTIONS.add_argument('--model', dest='model', type=str, default="ssvae")
    OPTIONS.add_argument('--epochs', dest='epochs', type=int, default=500)
    OPTIONS.add_argument('--patience', dest='patience', type=int, default=20)
    OPTIONS.add_argument('--multinli_data_path', dest='multinli_data_path',
                         type=str, default='../../data/multinli_1.0/')
    OPTIONS.add_argument('--snli_data_path', dest='snli_data_path',
                         type=str, default='../../data/snli_1.0/')
    OPTIONS.add_argument('--data', dest='data', default='snli')
    OPTIONS.add_argument('--model_path', dest='model_path',
                         type=str, default='saved_model/')
    OPTIONS.add_argument('--output_path', dest='output_path',
                         type=str, default='results/')
    OPTIONS.add_argument('--gpu', dest='gpu', type=int, default=-1)
    OPTIONS.add_argument('--resume', dest='resume', action='store_true')

    ARGS = vars(OPTIONS.parse_args())
    main(ARGS)
