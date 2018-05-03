from __future__ import print_function
import sys
import os

sys.path.append("..")
from model import GenreAgnosticInference # pylint: disable=E0611, C0413
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


def main(args):
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
    config = load_param("{}_{}_{}.json".format(model_name, run_id, signature))

    # define tokenizer
    tokenizer_method = 'spacy'

    # prepare the datasets
    (snli_train_iter, snli_val_iter, snli_test_iter), \
    (multinli_train_iter, multinli_match_iter, multinli_mis_match_iter),\
    TEXT_FIELD, LABEL_FIELD, GENRE_FIELD \
        = NLIDataloader(multinli_path, snli_path, config["pretained"]).load_nlidata(batch_size=config["batch_sz"],
                                                                                    gpu_option=device, tokenizer=tokenizer_method)
    vocab_size = len(TEXT_FIELD.vocab)
    num_class = len(LABEL_FIELD.vocab)
    num_genre = len(GENRE_FIELD.vocab)
    print("There are {} many genres in the data.".format(num_genre))

    # pick the training, validation, testing sets
    if args["data"] == "snli":
        train_iter = snli_train_iter
        val_iter = snli_val_iter
        test_iter = snli_test_iter
    else: # multinli
        # train_iter = combine_dataset(snli_train_iter, multinli_train_iter)
        train_iter = chain(multinli_train_iter, snli_train_iter)
        val_iter = snli_val_iter
        test_match_iter = multinli_match_iter
        test_mismatch_iter = multinli_mis_match_iter

    # build model
    if not resume:
        model = GenreAgnosticInference(config['x_dim'], num_class, num_genre, config['zy_dim'],
                                   config['zg_dim'], vocab_size, config['embed_dim'],
                                   config['hidden'], config['llm'], config['glm'])
    else:
        model = torch.load(model_path)

    model.init_weight(TEXT_FIELD.vocab.vectors)
    print("Model initialized")
    optimizer = Adam(params=model.parameters(), lr=config['lr'])
    guide = config_enumerate(model.generative_guide, 'paralell')
    generative_loss = SVI(model.generative_model, guide, optimizer,
                          loss=TraceEnum_ELBO(max_iarange_nesting=1))
    discriminative_loss = SVI(model.discriminative_model,
                              model.discriminative_guide, optimizer, loss=Trace_ELBO())
    losses = [generative_loss, discriminative_loss]
    if USE_GPU:
        model.cuda()

    # lr_schedular = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=config['lr_decay'])
    curr_patience = patience
    min_valid_loss = float('Inf')

    # training & validation
    complete = True
    for e in range(epochs):
    #     lr_schedular.step()
        model.train()
        model.zero_grad()
        snli_train_iter.init_epoch()
        snli_val_iter.init_epoch()
        epoch_losses_sup = [0.] * len(losses)
        epoch_losses_unsup = [0.] * len(losses)
        # print("Epoch {}: learning rate decay to {}".format(e, lr_schedular.get_lr()[0]))
        predictions_y = []
        predictions_g = []
        labels = []
        genres = []
        for batch_idx, batch in enumerate(train_iter):
            model.zero_grad()
            
            premise, _premis_lens = batch.premise
            hypothesis, _hypothesis_lens = batch.hypothesis
            label = batch.label
            try:
                genre = batch.genre
            except AttributeError:
                genre = None

            for loss_id, loss in enumerate(losses):
                if genre is not None:
                    new_loss = loss.step(premise, hypothesis, label, genre)
                    print("Loss produced by Pyro is of type: {}".format(type(new_loss)))
                    epoch_losses_sup[loss_id] += new_loss
                else:
                    new_loss = loss.step(premise, hypothesis, label)
                    epoch_losses_unsup[loss_id] += new_loss

            output_y, output_g = model(premise, hypothesis)
            predictions_y.append(output_y.cpu().data.numpy())
            labels.append(label.cpu().data.numpy())
            if genre is not None:
                genres.append(genre.cpu().data.numpy())
                predictions_g.append(output_g.cpu().data.numpy())

            if (batch_idx+1) % 100 == 0:
                print("Batch {}/{} complete! Supervised training\
                      loss {}, unsupervised {}".format(batch_idx+1,
                      len(train_iter), epoch_losses_sup, epoch_losses_unsup))

        # if np.isnan(train_loss):
        #     print("Training: NaN values happened, rebooting...\n\n")
        #     complete = False
        #     break

        predictions_y = np.concatenate(predictions_y)
        predictions_g = np.concatenate(predictions_g)
        labels = np.concatenate(labels)
        genres = np.concatenate(genres)

        predictions_y = np.argmax(predictions_y, axis=1)
        label_acc_score = accuracy_score(labels, predictions_y)
        predictions_g = np.argmax(predictions_g, axis=1)
        genre_acc_score = accuracy_score(labels, predictions_g)
        print("Epoch {} complete! Label accuracy: {}; genre\
              accuracy: {}".format(e, label_acc_score, genre_acc_score))


        model.eval()
        valid_losses_sup = [0.] * len(losses)
        valid_losses_unsup = [0.] * len(losses)
        predictions_g = []
        predictions_y = []
        labels = []
        genres = []
        for _, batch in enumerate(val_iter):
            premise, _premis_lens = batch.premise
            hypothesis, _hypothesis_lens = batch.hypothesis
            label = batch.label
            try:
                genre = batch.genre
            except AttributeError:
                genre = None

            for loss_id, loss in enumerate(losses):
                if genre is not None:
                    new_loss = loss.evaluate_loss(premise, hypothesis, label, genre)
                    valid_losses_sup[loss_id] += new_loss
                else:
                    new_loss = loss.evaluate_loss(premise, hypothesis, label)
                    valid_losses_unsup[loss_id] += new_loss

            output_y, output_g = model(premise, hypothesis)
            predictions_y.append(output_y.cpu().data.numpy())
            labels.append(label.cpu().data.numpy())
            if genre is not None:
                genres.append(genre.cpu().data.numpy())
                predictions_g.append(output_g.cpu().data.numpy())

        # if np.isnan(train_loss):
        #     print("Training: NaN values happened, rebooting...\n\n")
        #     complete = False
        #     break
        valid_loss = sum(valid_losses_sup) + sum(valid_losses_unsup)
        print("Validation loss: supervised: {}; unsupervised\
              loss: {}; total: {}".format(valid_losses_sup, valid_losses_unsup, valid_loss))

        predictions_y = np.concatenate(predictions_y)
        predictions_g = np.concatenate(predictions_g)
        labels = np.concatenate(labels)
        genres = np.concatenate(genres)

        predictions_y = np.argmax(predictions_y, axis=1)
        label_acc_score = accuracy_score(labels, predictions_y)
        predictions_g = np.argmax(predictions_g, axis=1)
        genre_acc_score = accuracy_score(labels, predictions_g)
        print("Validation label accuracy: {}; genre accuracy:\
              {}".format(label_acc_score, genre_acc_score))

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
        predictions = []
        labels = []
        genres = []
        if args["data"] == "snli":
            for _, batch in enumerate(test_iter):
                premise, _premis_lens = batch.premise
                hypothesis, _hypothesis_lens = batch.hypothesis
                label = batch.label
                try:
                    genre = batch.genre
                except AttributeError:
                    genre = None

                output, _ = best_model(premise, hypothesis)
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
                try:
                    genre = batch.genre
                except AttributeError:
                    genre = None

                output, _ = best_model(premise, hypothesis)
                predictions_match.append(output.cpu().data.numpy())
                labels_match.append(label.cpu().data.numpy())

            for _, batch in enumerate(test_mismatch_iter):
                premise, _premis_lens = batch.premise
                hypothesis, _hypothesis_lens = batch.hypothesis
                label = batch.label
                try:
                    genre = batch.genre
                except AttributeError:
                    genre = None

                output, _ = best_model(premise, hypothesis)
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


if __name__ == "__main__":
    OPTIONS = argparse.ArgumentParser()
    OPTIONS.add_argument('--run_id', dest='run_id', type=int, default=1)
    OPTIONS.add_argument('--signature', dest='signature', type=str, default="") # e.g. {model}_{data}
    OPTIONS.add_argument('--model', dest='model', type=str, default="gai")
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
    OPTIONS.add_argument('--resume', action='store_true')

    ARGS = vars(OPTIONS.parse_args())
    main(ARGS)
