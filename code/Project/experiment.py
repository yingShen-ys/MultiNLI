import sys
sys.path.append("..")
from model import GenreAgnosticInference, GAIA, ELBO # pylint: disable=E0611, C0413
from utils import NLIDataloader # pylint: disable=E0611, C0413
from utils import evaluate, combine_dataset, load_param # pylint: disable=E0611, C0413
from torch.optim import Adam
import numpy as np
from sklearn.metrics import accuracy_score
import torch
import time

multinli_path = '../../data/multinli_1.0/'
snli_path = '../../data/snli_1.0/'
model_path = './saved_models/final_model2.pt'
epochs = 20
llm = 45
glm = 25
lr1 = 3e-3
lr2 = 5e-4
lr3 = 7e-4
gpu = 0


model = GAIA(x_dim=1200, y_dim=3,
             g_dim=10, zy_dim=60,
             zg_dim=200, rnn_type='gru',
             vocab_size=88376, embedding_size=300,
             hidden_size=300)

lstm_params = model.sentence_encoder.parameters()
encoders_params = list(model.encoder_g.parameters()) + list(model.encoder_y.parameters()) + list(model.encoder_zg.parameters()) + list(model.encoder_zy.parameters())
decoders_params = list(model.decoder_g.parameters()) + list(model.decoder_x.parameters()) + list(model.decoder_y.parameters())
optimizer = Adam([{'params': lstm_params, 'lr': lr1},
                  {'params': encoders_params, 'lr': lr2},
                  {'params': decoders_params, 'lr': lr3}])

print("Model and optimizer built! Train for {} epochs!".format(epochs))

print("Start loading data!")
now = time.time()
(snli_train_iter, snli_val_iter, snli_test_iter), \
(multinli_train_iter, multinli_match_iter, multinli_mis_match_iter),\
TEXT_FIELD, LABEL_FIELD, GENRE_FIELD \
    = NLIDataloader(multinli_path, snli_path, "glove.840B.300d").load_nlidata(batch_size=64,
                                                                                gpu_option=gpu, tokenizer='spacy')
elapsed = time.time() - now
print("Data loaded, time elapsed: {}".format(elapsed))

vocab_size = len(TEXT_FIELD.vocab)
print("Vocabulary size: {}".format(vocab_size))
num_class = len(LABEL_FIELD.vocab) # 3
num_genre = len(GENRE_FIELD.vocab) # 10


min_valid_loss = float("Inf")
patience = 20
for e in range(epochs):
    print("Starting epoch {}/{}...".format(e+1, epochs))
    multinli_match_iter.init_epoch()
    multinli_mis_match_iter.init_epoch()
    multinli_train_iter.init_epoch()
    snli_val_iter.init_epoch()
    snli_train_iter.init_epoch()
    snli_test_iter.init_epoch()
    num_samples = len(multinli_train_iter)

    epoch_loss = 0.0
    predictions_y = []
    predictions_g = []
    labels = []
    genres = []
    model.train()
    for batch_idx, batch in enumerate(multinli_match_iter):
        model.zero_grad()
        xp, _ = batch.premise
        xh, _ = batch.hypothesis
        y = batch.label
        try:
            g = batch.genre
        except AttributeError:
            g = None

        info = model(xp, xh, y, g)
        y_pred, g_pred = info[0:2]
        loss = ELBO(info, y, g, llm, glm)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        predictions_y.append(y_pred.cpu().data.numpy())
        labels.append(y.cpu().data.numpy())
        if g is not None:
            genres.append(g.cpu().data.numpy())
            predictions_g.append(g_pred.cpu().data.numpy())

        if (batch_idx+1) % 500 == 0:
            print("Batch {}/{} complete! Training loss {}".format(batch_idx+1, num_samples, epoch_loss/(batch_idx+1)))

        if np.isnan(epoch_loss).any():
            print("Training: NaN values happened, rebooting...\n\n")
            complete = False
            break

    if not complete: # encountered NaN values
        valid_acc_at_minimal_loss = 0
        f1_match, acc_score_match, f1_mismatch, acc_score_mismatch = 0., 0., 0., 0.
        break

    predictions_y = np.concatenate(predictions_y)
    predictions_g = np.concatenate(predictions_g)
    labels = np.concatenate(labels)
    genres = np.concatenate(genres)

    predictions_y = np.argmax(predictions_y, axis=1)
    label_acc_score = accuracy_score(labels, predictions_y)
    predictions_g = np.argmax(predictions_g, axis=1)
    genre_acc_score = accuracy_score(genres, predictions_g)
    print("Epoch {}/{} complete! Label accuracy: {}; genre accuracy: {}".format(e, epochs, label_acc_score, genre_acc_score))

    valid_loss = 0.0
    predictions_y = []
    predictions_g = []
    labels = []
    genres = []
    model.eval()
    for batch_idx, batch in enumerate(snli_val_iter):
        model.zero_grad()
        xp, _ = batch.premise
        xh, _ = batch.hypothesis
        y = batch.label
        try:
            g = batch.genre
        except AttributeError:
            g = None

        info = model(xp, xh, y, g)
        y_pred, g_pred = info[0:2]
        loss = ELBO(info, y, g, llm, glm)
        loss.backward()
        optimizer.step()
        valid_loss += loss.item()

        predictions_y.append(y_pred.cpu().data.numpy())
        labels.append(y.cpu().data.numpy())
        if g is not None:
            genres.append(g.cpu().data.numpy())
            predictions_g.append(g_pred.cpu().data.numpy())

    print("Validation loss: {}".format(valid_loss / len(snli_val_iter)))

    predictions_y = np.concatenate(predictions_y)
    # predictions_g = np.concatenate(predictions_g)
    labels = np.concatenate(labels)
    # genres = np.concatenate(genres)

    predictions_y = np.argmax(predictions_y, axis=1)
    label_acc_score = accuracy_score(labels, predictions_y)
    # predictions_g = np.argmax(predictions_g, axis=1)
    # genre_acc_score = accuracy_score(genres, predictions_g)
    print("Validation label accuracy: {}".format(label_acc_score))

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

if complete:
    best_model = torch.load(model_path)
    best_model.eval()
    predictions = []
    labels = []
    for _, batch in enumerate(snli_test_iter):
        model.zero_grad()

        premise, _premis_lens = batch.premise
        hypothesis, _hypothesis_lens = batch.hypothesis
        label = batch.label
        try:
            genre = batch.genre
        except AttributeError:
            genre = None

        output_y, _ = best_model.classify(premise, hypothesis)

        predictions.append(output_y.cpu().data.numpy())
        labels.append(label.cpu().data.numpy())

        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)

        f1, acc_score = evaluate(predictions, labels, LABEL_FIELD.vocab, "snli_cm.jpg")

        print("Test F1 on SNLI:", f1)
        print("Entailment Classification Acc on SNLI:", acc_score)

    # multinli
    predictions_match = []
    labels_match = []
    predictions_mismatch = []
    labels_mismatch = []
    for _, batch in enumerate(multinli_match_iter):
        premise, _premis_lens = batch.premise
        hypothesis, _hypothesis_lens = batch.hypothesis
        label = batch.label
        try:
            genre = batch.genre
        except AttributeError:
            genre = None

        output_y, _ = best_model.classify(premise, hypothesis)
        predictions.append(output_y.cpu().data.numpy())
        labels.append(label.cpu().data.numpy())

    for _, batch in enumerate(multinli_mis_match_iter):
        premise, _premis_lens = batch.premise
        hypothesis, _hypothesis_lens = batch.hypothesis
        label = batch.label
        try:
            genre = batch.genre
        except AttributeError:
            genre = None

        output_y, _ = best_model.classify(premise, hypothesis)
        predictions.append(output_y.cpu().data.numpy())
        labels.append(label.cpu().data.numpy())

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


