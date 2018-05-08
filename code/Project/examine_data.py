import sys
sys.path.append("..")
from model import GenreAgnosticInference, GAIA, ELBO # pylint: disable=E0611, C0413
from utils import NLIDataloader # pylint: disable=E0611, C0413
from utils import evaluate, combine_dataset, load_param # pylint: disable=E0611, C0413

multinli_path = '../../data/multinli_1.0/'
snli_path = '../../data/snli_1.0/'

(snli_train_iter, snli_val_iter, snli_test_iter), \
(multinli_train_iter, multinli_match_iter, multinli_mis_match_iter),\
TEXT_FIELD, LABEL_FIELD, GENRE_FIELD \
    = NLIDataloader(multinli_path, snli_path, "glove.840B.300d").load_nlidata(batch_size=4,
                                                                                gpu_option=-1, tokenizer='spacy')
vocab_size = len(TEXT_FIELD.vocab)
num_class = len(LABEL_FIELD.vocab)
num_genre = len(GENRE_FIELD.vocab)


model = GAIA(300, num_class, num_genre, 25,
                            50, 'lstm', vocab_size, 300,
                            300, 1.0, 1.0)

for batch in multinli_train_iter:
    xp, _ = batch.premise
    xh, _ = batch.hypothesis
    y = batch.label
    g = batch.genre

    info = model(xp, xh, y, g)
    loss = ELBO(info, y, g, 1.0, 1.0)
