import numpy as np
seed = 233
np.random.seed(seed)
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import chain


def combine_dataset(iter1, iter2):
    """
    combine two iters
    :param iter1:
    :param iter2:
    :return:
    """
    for iter_combine in chain(iter1, iter2):
        yield iter_combine

def evaluate(predictions, labels, label_dict, cm_path):
    """
    Evaluate the test results, return f1 & acc score, save the confusion matrix figure
    :param predictions:
    :param labels:
    :param label_dict: {index: label_str, ...}
    :param cm_path: path for saving the confusion matrix figure
    :return: f1 & acc score
    """
    num_class = predictions.shape[1]
    predictions = np.argmax(predictions, axis=1)

    acc_score = {}
    f1 = {}
    f1["all"] = f1_score(labels, predictions, average='weighted')
    acc_score["all"] = accuracy_score(labels, predictions)
    cm = confusion_matrix(labels, predictions)
    save_confusion_matrix(cm, num_class, label_dict, cm_path)


    accs = cm.diagonal()/ cm.sum(axis=1).astype(float)
    print(accs)

    for c in range(num_class):
        label = label_dict.itos[c]
        acc_score[label] = accs[c]

    return f1, acc_score

def save_confusion_matrix(cm, num_classes, label_dict, cm_path, cmap=plt.cm.Blues):
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    classes = [d for d in label_dict.itos]
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig.savefig(cm_path)