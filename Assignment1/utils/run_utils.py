import numpy as np
seed = 233
np.random.seed(seed)
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import chain


def combine_dataset(iter1, iter2):
    for iter_combine in chain(iter1, iter2):
        yield iter_combine

def evaluate(predictions, labels, label_dict, cm_path):
    num_class = predictions.shape[1]
    predictions = np.argmax(predictions, axis=1)

    acc_score = {}
    f1 = {}
    f1["all"] = f1_score(labels, predictions, average='weighted')
    acc_score["all"] = accuracy_score(labels, predictions)
    cm = confusion_matrix(labels, predictions)
    save_confusion_matrix(cm, num_class, label_dict, cm_path)

    for c in range(num_class):
        label = label_dict.itos[c]
        print(label)
        label_idx = np.where(labels == c)
        acc_score[label] = accuracy_score(labels[label_idx], predictions[label_idx])
        f1[label] = f1_score(labels[label_idx], predictions[label_idx], average='weighted')

    return f1, acc_score

def save_confusion_matrix(cm, num_classes, label_dict, cm_path, title='Confusion matrix', cmap=plt.cm.Blues):
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    classes = [d for d in label_dict.itos]
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig.savefig(cm_path)