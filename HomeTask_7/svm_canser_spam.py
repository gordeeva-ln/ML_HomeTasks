import torch.nn as nn
import torch
import torch.nn.functional as F
import pandas as pd
from torchvision import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import time


def load_data(path):
    data = pd.read_csv(path)
    # normalize(data)
    data_y = data['label'].values
    data_x = data.drop(columns=['label']).values
    return data_y, data_x


def forest_svm(data_x, data_y):

    X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, train_size=0.8, test_size=0.2)

    start_forest = time.time()
    rf = RandomForestClassifier(n_estimators=100, criterion='entropy')

    rf.fit(X_train, y_train)
    time_forest = time.time() - start_forest

    predictions = rf.predict(X_test)

    errors = abs(predictions - y_test)
    accuracy = rf.score(X_test, y_test)
    print('Accuracy:', 100 * round(accuracy, 2), '%.')
    print("Forest time: {:.3f} sec".format(time_forest))

    start_svm = time.time()
    clf = svm.SVC(kernel='linear', gamma='auto')
    clf.fit(X_train, y_train)
    time_svm = time.time() - start_svm
    predictions = clf.predict(X_test)
    errors = abs(predictions - y_test)

    accuracy = clf.score(X_test, y_test)
    print('Accuracy:', 100 * round(accuracy, 2), '%.')
    print("SVM time: {:.3f} sec".format(time_svm))


# class Profiler(object):
#     def __enter__(self):
#         self._startTime = time.time()
#
#     def __exit__(self, type, value, traceback):
#         print("Forest time: {:.3f} sec".format(time.time() - self._startTime))


def main():
    cancer_y, cancer_x = load_data('data/cancer.csv')

    cancer_x = np.array([i / np.max(cancer_x) for i in cancer_x])
    cancer_y = np.array([-1 if i == 'M' else 1 for i in cancer_y])

    forest_svm(cancer_x, cancer_y)

    spam_y, spam_x = load_data('data/spam.csv')

    forest_svm(spam_x, spam_y)


if __name__ == '__main__':
    # with Profiler() as p:
    main()

# cancer
# Accuracy: 94.0 %.
# Forest time: 0.103 sec
# Accuracy: 89.0 %.
# SVM time: 0.004 sec

# spam
# Accuracy: 94.0 %.
# Forest time: 0.370 sec
# Accuracy: 94.0 %.
# SVM time: 202.879 sec

# cancer
# Accuracy: 95.0 %.
# Forest time: 0.103 sec
# Accuracy: 88.0 %.
# SVM time: 0.003 sec

# spam
# Accuracy: 94.0 %.
# Forest time: 0.500 sec
# Accuracy: 92.0 %.
# SVM time: 442.464 sec
