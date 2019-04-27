import torch.nn as nn
import torch
import torch.nn.functional as F
import pandas as pd
from torchvision import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from sklearn import svm


def svm_kernel(y, X, kernel, num):
    if kernel == 'poly2':
        clf = svm.SVC(kernel='poly', degree=2, gamma='auto')
    elif kernel == 'poly3':
        clf = svm.SVC(kernel='poly', degree=3, gamma='auto')
    elif kernel == 'poly5':
        clf = svm.SVC(kernel='poly', degree=5, gamma='auto')
    else:
        clf = svm.SVC(kernel=kernel, gamma='auto')
    clf.fit(X, y)

    plt.figure(num)
    plt.clf()

    plt.scatter(X[:, 0], X[:, 1], s=80, facecolors='none',
                zorder=10, edgecolor='k')

    plt.axis('tight')
    edge = 5

    XX, YY = np.mgrid[X[:, 0].min() - edge:X[:, 0].max() + edge:200j, X[:, 1].min() - edge:X[:, 1].max() + edge:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Accent)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'],
                linestyles=['--', '-', '--'], levels=[-.5, 0, .5])
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], color='blue')
    plt.title(kernel)
    return plt


def main():
    data = pd.read_csv('data/blobs2.csv')
    y = data['label'].values
    X = data.drop(columns=['label']).values

    for num, kernel in enumerate(('linear', 'rbf', 'poly2', 'poly3', 'poly5')):
        plt = svm_kernel(y, X, kernel, num)

    plt.show()


if __name__ == '__main__':
    main()
