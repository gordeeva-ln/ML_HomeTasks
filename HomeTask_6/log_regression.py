import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from sklearn.model_selection import train_test_split
import random


def load_data(path):
    data = pd.read_csv(path)
    data_y = data['label'].values
    data_x = data.drop(columns=['label']).values
    return data_y, data_x


def split_data(data_x, data_y):
    # train_x, test_x, train_y, test_y
    return train_test_split(data_x, data_y, train_size=0.8, test_size=0.2)


def loss(y, x, w):
    sum = 0
    n = y.shape[0]
    #print(y.shape)
    #print(x.shape)
    #print(w.shape)
    for i in range(n):
        # print(w[0])
        arg1 = y[i] * w
        # print(arg1.shape)
        # print(x[i].shape)
        arg = np.dot(np.transpose(arg1), x[i])
        # print(arg)
        sum += np.log(1 + np.exp(-arg))
        # print(1 + np.exp(-arg))
    return sum / n


def logist(batch_size, epoh_count, xs_train, y_train, xs_test, y_test):
    w0 = np.random.random(xs_train.shape[1])
    # print(w0.shape)
    nu = 20
    n_tr = xs_train.shape[0]
    n_test = xs_test.shape[0]
    w = w0
    loss_test = 1
    loss_prev_test = 2
    acc = []
    i = 0
    w_new = w

    while loss_test < loss_prev_test:  # and loss_prev_test - loss_test > pow(10, -10):
        b = 1  # number of batch
        index = list(range(n_tr))
        random.shuffle(index)
        for i in range(n_tr):

            arg = np.dot(y_train[i] * np.transpose(w), xs_train[i])
            w_new += nu * y_train[i] * xs_train[i] / (1 + np.exp(arg))
            if i == b * batch_size:
                w = w_new
                b += 1
                loss_prev_test = loss_test
                loss_test = loss(y_test, xs_test, w)
                acc.append(loss_test)
                print("Batch ", b, " have loss ", loss_test)
    return np.array(acc)


def main():
    batch_size = 270
    epoh_count = 5
    # load spam
    spam_y, spam_x = load_data('data/spam.csv')

    # load cancer
    cancer_y, cancer_x = load_data('data/cancer.csv')

    cancer_x = np.array([i / np.max(cancer_x) for i in cancer_x])
    cancer_y = np.array([0 if i == 'M' else 1 for i in cancer_y])

    # split spam
    spam_train_x, spam_test_x, spam_train_y, spam_test_y = split_data(spam_x, spam_y)

    # split cancer
    cancer_train_x, cancer_test_x, cancer_train_y, cancer_test_y = split_data(cancer_x, cancer_y)

    # build graphic for cancer
    # graphic = logist(batch_size, epoh_count, cancer_train_x, cancer_train_y, cancer_test_x, cancer_test_y)
    # plt.plot(range(0, graphic.shape[0] - 1), graphic[1:])
    # print(graphic)
    # plt.show()

    # build graphic for spam
    graphic = logist(batch_size, epoh_count, spam_train_x, spam_train_y, spam_test_x, spam_test_y)
    plt.plot(range(0, graphic.shape[0] - 1), graphic[1:])
    plt.show()


if __name__ == '__main__':
    main()
