import torch.nn as nn
import torch
import torch.nn.functional as F
import pandas as pd
from torchvision import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
from PIL import Image
from os import listdir


def load_data(path):
    data = pd.read_csv(path)
    data_y = data['label'].values
    data_x = data.drop(columns=['label']).values
    return data_y, data_x


def convert_to_tensor(X):
    data = np.zeros((len(X), 1, 28, 28))
    j = 0
    for x in X:
        matrix = np.zeros((1, 28, 28))
        for i in range(28 - 1):
            matrix[0][i] = x[i * 28: (i + 1) * 28]
        data[j] = torch.from_numpy(matrix)
        j += 1
    return data


def main():
    # data loader
    y, X = load_data('data/mnist.csv')
    n = y.shape[0]
    mnist = convert_to_tensor(X)  # array with tensors 28 x 28
    mnist_y = y  # array with 10000 labels
    data = [(torch.from_numpy(mnist[i]), mnist_y[i]) for i in range(n)]
    loader = DataLoader(dataset=data, batch_size=200)
    #plt.imshow(data[1][0][0])
    #plt.show()

    x = []
    for simbol in listdir("data/notMNIST_small"):
        for image in listdir("data/notMNIST_small/" + simbol):
            # bad images
            if (image != "RGVtb2NyYXRpY2FCb2xkT2xkc3R5bGUgQm9sZC50dGY=.png"
                    and image != "Q3Jvc3NvdmVyIEJvbGRPYmxpcXVlLnR0Zg==.png"):
                with Image.open("data/notMNIST_small/" + simbol + "/" + image) as f:
                    x.append(np.array(f.getdata()))
    x = np.array(x)
    plt.imshow(x[0].reshape(28, 28))
    plt.show()


if __name__ == '__main__':
    main()