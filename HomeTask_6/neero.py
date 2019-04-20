import torch.nn as nn
import torch
import torch.nn.functional as F
import pandas as pd
from torchvision import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5, 1)
        self.conv2 = nn.Conv2d(8, 64, 5, 1, 2)
        self.conv3 = nn.Conv2d(64, 512, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 512, 64)
        self.fc2 = nn.Linear(64, 10)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        #x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 512)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(model, train_loader, device, optimizer, epoch):
    model.train()
    log_interval = 10

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        #print(train_loader.dataset[1][0].shape, train_loader.dataset[1][1])
        #print(data.shape, target, batch_idx)
        # print(batch_idx, data)
        if batch_idx != 0:
            #data, target = torch.Tensor(data), torch.Tensor(target)
            optimizer.zero_grad()
            #print(data.shape)
            output = model(data)
            # print(output.shape)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data.type ('torch.FloatTensor'))
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # params
    learning_rate = 0.001
    epochs_count = 10
    batch_size = 64
    total_step = batch_size * epochs_count

    # settings
    SEED = 1
    cuda = torch.cuda.is_available()
    # For reproducibility
    torch.manual_seed(SEED)

    if cuda:
        torch.cuda.manual_seed(SEED)

    device = torch.device("cuda" if cuda else "cpu")

    # data loader
    y, X = load_data('data/mnist.csv')

    mnist_train_x, mnist_test_x, mnist_train_y, mnist_test_y = train_test_split(X, y)

    n_train = mnist_train_y.shape[0]
    n_test = mnist_test_y.shape[0]
    mnist_train_x = convert_to_tensor(mnist_train_x)
    mnist_test_x = convert_to_tensor(mnist_test_x)

    # data_train = torch.utils.data.TensorDataset(torch.from_numpy(mnist_train_x).type ('torch.FloatTensor'),
    #                                             torch.from_numpy(mnist_train_y))
    # data_test = torch.utils.data.TensorDataset(torch.from_numpy(mnist_test_x).type ('torch.FloatTensor'),
    #                                            torch.from_numpy(mnist_test_y))
    data_train = [(torch.from_numpy(mnist_train_x[i]).type ('torch.FloatTensor'),
                   mnist_train_y[i]) for i in range(n_train)]
    data_test = [(torch.from_numpy(mnist_test_x[i]).type ('torch.FloatTensor'),
                  mnist_test_y[i]) for i in range(n_test)]

    labels_count = 10

    train_loader = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=data_test, batch_size=batch_size, shuffle=False)

    # print(e.shape)

    model = Net().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    for epoch in range(1, epochs_count + 1):
        train(model, train_loader, device, optimizer, epoch)
        test(model, device, test_loader)


if __name__ == '__main__':
    main()
