import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
import matplotlib.pyplot as plt
import torchvision


class Homework1Net_Relu(nn.Module):
    def __init__(self):
        super(Homework1Net_Relu, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20), nn.ReLU(),
            nn.Linear(20, 100), nn.ReLU(),
            # nn.Linear(100, 100), nn.ReLU(),
            nn.Linear(100, 20), nn.ReLU(),
            nn.Linear(20, 1)
        )

    def forward(self, X):
        return self.net(X)


class Homework1Net_Sigmoid(nn.Module):
    def __init__(self):
        super(Homework1Net_Sigmoid, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20), nn.Sigmoid(),
            nn.Linear(20, 128), nn.Sigmoid(),
            # nn.Linear(128, 128), nn.Sigmoid(),
            nn.Linear(128, 20), nn.Sigmoid(),
            nn.Linear(20, 10), nn.Sigmoid(),
            nn.Linear(10, 1)
        )

    def forward(self, X):
        return self.net(X)


def data_load(num):
    x_tensor = torch.linspace(0, 4 * np.pi, num)
    x_data = torch.unsqueeze(x_tensor, dim=1)
    y_data = torch.sin(x_data)
    return x_data, y_data


class mydata(torch.utils.data.Dataset):
    def __init__(self, x_data, y_data):
        self.X, self.y = x_data, y_data

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return len(self.X.data.size())


def evaluate_accuracy(net_eva, test_data):
    if isinstance(net_eva, torch.nn.Module):
        net_eva.eval()
    x, y_eva = data_load(1000)
    y_hat_eva = net_eva(x)
    cmp = ((y_hat_eva - y_eva) ** 2).sum()
    accuracy = cmp / 1000
    return (accuracy ** 0.5).item()


if __name__ == "__main__":
    lr, num_epochs, batch_size = 0.05, 10, 500
    x_data, y_data = data_load(60000)
    train_db = mydata(x_data, y_data)
    # train_db, test_db = torch.utils.data.random_split(dataset, [50000, 10000])
    x_data, y_data = data_load(10000)
    test_db = mydata(x_data, y_data)
    print('db1: ', len(train_db), ' db2: ', len(test_db))
    train_iter = torch.utils.data.DataLoader(train_db, batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test_db, batch_size=batch_size, shuffle=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(device)
    # net = Homework1Net_Relu()

    net = Homework1Net_Relu()
    net.to(device)
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    plt.figure("regression")
    for epoch in range(num_epochs):
        net.train()
        for X, y in train_iter:
            print(X.size(),y.size())
            optimizer.zero_grad()
            # X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            # with torch.no_grad():
        # if (epoch + 1) % 100 == 0:
        print('epoch: ', epoch + 1, 'train loss: ', l.item())
        # print('test loss: ', evaluate_accuracy(net))
