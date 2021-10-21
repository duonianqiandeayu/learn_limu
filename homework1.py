import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
import matplotlib.pyplot as plt


class Homework1Net(nn.Module):
    def __init__(self):
        super(Homework1Net, self).__init__()
        self.net = nn.Sequential(
         nn.Linear(1, 20), nn.ReLU(),
         nn.Linear(20, 100), nn.ReLU(),
         nn.Linear(100, 20), nn.ReLU(),
         nn.Linear(20, 1)
        )

    def forward(self, X):
        return self.net(X)


def data_load(num):
    x_tensor = torch.linspace(0, 4*np.pi, num)
    x_data = torch.unsqueeze(x_tensor, dim=1)
    y_data = torch.sin(x_data)
    return x_data, y_data


def evaluate_accuracy(net_eva):
    if isinstance(net_eva, torch.nn.Module):
        net_eva.eval()
    x, y_eva = data_load(1000)
    y_hat_eva = net_eva(x)
    cmp = ((y_hat_eva - y_eva)**2).sum()
    accuracy = cmp/1000
    return accuracy ** 0.5


if __name__ == "__main__":
    lr, num_epochs, batch_size = 0.05, 500, 1000
    x_data, y_data = data_load(10000)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(device)
    net = Homework1Net()
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    plt.figure("regression")
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        X = x_data.to(device)
        y = y_data.to(device)
        optimizer.zero_grad()
        y_hat = net(X)
        l = loss(y_hat, y)
        l.backward()
        optimizer.step()
        # with torch.no_grad():
        if (epoch + 1) % 100 == 0:
            print('epoch: ', epoch+1, ' loss: ', l.item())
            print(evaluate_accuracy(net))
