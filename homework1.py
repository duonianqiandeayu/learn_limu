import torch
from torch import nn
import numpy as np


class Homework1Net(nn.Module):
    def __init__(self):
        super(Homework1Net, self).__init__()
        self.hidden1 = nn.Linear(1, 128)
        self.output = nn.Linear(128, 1)

    def forward(self, X):
        y_pred = self.output(nn.ReLU(self.hidden1(X)))
        return y_pred


def data_load():
    x_tensor = torch.linspace(0, 4*np.pi, 10000)
    x_data = torch.unsqueeze(x_tensor, dim=1)
    y_data = torch.sin(x_data)
    return x_data, y_data

def evaluate_accuracy(net, X, y):
    if isinstance(net, torch.nn.Module):
        net.eval()




if __name__ == "__mian__":
    lr, num_epochs,batch_size = 0.05, 5, 1000
    x_data, y_data = data_load()
    device = torch.device("cuda")
    net = Homework1Net()
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

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
        evaluate_accuracy(net, )
