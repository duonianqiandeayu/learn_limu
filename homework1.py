import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
import torchvision
import tensorboardX
import matplotlib.pyplot as plt
import torch.utils.tensorboard
from torch.utils.tensorboard import SummaryWriter


class Homework1Net_Relu(nn.Module):
    def __init__(self):
        super(Homework1Net_Relu, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20), nn.ReLU(),
            nn.Linear(20, 128), nn.ReLU(),
            # nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 20), nn.ReLU(),
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
            nn.Linear(20, 1), nn.Sigmoid(),
            # nn.Linear(10, 1)
        )

    def forward(self, X):
        return self.net(X)


def data_load(num):
    x_tensor = torch.linspace(0, 4 * np.pi, num)
    x = torch.unsqueeze(x_tensor, dim=1)
    y = torch.sin(x)
    return x, y


class MyData(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.X, self.y = x, y

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return len(self.X.data.size())


def evaluate_accuracy(net_eva, test_iter, item_num):
    if isinstance(net_eva, torch.nn.Module):
        net_eva.eval()
    for X, y in test_iter:
        X, y = X.to(device), y.to(device)
        y_hat_eva = net_eva(X)
        cmp = ((y_hat_eva - y) ** 2).sum()
        mse = cmp / item_num
    return (mse ** 0.5).item()


if __name__ == "__main__":
    writer = SummaryWriter('./path/to/log')

    lr, num_epochs, batch_size = 0.05, 150, 2000
    x_data, y_data = data_load(60000)
    train_db = MyData(x_data, y_data)
    x_data, y_data = data_load(10000)
    test_db = MyData(x_data, y_data)

    print('db1: ', len(train_db), ' db2: ', len(test_db))
    train_iter = torch.utils.data.DataLoader(train_db, batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test_db, batch_size=batch_size, shuffle=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(device)

    net = Homework1Net_Sigmoid()
    # net = Homework1Net_Relu()

    net.to(device)
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(num_epochs):
        net.train()
        for X, y in train_iter:
            X, y =X.to(device), y.to(device)
            # print(X.size(), y.size())
            optimizer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            # with torch.no_grad():

        # if (epoch + 1) % 100 == 0:
            print('epoch: ', epoch + 1, 'train loss: ', l.item())
        writer.add_scalar('loss2—1/train_loss', l.item(), epoch+1)

        test_loss = evaluate_accuracy(net, test_iter, 10000)
        print('test loss: ', test_loss)
        writer.add_scalar('loss2-1/test_loss', test_loss, epoch+1)
        with torch.no_grad():
            x, y = data_load(100)
            x, y =x.to(device), y.to(device)
            y_hat = net(x)
            plt.plot(x.data.cpu().numpy(), y.data.cpu().numpy(), label='label')
            plt.plot(x.data.cpu().numpy(), y_hat.data.cpu().numpy(), label='predict')
            plt.title('sin func')
            plt.xlabel('x')
            plt.ylabel('sin(x)')
            plt.legend()
            plt.savefig(fname='fit.png')
            plt.show()
    # net.eval()
    # x, y = data_load(100)
    # x = x.to(device)
    # with torch.no_grad():
    #     predict = net(x)
    #     plt.plot(x.data.cpu().numpy(), y.data.cpu().numpy(), label='label')
    #     plt.plot(x.data.cpu().numpy(), predict.data.cpu().numpy(), label='predict')
    #     plt.title('sin func')
    #     plt.xlabel('x')
    #     plt.ylabel('sin(x)')
    #     plt.legend()
    #     plt.savefig(fname='fit.png')
    #     plt.show()