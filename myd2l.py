import torch
import torchvision
from torch import nn
from torch.utils import data
from torchvision.transforms import transforms
from d2l import torch as d2l


class reshape(torch.nn.Module):
    def forward(self,x):
        return x.view(-1, 1, 28, 28)


def evaluat_accuracy_gpu(net, data_iter, device=None):
    '''使用GPU计算模型在数据集上得精度'''
    if isinstance(net, torch.nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        if isinstance(X, list):
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0]/metric[1]


def get_dataloader_workers():
    """Use 4 processes to read the data."""
    return 0


def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="../data",
                                                    train=True, transform=trans, download=False)
    mnist_test = torchvision.datasets.FashionMNIST(root="../data",
                                                   train=False, transform=trans, download=False)
    print(len(mnist_train))
    print(len(mnist_test))

    return (data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True, num_workers=get_dataloader_workers()))


# def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
#     def init_weights(m):
#         if type(m) == nn.Linear or type(m) == nn.Conv2d:
#             nn.init.xavier_uniform_(m.weight)
#     net.apply(init_weights)
#     print('training on ',device)
#     net.to(device)
#     optimizer = torch.optim.SGD(net.parameters(), lr=lr)
#     loss = nn.CrossEntropyLoss()
#     animator = d2l.Accumulator(xlabel='epoch', xlim = [1, num_epochs],
#                                legend=['train loss', 'train acc', 'test acc'])
#     timer, num_batches = d2l.Timer(), len(train_iter)
#     for epoch in range(num_epochs):
#         metric = d2l.Accumulator(3)
#         net.train()
#         for i, (X, y) in enumerate(train_iter):
#             timer.start()
#             optimizer.zero_grad()
#             X, y = X.to(device), y.to(device)
#             y_hat = net(X)
#             l = loss(y_hat, y)
#             l.backward()
#             optimizer.step()
#             with torch.no_grad():
#                 metric.add(l * X.shape[0], d2l.accuracy(y_hat,y), X.shape[0])
#             timer.stop()
#             train_l = metric[0] / metric[2]
#             train_acc = metric[1] / metric[2]
#             if (i+1) % (num_batches // 5) == 0 or i == num_batches -1:
#                 animator.add((epoch + (i + 1)) / num_batches, (train_l, train_acc, None))
#         test_acc = evaluat_accuracy_gpu(net, test_iter)
#         animator.add((epoch + 1 , (None, None, test_acc)))
#         print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
#           f'test acc {test_acc:.3f}')
#         print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
#           f'on {str(device)}')

def train_ch6_2(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on ', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat,y), X.shape[0])

            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i+1) % (num_batches // 5) == 0 or i == num_batches -1:
                animator.add((epoch + (i + 1)) / num_batches, (train_l, train_acc, None))
        test_acc = evaluat_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
        print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
        print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
