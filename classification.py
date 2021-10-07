from torch import nn
from torch.utils import data
import numpy
import torch
from torch import nn
from torch.utils import data
import numpy
import torch
from d2l import torch as d2l
from torchvision import transforms
import torchvision


d2l.use_svg_display()

def get_dataloader_workers():
    return 4

def load_data_fashion_mnist(batch_size, resize= None):

    trans=[transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))

    mnist_train = torchvision.datasets.FashionMNIST(root="../data",
        train=True, transform=trans,download=False)
    mnist_test = torchvision.datasets.FashionMNIST(root="../data",
        train=False, transform=trans,download=False)
    print(len(mnist_train))
    print(len(mnist_test))
    
    return (data.DataLoader(mnist_train, batch_size=batch_size ,shuffle=True, num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size=batch_size ,shuffle=True, num_workers=get_dataloader_workers()))


# train_iter = data.DataLoader(mnist_train, batch_size=batch_size ,shuffle=True, num_workers=get_dataloader_workers())

# timer=d2l.Timer()
# for X,y in train_iter:
#     continue
# print(f'{timer.stop():.2f}')

batch_size = 256

train_iter, test_iter = load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10
w = torch.normal(0, 0.01, size= (num_inputs,num_outputs), requires_grad=True)
b = torch.zeros(0, requires_grad=True)

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp/partition

# X= torch.normal(0,1 , (2,5))
# xprob= softmax(X)
# print(xprob, xprob.sum(1,keepdim=True))

def net(X):
    return softmax(torch.matmul(X.reshape((-1, w.shape[0])), w) + b)

def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)),y])

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] >1 :
        y_hat=y_hat.argmax(axis=1)
    
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

# accuracy(y_hat, y) / len(y)

def evaluate_accuracy(net,data_iter):
    '''计算在指定数据集上的模型的精度'''
    if isinstance(net, torch.nn.Module):
        net.eval() #设置维评估模式
    metric = d2l.Accumulator(2) # 正确预测数、预测总数  metric：度量
    for X, y in data_iter:
        metric.add(accuracy(net(X),y), y.numel())
    return metric[0] / metric[1]   #正确样本数 / 总样本数

def train_epoch_ch3(net, train_iter, loss, updater):
    '''softmax回归训练'''
    if isinstance(net , torch.nn.Module):
        net.train()
    metric=d2l.Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(
                float(1) * len(y), accuracy(y_hat=y_hat,y=y), y.size().numel())
        else:
            l.sum().backward
            updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat,y))
    return metric[0] / metric[2] , metric[1] / metric [2]

