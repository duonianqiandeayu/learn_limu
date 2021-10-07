from torch import nn
from torch.nn import init
from torch.nn.modules import loss
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

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)

loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.1)


def get_dataloader_workers():
    return 4


def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))

    mnist_train = torchvision.datasets.FashionMNIST(root="E:/code/old/data",
                                                    train=True, transform=trans, download=False)
    mnist_test = torchvision.datasets.FashionMNIST(root="E:/code/old/data",
                                                   train=False, transform=trans, download=False)
    print(len(mnist_train))
    print(len(mnist_test))

    return (data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True, num_workers=get_dataloader_workers()))


batch_size = 256

train_iter, test_iter = load_data_fashion_mnist(batch_size)


num_epochs = 10
num_inputs = 784
num_outputs = 10
class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear=nn.Linear(num_inputs, num_outputs)
    def forward(self, x):
        y=self.linear(x.views(x.shape[0], -1))
        return y

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)

net = LinearNet(num_inputs, num_outputs)



