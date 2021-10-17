import torch
from torch import nn
from myd2l import reshape
from myd2l import evaluat_accuracy_gpu
from myd2l import load_data_fashion_mnist
from myd2l import train_ch6_2



net = torch.nn.Sequential(
    reshape(), nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2,stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10)
)

# X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
# for layer in net:
#     X = layer(X)
#     print(layer.__class__.__name__, 'output shape:\t', X.shape)
batch_size = 512
train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)


device = torch.device("cuda")
lr, num_epochs = 0.9, 20
print(device)
train_ch6_2(net, train_iter,test_iter,num_epochs,lr,device)

