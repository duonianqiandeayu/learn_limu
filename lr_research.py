import numpy as np
from numpy import pi
import matplotlib
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *


# 准备训练的数据集，num是训练数据集大小
num = 2000
x = np.linspace(start=0, stop=4 * pi, num=2000)  # step 是指分割成多个数据点
y = np.sin(x)
X = np.expand_dims(x, axis=1)
Y = y.reshape(num, -1)
dataset = TensorDataset(torch.tensor(X, dtype=torch.float), torch.tensor(Y, dtype=torch.float))

# 测试的数据集
x_tensor = torch.linspace(0, 4*pi, 100)
x_test = torch.unsqueeze(x_tensor, dim=1)
y_test = torch.sin(x_test)


# 根据训练数据画出预测图，和原图进行比较
def fig_train_pred(x_input, y_input, model, type):
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    y_hat = model(torch.tensor(X, dtype=torch.float))
    plt.plot(x_input, y_input, label="former data")
    plt.plot(x_input, y_hat.detach().numpy(), label="拟合之后的直线")
    plt.title("{}".format(type))

    plt.legend()
    plt.savefig(fname="result_{}.png".format(type))
    # plt.show()


# 学习率
LR_list = [0.001, 0.01, 0.1]
for learn_rate in LR_list:
    # 数据预处理
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
    # 选则网络模型, 损失函数以及优化器
    sin_fit = model_ReLU()
    loss = nn.MSELoss()     # 损失函数
    myOptim = optim.Adam(params=sin_fit.parameters(), lr=learn_rate)    # 优化器
    test_loss = []

    # 训练步骤开始
    writer = SummaryWriter(log_dir="log", comment="loss")
    for epoch in range(500):
        sum_loss = 0.0
        for x_train, y_train in dataloader:
            y_pred = sin_fit(x_train)

            result_loss = loss(y_train, y_pred)     # 计算损失
            myOptim.zero_grad()                     # 清除上一层的梯度数据
            result_loss.backward()                  # 反向传回参数，修改网络权重
            myOptim.step()                          # 优化器进行单步运算
            # 计算一轮运算的损失
            sum_loss += result_loss
        # if epoch > 100:
            writer.add_scalar("lr={}".format(learn_rate), sum_loss.item(), epoch)
        # if (epoch + 1) % 100 == 0:
        #     print("step:{}, loss:{}".format(epoch + 1, sum_loss.item()))
    writer.close()

    # 根据这一轮训练的模型，画出对比图形
    y_hat = sin_fit(torch.tensor(X, dtype=torch.float))     # 根据训练后模型，计算估计值

    plt.plot(x, y, label="raw")
    plt.plot(x, y_hat.detach().numpy(), label="fitted")
    plt.title("y=sin(x)")
    plt.savefig(fname="lr {}.png".format(learn_rate))
    plt.close()

    # 在测试集上进行验证
    y_test_preds = sin_fit(x_test)
    loss = ((y_test - y_test_preds) ** 2).sum()
    loss_average = loss / 100
    loss_average = (loss_average ** 0.5).item()
    test_loss.append(loss_average)
    print("lr={}, average test loss={}".format(learn_rate, loss_average))











