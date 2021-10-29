import torch
from torch import nn
from torch.nn import Linear, ReLU, Sigmoid, LeakyReLU


# 神经网络模型
class model_ReLU(nn.Module):
    def __init__(self):
        super(model_ReLU, self).__init__()
        self.mod_relu = nn.Sequential(
            Linear(1, 10), ReLU(),
            Linear(10, 100), ReLU(),
            Linear(100, 10), ReLU(),
            Linear(10, 1))

    def forward(self, input):
        return self.mod_relu(input)


class model_Sigmoid(nn.Module):
    def __init__(self):
        super(model_Sigmoid, self).__init__()
        self.model_Sigmoid = nn.Sequential(
            Linear(1, 10), Sigmoid(),
            Linear(10, 100), Sigmoid(),
            Linear(100, 10), Sigmoid(),
            Linear(10, 1))

    def forward(self, input):
        return self.model_Sigmoid(input)


class model_LKReLU(nn.Module):
    def __init__(self):
        super(model_LKReLU, self).__init__()
        self.model_LKReLU = nn.Sequential(
            Linear(1, 10), LeakyReLU(),
            Linear(10, 100), LeakyReLU(),
            Linear(100, 10), LeakyReLU(),
            Linear(10, 1))

    def forward(self, input):
        return self.model_LKReLU(input)


# 研究网络深度
class model_depth1(nn.Module):
    def __init__(self):
        super(model_depth1, self).__init__()
        self.model_depth1 = nn.Sequential(
            Linear(1, 10), ReLU(),
            Linear(10, 100), ReLU(),
            Linear(100, 10), ReLU(),
            Linear(10, 1))

    def forward(self, input):
        return self.model_depth1(input)


class model_depth2(nn.Module):
    def __init__(self):
        super(model_depth2, self).__init__()
        self.model_depth2 = nn.Sequential(
            Linear(1, 10), ReLU(),
            Linear(10, 100), ReLU(),
            Linear(100, 1000), ReLU(),
            Linear(1000, 100), ReLU(),
            Linear(100, 10), ReLU(),
            Linear(10, 1)
        )

    def forward(self, input):
        return self.model_depth2(input)


# 研究网络宽度
class model_width1(nn.Module):
    def __init__(self):
        super(model_width1, self).__init__()
        self.model = nn.Sequential(
            Linear(1, 10), ReLU(),
            Linear(10, 100), ReLU(),
            Linear(100, 10), ReLU(),
            Linear(10, 1))

    def forward(self, input, width):
        if width == 0:
            return self.model(input)


class model_width2(nn.Module):
    def __init__(self):
        super(model_width2, self).__init__()
        self.model = nn.Sequential(
            Linear(1, 40), ReLU(),
            Linear(40, 400), ReLU(),
            Linear(400, 40), ReLU(),
            Linear(40, 1))

    def forward(self, input, width):
        if width == 0:
            return self.model(input)


# torch.save(sin_fit, "motivate_ReLU.pth")
# motivate_relu = torch.load("motivate_ReLU.pth")


# 准备500个测试数据
# x_tensor = torch.linspace(0, 4*pi, 100)
# x_test = torch.unsqueeze(x_tensor, dim=1)
# y_test = torch.sin(x_test)
# y_test_preds = sin_fit(x_test)
#
# loss = ((y_test - y_test_preds)**2).sum()
# loss_average = loss / 100
# loss_average = (loss_average**0.5).item()

# model_list = [model_ReLU, model_Sigmoid, model_LKReLU, model_depth1, model_depth2, model_width1, model_width2]
# for model in model_list:
#     print(model)
