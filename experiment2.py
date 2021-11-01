import os.path

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
# import torch.utils.tensorboard
# import tensorboardX
# from tensorboardX import SummaryWriter
from PIL import Image
from torchvision.datasets import ImageFolder


# class MyData(torch.utils.data.Dataset):
#     def __init__(self, root_dir, label_dir):
#         self.root_dir = root_dir
#         self.label_dir = label_dir
#         self.path = os.path.join(self.root_dir,self.label_dir)
#         self.img_path = os.listdir(self.path)
#
#     def __getitem__(self, idx):
#         img_name = self.img_path[idx]
#         img = os.path.join(self.path,img_name)
#         im =Image.open(img)
#         label = self.label_dir
#         return im, label
#
#     def __len__(self):
#         return len(self.img_path)R


if __name__ == "__main__":

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),  # 随机裁剪到224x224大小
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(degrees=15),  # 随机旋转
        transforms.ToTensor(),  # 转化成Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正则化
    ])
    train_dir ="/Users/yanzhaoyu/Downloads/ImageData2/train"
    tran_dataset = ImageFolder(root=train_dir, transform=train_transform)
    print(len(dataset))

    # val_transform = transforms.Compose
