import os
import pickle
import struct
from PIL import Image
import sys
import json
import torch
import time
import torch.nn as nn
from torchvision import transforms, datasets, utils
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import torchvision.models as models

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def forward_hook(module, input, output):
    flag = "input"
    shape = None
    data = None
    if (flag == "input"):
        shape = input[0].shape
        data = input[0].tolist()
    else:
        shape = output.shape
        data = output.tolist()
    print(shape)
    data_f = []
    with open("./backup/in_py", "wb") as fp:
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    data_f += data[i][j][k]
        for i in range(len(data_f)):
            fp.write(struct.pack('f', data_f[i]))
        fp.close()

# grad_input 中各部分梯度的‌顺序取决于模块 forward 函数的输入顺序‌：
# ‌nn.Linear‌（有 bias）：
# grad_input 顺序：[bias_grad, input_grad, weight_grad]
# ‌nn.Conv2d‌（有 bias）：
# grad_input 顺序：[input_grad, weight_grad, bias_grad]
# ‌nn.ReLU‌（无参数）：
# grad_input 仅包含：[input_grad]
def backward_hook(module, grad_input, grad_output):
    # print("gradshape:{}".format(grad_input[0].shape))
    input_grad = grad_input[1]
    shape = input_grad.shape
    print(shape)
    grad = input_grad.tolist()
    # print(grad)
    data = []
    with open("./backup/grad_py", "wb") as fp:
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    data += grad[i][j][k]
        print(len(data))
        for i in range(len(data)):
            fp.write(struct.pack('f', data[i]))
        fp.close()

class MyDataset(Dataset):
    def __init__(self, txt_path, transform = None, target_transform = None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1]))) # 类别转为整型int
            self.imgs = imgs 
            self.transform = transform
            self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB') 
        if self.transform is not None:
            img = self.transform(img) 
        return img, label

    def __len__(self):
        return len(self.imgs)

class CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR, self).__init__()

        self.l1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
        self.l2 = nn.BatchNorm2d(128)
        self.l3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.l4 = nn.BatchNorm2d(128)
        self.l5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.l6 = nn.BatchNorm2d(128)
        self.l7 = nn.MaxPool2d(2, 2, 0)
        
        self.l8 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.l9 = nn.BatchNorm2d(256)
        self.l10 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.l11 = nn.BatchNorm2d(256)
        self.l12 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.l13 = nn.BatchNorm2d(256)
        self.l14 = nn.MaxPool2d(2, 2, 0)
        
        self.l15 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.l16 = nn.BatchNorm2d(512)
        self.l17 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.l18 = nn.BatchNorm2d(512)
        self.l19 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.l20 = nn.BatchNorm2d(512)
        
        self.l21 = nn.Conv2d(512, 10, kernel_size=1, stride=1, padding=1)
        self.l22 = nn.AvgPool2d(9, 9, 0)
        self.l23 = nn.CrossEntropyLoss()

    def forward(self, x, labels):
        x = self.l1(x)
        x = torch.relu(self.l2(x))
        x = self.l3(x)
        x = torch.relu(self.l4(x))
        x = self.l5(x)
        x = torch.relu(self.l6(x))
        x = self.l7(x)
        x = self.l8(x)
        x = torch.relu(self.l9(x))
        x = self.l10(x)
        x = torch.relu(self.l11(x))
        x = self.l12(x)
        x = torch.relu(self.l13(x))
        x = self.l14(x)
        x = self.l15(x)
        x = torch.relu(self.l16(x))
        x = self.l17(x)
        x = torch.relu(self.l18(x))
        x = self.l19(x)
        x = torch.relu(self.l20(x))
        x = torch.relu(self.l21(x))
        x = self.l22(x)
        x = torch.flatten(x, 1)
        x = self.l23(x, labels)
        return x

data_transform = transforms.Compose([
    transforms.Resize((28,28)),  # 随机裁剪成224×224
    transforms.ToTensor(),  # 转换成Tensor（PyTorch能处理的数据格式）
    # 标准化：让数据更符合模型训练要求，数值范围更稳定
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

num_epochs = 1
batch_size = 1
train_data = MyDataset('./data/cifar10/train_py_t.txt', transform=data_transform)
trainloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

model = CIFAR(num_classes=10)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

fp = open("./backup/LW_py", "wb")
data_f = []
for name, param in model.named_parameters():
    # print(f"Layer: {name}, Parameter Shape: {param.shape}") #param.shape [filters, channels, ksize, ksize]  [outputs, inputs]
    if ("weight" in name and len(param.shape) == 4):
        data = param.tolist()
        for i in range(param.shape[0]):
            for j in range(param.shape[1]):
                for k in range(param.shape[2]):
                    data_f += data[i][j][k]
    if ("weight" in name and len(param.shape) == 2):
        data = param.tolist()
        for i in range(param.shape[0]):
            data_f += data[i]
    if ("weight" in name and len(param.shape) == 1):
        data_f += param.tolist()
    if ("bias" in name):
        data = param.tolist()
        data_f += data
# print(len(data_f))
for i in range(len(data_f)):
    fp.write(struct.pack('f', data_f[i]))
fp.close()

model.l1.register_forward_hook(forward_hook)
# model.l16.register_backward_hook(backward_hook)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0  # 记录每轮的损失

    for i, data in enumerate(trainloader, 0):
        # 获取输入：图片（inputs）和对应的标签（labels，比如0代表玫瑰，1代表郁金香）
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs, labels)
        outputs.backward()
        optimizer.step()
        print(outputs.item())
        # running_loss += outputs.item()
        # 每100个批量打印一次训练情况
        # if i % 100 == 99:
        #     print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
        #     running_loss = 0.0
        # print("loss: {}".format(outputs.item()))
