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

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()

        self.l1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.l2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.l3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.l4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.l5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.l6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.l7 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.l8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.l9 = nn.Conv2d(64, 128, kernel_size=1, stride=2)

        self.l10 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.l11 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.l12 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.l13 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.l14 = nn.Conv2d(128, 256, kernel_size=1, stride=2)

        self.l15 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.l16 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.l17 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.l18 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.l19 = nn.Conv2d(256, 512, kernel_size=1, stride=2)

        self.l20 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.l21 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.l22 = nn.AvgPool2d(kernel_size=7)
        self.l23 = nn.Linear(512, num_classes)
        self.l24 = nn.CrossEntropyLoss()

    def forward(self, x, labels):
        x = torch.relu(self.l1(x))
        x = self.l2(x)

        y = torch.relu(self.l3(x))
        y = self.l4(y)
        x = torch.relu(x+y)

        y = torch.relu(self.l5(x))
        y = self.l6(y)
        x = torch.relu(x+y)

        y = torch.relu(self.l7(x))
        y = self.l8(y)
        z = self.l9(x)
        x = torch.relu(z+y)

        y = torch.relu(self.l10(x))
        y = self.l11(y)
        x = torch.relu(x+y)
        
        y = torch.relu(self.l12(x))
        y = self.l13(y)
        z = self.l14(x)
        x = torch.relu(z+y)
        
        y = torch.relu(self.l15(x))
        y = self.l16(y)
        x = torch.relu(x+y)
        
        y = torch.relu(self.l17(x))
        y = self.l18(y)
        z = self.l19(x)
        x = torch.relu(z+y)
        
        y = torch.relu(self.l20(x))
        y = self.l21(y)
        x = torch.relu(x+y)
        
        x = self.l22(x) # 需要激活
        x = torch.flatten(x, start_dim=1)
        x = self.l23(x)
        x = self.l24(x, labels)
        return x

data_transform = transforms.Compose([
    transforms.Resize((224,224)),  # 随机裁剪成224×224
    transforms.ToTensor(),  # 转换成Tensor（PyTorch能处理的数据格式）
    # 标准化：让数据更符合模型训练要求，数值范围更稳定
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

num_epochs = 4
batch_size = 4
train_data = MyDataset('./data/flower/train_test.txt', transform=data_transform)
trainloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

model = ResNet(num_classes=5)
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

# model.l9.register_forward_hook(forward_hook)
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

import torch.nn as nn
class ChannelShuffleModule(nn.Module):
    def __init__(self, channels, groups):
        super().__init__()
        assert channels % groups == 0
        self.channels = channels
        self.groups = groups
        self.channel_per_group = self.channels // self.groups

    def forward(self, x):
        '''
            x.shape: [b, c, series, modal]
        '''
        batch, _, series, modal = x.size()
        x = x.reshape(batch, self.groups, self.channel_per_group, series, modal)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batch, self.channels, series, modal)
        return x

class ShuffleNet(nn.Module):
    def __init__(self, train_shape, category, kernel_size=3):
        super(ShuffleNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(1, 1, (kernel_size, 1), (2, 1), (kernel_size // 2, 0), groups=1),
            nn.Conv2d(1, 64, 1, 1, 0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ChannelShuffleModule(channels=64, groups=8),

            nn.Conv2d(64, 64, (kernel_size, 1), (2, 1), (kernel_size // 2, 0), groups=64),
            nn.Conv2d(64, 128, 1, 1, 0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ChannelShuffleModule(channels=128, groups=8),

            nn.Conv2d(128, 128, (kernel_size, 1), (2, 1), (kernel_size // 2, 0), groups=128),
            nn.Conv2d(128, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ChannelShuffleModule(channels=256, groups=16),

            nn.Conv2d(256, 256, (kernel_size, 1), (2, 1), (kernel_size // 2, 0), groups=256),
            nn.Conv2d(256, 512, 1, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            ChannelShuffleModule(channels=512, groups=16)
        )
        self.ada_pool = nn.AdaptiveAvgPool2d((1, train_shape[-1]))
        self.fc = nn.Linear(512*train_shape[-1], category)

    def forward(self, x):
        x = self.layer(x)
        x = self.ada_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
