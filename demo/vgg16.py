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

data_transform = transforms.Compose([
    transforms.Resize((32,32)),  # 随机裁剪成224×224
    transforms.ToTensor(),  # 转换成Tensor（PyTorch能处理的数据格式）
    # 标准化：让数据更符合模型训练要求，数值范围更稳定
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

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

class Vgg16(nn.Module):
    def __init__(self, num_classes=10):
        super(Vgg16, self).__init__()
        self.l1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.l2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.l3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.l4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.l5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.l6 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.l7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.l8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.l9 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.l10 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.l11 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.l12 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.l13 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.l14 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.l15 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.l16 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.l17 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.l18 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.l19 = nn.Linear(512, 512)
        self.l20 = nn.Dropout(p=0.5)
        self.l21 = nn.Linear(512, 512)
        self.l22 = nn.Dropout(p=0.5)
        self.l23 = nn.Linear(512, 10)
        self.l24 = nn.CrossEntropyLoss()

    def forward(self, x, labels):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = self.l3(x)

        x = torch.relu(self.l4(x))
        x = torch.relu(self.l5(x))
        x = self.l6(x)

        x = torch.relu(self.l7(x))
        x = torch.relu(self.l8(x))
        x = torch.relu(self.l9(x))
        x = self.l10(x)

        x = torch.relu(self.l11(x))
        x = torch.relu(self.l12(x))
        x = torch.relu(self.l13(x))
        x = self.l14(x)

        x = torch.relu(self.l15(x))
        x = torch.relu(self.l16(x))
        x = torch.relu(self.l17(x))
        x = self.l18(x)

        x = torch.flatten(x, 1)
        x = torch.relu(self.l19(x))
        x = self.l20(x)
        x = torch.relu(self.l21(x))
        x = self.l22(x)
        x = torch.relu(self.l23(x))
        x = self.l24(x, labels)
        return x
num_epochs = 50
batch_size = 32
train_data = MyDataset('./data/cifar10/train_py.txt', transform=data_transform)
trainloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

optim.Adam()
model = Vgg16(num_classes=10)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    loss = 0.0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs, labels)
        # print(outputs.item())
        outputs.backward()
        optimizer.step()
        loss += outputs.item()
        # 每100个批量打印一次训练情况
        if i % 100 == 99:
            print(f'[{epoch + 1}, {i + 1}] loss: {loss / 100:.5f}')
            loss = 0.0
        # print("loss: {}".format(outputs.item()))
