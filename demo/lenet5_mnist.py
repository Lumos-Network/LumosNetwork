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

class Lenet5(nn.Module):
    def __init__(self, num_classes=2):
        super(Lenet5, self).__init__()
        self.l1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0)
        self.l2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.l3 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.l4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.l5 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)
        self.l6 = nn.Linear(120, 84)
        self.l7 = nn.Linear(84, num_classes)
        self.l8 = nn.CrossEntropyLoss()

    def forward(self, x, labels):
        x = torch.relu(self.l1(x))
        x = self.l2(x)
        x = torch.relu(self.l3(x))
        x = self.l4(x)
        x = torch.relu(self.l5(x))
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x, labels)
        return x

model = Lenet5(num_classes=10)
# for name, param in model.named_parameters():
#     print(name)
#     if ("weight" in name and len(param.shape) == 4):
#         print(param.shape)
#         data = param.tolist()
#         print(data[0][0][0])
#         print(data[0][0][1])
