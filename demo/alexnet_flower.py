import os
import sys
import json
import torch
import time
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import torchvision.models as models

import torch
import torch.nn as nn
 
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):  # num_classes：类别数，默认1000（ImageNet）
        super(AlexNet, self).__init__()
        # 卷积部分：5层卷积+3层池化
        self.features = nn.Sequential(
            # Conv1 + Pool1
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),  # 输入3通道，输出96通道
            nn.ReLU(inplace=True),  # inplace=True：节省内存
            nn.MaxPool2d(kernel_size=3, stride=2),  # 池化
        
            # Conv2 + Pool2
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        
            # Conv3 + Conv4 + Conv5 + Pool3
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # 全连接部分：3层全连接
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),  # Dropout概率50%
            nn.Linear(256 * 6 * 6, 4096),  # 输入是6×6×256的拉平向量
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            
            nn.Linear(4096, num_classes),  # 输出是类别数
        )
    
    # 前向传播：定义数据在网络里的流动路径
    def forward(self, x):
        x = self.features(x)  # 经过卷积部分
        x = torch.flatten(x, start_dim=1)  # 拉平向量（从第1维开始，第0维是批量数）
        x = self.classifier(x)  # 经过全连接部分
        return x
 
# 初始化网络：花分类是5类，所以num_classes=5
model = AlexNet(num_classes=5)
# 检查网络结构（可选）
# for param in model.parameters():
#     print(param.shape)

param = list(model.parameters())[0]
x1 = list(param)[1]
x2 = list(x1)[0]
print(x1.shape)
print(x2.shape)
print(x2)
