import os
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

class AlexNet(nn.Module):
    def __init__(self, num_classes=5):  # num_classes：类别数，默认1000（ImageNet）
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

data_transform = transforms.Compose([
    transforms.Resize((224,224)),  # 随机裁剪成224×224
    transforms.ToTensor(),  # 转换成Tensor（PyTorch能处理的数据格式）
    # 标准化：让数据更符合模型训练要求，数值范围更稳定
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

num_epochs = 1
batch_size = 1
train_data = MyDataset('./data/flower/train_test.txt', transform=data_transform)
trainloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

model = AlexNet(num_classes=5)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # lr是学习率，控制更新速度
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
        outputs = model(inputs)
        param = list(model.parameters())[0] #索引0表示第一层
        x1 = list(param)[0] #索引0表示第一个filter
        c10 = list(x1)[0] #索引0表示第一个channel
        c11 = list(x1)[1]
        c12 = list(x1)[2]
        print(x1.shape)
        print(c10)
        print(c11)
        print(c12)
        
        x2 = list(param)[1] #索引0表示第一个filter
        c20 = list(x1)[0] #索引0表示第一个channel
        c21 = list(x1)[1]
        c22 = list(x1)[2]
        print(x2.shape)
        print(c20)
        print(c21)
        print(c22)
        
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # 每100个批量打印一次训练情况
        if i % 100 == 99:
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0
    
    # 每轮训练结束后，在验证集上测试效果
    model.eval()  # 评估模式：关闭Dropout
    correct = 0  # 正确预测的数量
    total = 0    # 总图片数量
    
    # 验证集不需要计算梯度，节省资源
    with torch.no_grad():
        for data in trainloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            # 取概率最大的作为预测结果
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # 打印每轮的验证准确率
    print(f'Epoch {epoch + 1}, 验证准确率: {100 * correct / total:.2f}%')

print('训练完成！')

# 保存训练好的模型，方便后续使用
torch.save(model.state_dict(), 'alexnet_flower.pth')
print('模型已保存为 alexnet_flower.pth')



