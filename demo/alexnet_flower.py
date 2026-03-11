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
    data_f = []
    data = output.tolist()
    # print(output.shape)
    with open("./backup/out_py", "wb") as fp:
        for i in range(output.shape[0]):
            # for j in range(output.shape[1]):
                # for k in range(output.shape[2]):
            data_f += data[i]
        for i in range(len(data_f)):
            fp.write(struct.pack('f', data_f[i]))
        fp.close()
    
    input = list(input)[0]
    data_f = []
    data = input.tolist()
    print(input.shape)
    with open("./backup/in_py", "wb") as fp:
        for i in range(input.shape[0]):
            # for j in range(input.shape[1]):
                # for k in range(input.shape[2]):
            data_f += data[i]
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
    print(grad_output[0].shape)
    bias_grad = grad_output[0]
    grad = bias_grad.tolist()
    with open("./backup/grad_py", "wb") as fp:
        for i in range(len(grad[0])):
            for j in range(len(grad[0][0])):
                for k in range(len(grad[0][0][0])):
                    fp.write(struct.pack('f', grad[0][i][j][k]))
        
        # for i in range(len(grad[0])):
        #     fp.write(struct.pack('f', grad[0][i]))
        fp.close()
    
    # print(grad_input[0].shape)

def backward_hook_1(module, grad_input, grad_output):
    print(grad_output[0].shape)
    bias_grad = grad_output[0]
    grad = bias_grad.tolist()
    with open("./backup/grad_py_maxin", "wb") as fp:
        for i in range(len(grad[0])):
            for j in range(len(grad[0][0])):
                for k in range(len(grad[0][0][0])):
                    fp.write(struct.pack('f', grad[0][i][j][k]))
        
        # for i in range(len(grad[0])):
        #     fp.write(struct.pack('f', grad[0][i]))
        # print(grad)
        fp.close()
    
    # print(grad_input[0].shape)

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
    def __init__(self, num_classes=2):  # num_classes：类别数，默认1000（ImageNet）
        super(AlexNet, self).__init__()
        # 卷积部分：5层卷积+3层池化
        self.l1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)  # 输入3通道，输出96通道
        self.l2 = nn.MaxPool2d(kernel_size=3, stride=2)  # 池化
        self.l3 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.l4 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.l5 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.l6 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.l7 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.l8 = nn.MaxPool2d(kernel_size=3, stride=2)

        # self.l9 = nn.Dropout(p=0.5)
        self.l10 = nn.Linear(256 * 6 * 6, 4096)
        # self.l11 = nn.Dropout(p=0.5)
        self.l12 = nn.Linear(4096, 4096)
        self.l13 = nn.Linear(4096, num_classes)
        
        self.l14 = nn.CrossEntropyLoss()
    
    # 前向传播：定义数据在网络里的流动路径
    def forward(self, x, labels):
        x = torch.relu(self.l1(x))
        x = self.l2(x)
        x = torch.relu(self.l3(x))
        x = self.l4(x)
        x = torch.relu(self.l5(x))
        x = torch.relu(self.l6(x))
        x = torch.relu(self.l7(x))
        x = self.l8(x)
        
        x = torch.flatten(x, start_dim=1)
        # x = self.l9(x)
        x = torch.relu(self.l10(x))
        # x = self.l11(x)
        x = torch.relu(self.l12(x))
        x = self.l13(x)
        x = self.l14(x, labels)
        # x = self.l15(x, labels)
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
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # lr是学习率，控制更新速度

# fp = open("./backup/LW_py", "wb")
# data_f = []
for name, param in model.named_parameters():
    print(f"Layer: {name}, Parameter Shape: {param.shape}") #param.shape [filters, channels, ksize, ksize]  [outputs, inputs]
#     if ("weight" in name and len(param.shape) == 4):
#         data = param.tolist()
#         for i in range(param.shape[0]):
#             for j in range(param.shape[1]):
#                 for k in range(param.shape[2]):
#                     data_f += data[i][j][k]
#     if ("weight" in name and len(param.shape) == 2):
#         data = param.tolist()
#         for i in range(param.shape[0]):
#             data_f += data[i]
#     if ("bias" in name):
#         data = param.tolist()
#         data_f += data
# print(len(data_f))
# for i in range(len(data_f)):
#     fp.write(struct.pack('f', data_f[i]))
# fp.close()

# # model.l14.register_forward_hook(forward_hook)
# model.l3.register_backward_hook(backward_hook)
# # model.l8.register_backward_hook(backward_hook_1)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # device = torch.device("cpu")
# model.to(device)
# # criterion = nn.CrossEntropyLoss()

# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0  # 记录每轮的损失

#     for i, data in enumerate(trainloader, 0):
#         # 获取输入：图片（inputs）和对应的标签（labels，比如0代表玫瑰，1代表郁金香）
#         inputs, labels = data[0].to(device), data[1].to(device)
#         optimizer.zero_grad()
#         outputs = model(inputs, labels)
#         # print(labels)
#         # loss = criterion(outputs, labels)
#         # print(outputs.item())
#         outputs.backward()
#         optimizer.step()
#         print(outputs.item())
#         # running_loss += outputs.item()
#         # 每100个批量打印一次训练情况
#         # if i % 100 == 99:
#         #     print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
#         #     running_loss = 0.0
#         # print("loss: {}".format(outputs.item()))

# fp = open("./backup/LWF_py", "wb")
# data_f = []
# for name, param in model.named_parameters():
#     print(f"Layer: {name}, Parameter Shape: {param.shape}") #param.shape [filters, channels, ksize, ksize]  [outputs, inputs]
#     if ("weight" in name and len(param.shape) == 4):
#         data = param.tolist()
#         for i in range(param.shape[0]):
#             for j in range(param.shape[1]):
#                 for k in range(param.shape[2]):
#                     data_f += data[i][j][k]
#     if ("weight" in name and len(param.shape) == 2):
#         data = param.tolist()
#         for i in range(param.shape[0]):
#             data_f += data[i]
#     if ("bias" in name):
#         data = param.tolist()
#         data_f += data
# print(len(data_f))
# for i in range(len(data_f)):
#     fp.write(struct.pack('f', data_f[i]))
# fp.close()
#     # if param.grad is not None:
#     #     print(f"Parameter: {name}, Gradient: {param.grad}")
#     # # 每轮训练结束后，在验证集上测试效果
#     # model.eval()  # 评估模式：关闭Dropout
#     # correct = 0  # 正确预测的数量
#     # total = 0    # 总图片数量
    
#     # # 验证集不需要计算梯度，节省资源
#     # with torch.no_grad():
#     #     for data in trainloader:
#     #         images, labels = data[0].to(device), data[1].to(device)
#     #         outputs = model(images, labels)
#     #         # 取概率最大的作为预测结果
#     #         _, predicted = torch.max(outputs.data, 1)
#     #         total += labels.size(0)
#     #         correct += (predicted == labels).sum().item()
    
#     # # 打印每轮的验证准确率
#     # print(f'Epoch {epoch + 1}, 验证准确率: {100 * correct / total:.2f}%')

# # print('训练完成！')

# # # 保存训练好的模型，方便后续使用
# # torch.save(model.state_dict(), 'alexnet_flower.pth')
# # print('模型已保存为 alexnet_flower.pth')



