import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, datasets, utils
import torch.optim as optim
import struct
import numpy as np

VOC_COLORMAP = [
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
    [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
    [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
    [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128]
]

def forward_hook(module, input, output):
    flag = "output"
    shape = None
    data = None
    if (flag == "input"):
        shape = input[0].shape
        data = input[0].tolist()
    else:
        shape = output.shape
        data = output.tolist()
    print("OutShape: {}".format(shape))
    data_f = []
    with open("./backup/in_py", "wb") as fp:
        for i in range(shape[0]):
            for j in range(shape[1]):
                # for k in range(shape[2]):
                data_f += data[i][j]
        for i in range(len(data_f)):
            fp.write(struct.pack('i', int(data_f[i])))
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
            pathi = line.rstrip()
            pathl = pathi[:-4]+"l.png"
            imgs.append((pathi, pathl)) # 类别转为整型int
            self.imgs = imgs 
            self.transform = transform
            self.target_transform = target_transform

    def __getitem__(self, index):
        pathi, pathl = self.imgs[index]
        img = Image.open(pathi).convert('RGB') 
        if self.transform is not None:
            img = self.transform(img)
        label = Image.open(pathl).convert('RGB')
        label = label.resize((320, 320), resample=Image.NEAREST)
        if self.target_transform is not None:
            label = self.target_transform(label)
        else:
            label = np.array(label)
            if len(label.shape) != 3 or label.shape[2] != 3:
                raise ValueError(f"掩码维度错误: {label.shape}, 期望为 (H,W,3)")
            label_copy = np.zeros((320, 320), dtype=np.uint8)
            for k, color in enumerate(VOC_COLORMAP):
                r_match = label[:, :, 0] == color[0]
                g_match = label[:, :, 1] == color[1]
                b_match = label[:, :, 2] == color[2]
                color_match = r_match & g_match & b_match
                label_copy[color_match] = k
            label = torch.from_numpy(label_copy).long()
        return img, label

    def __len__(self):
        return len(self.imgs)

class DCONV(nn.Module):
    def __init__(self):
        super(DCONV, self).__init__()
        self.l1 = nn.Conv2d(3, 21, 5, 1, 1)
        self.l2 = nn.ConvTranspose2d(21, 21, 5, 1, 1, bias=False)
        self.l3 = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, x, labels):
        x = self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)
        x = self.l3(x, labels)
        return x

data_transform = transforms.Compose([
    transforms.ToTensor(),  # 转换成Tensor（PyTorch能处理的数据格式）
    # 标准化：让数据更符合模型训练要求，数值范围更稳定
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

num_epochs = 1
batch_size = 4
train_data = MyDataset('./data/VOCT/train.txt', transform=data_transform)
trainloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)

model = DCONV()
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

# model.l3.register_forward_hook(forward_hook)
model.l2.register_backward_hook(backward_hook)

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

# import torch
# import torch.nn as nn
# from torchvision import models
# import struct

# # ====================== 1. 手动搭建 VGG16 网络 ======================
# class VGG16(nn.Module):
#     def __init__(self, num_classes=1000):
#         super(VGG16, self).__init__()
#         self.features = nn.Sequential(
#             # Block 1
#             nn.Conv2d(3, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             # Block 2
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             # Block 3
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             # Block 4
#             nn.Conv2d(256, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             # Block 5
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )

#         # 分类头
#         self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
#         self.classifier = nn.Sequential(
#             nn.Linear(512 * 7 * 7, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, num_classes),
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x

# # ====================== 2. 加载官方预训练权重 ======================
# model = VGG16(num_classes=1000)

# # 2. 加载 torchvision 提供的官方预训练权重
# pretrained_vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

# # 3. 把预训练权重加载到自己的模型
# model.load_state_dict(pretrained_vgg.state_dict())

# fp = open("./backup/LW_py", "wb")
# data_f = []
# for name, param in model.named_parameters():
#     # print(f"Layer: {name}, Parameter Shape: {param.shape}") #param.shape [filters, channels, ksize, ksize]  [outputs, inputs]
#     if "features" not in name:
#         break
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
#     if ("weight" in name and len(param.shape) == 1):
#         data_f += param.tolist()
#     if ("bias" in name):
#         data = param.tolist()
#         data_f += data
# # print(len(data_f))
# for i in range(len(data_f)):
#     fp.write(struct.pack('f', data_f[i]))
# fp.close()

