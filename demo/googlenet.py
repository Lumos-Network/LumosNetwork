from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, datasets, utils
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import struct

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
    input_grad = grad_input[0]
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

#   创建所需的模板文件
#   基本卷积模板
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

#   Inception 结构模板
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)  # 保证输出大小等于输入大小
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)  # 保证输出大小等于输入大小
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        #   将输入特征矩阵分别输入到四个分支
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        #   将输出放入一个列表中
        outputs = [branch1, branch2, branch3, branch4]
        #   通过torch.cat合并四个输出，合并维度为1，即按照通道维度合并
        return torch.cat(outputs, 1)

#   InceptionAux 辅助分类器模板
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)  # output[batch, 128, 4, 4]

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        #   aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14

        x = self.averagePool(x)
        #   aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4

        x = self.conv(x)
        #   N x 128 x 4 x 4

        #   特征矩阵展平，从channel维度开始展平
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.5, training=self.training)
        #   N x 2048

        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.5, training=self.training)
        #   N x 1024

        x = self.fc2(x)
        #   N x num_classes
        return x

#   定义GoogLeNet网络
class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, init_weight=False):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)  # ceil_mode=True 计算为小数时，向上取整
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, padding=0)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        #   辅助分类器
        # if aux_logits:
        #     self.aux1 = InceptionAux(512, num_classes)
        #     self.aux2 = InceptionAux(528, num_classes)

        #   AdaptiveAvgPool2d 自适应全局平均池化
        self.avgpool = nn.AvgPool2d(7, 7, 0)
        # self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        self.loss = nn.CrossEntropyLoss()
        if init_weight:
            self._initialize_weights()

    def forward(self, x, labels):
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        # if self.training and self.aux_logits:  # eval model lose this layer
        #     aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        # if self.training and self.aux_logits:  # eval model lose this layer
        #     aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        # x = self.dropout(x)
        x = self.fc(x)
        x = self.loss(x, labels)
        # N x 1000 (num_classes)
        # if self.training and self.aux_logits:  # eval model lose this layer
        #     return x, aux2, aux1
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

data_transform = transforms.Compose([
    transforms.Resize((224,224)),  # 随机裁剪成224×224
    transforms.ToTensor(),  # 转换成Tensor（PyTorch能处理的数据格式）
    # 标准化：让数据更符合模型训练要求，数值范围更稳定
    transforms.Normalize([0.491, 0.482, 0.446], [0.247, 0.243, 0.261])
])

num_epochs = 1
batch_size = 4
train_data = MyDataset('./data/flower/train_test.txt', transform=data_transform)
trainloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)

model = GoogLeNet(num_classes=5)
print(model)
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

# model.avgpool.register_forward_hook(forward_hook)
model.maxpool4.register_backward_hook(backward_hook)

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
