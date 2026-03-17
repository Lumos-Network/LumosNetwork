import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.l1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)
        self.l2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.l3 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.l4 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.l5 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(384)
        self.l6 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.l7 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.l8 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.l9 = nn.Dropout(p=0.5)
        self.l10 = nn.Linear(256 * 6 * 6, 4096)
        self.bn2 = nn.BatchNorm1d(4096)
        self.l11 = nn.Dropout(p=0.5)
        self.l12 = nn.Linear(4096, 4096)
        self.l13 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.features(x)  # 经过卷积部分
        x = torch.flatten(x, start_dim=1)  # 拉平向量（从第1维开始，第0维是批量数）
        x = self.classifier(x)  # 经过全连接部分
        return x

# 数据预处理：把图片转换成网络需要的格式，同时做数据增强（提高模型泛化能力）
data_transform = {
    # 训练集：除了基本转换，还做随机裁剪、翻转，增加数据多样性
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),  # 随机裁剪成224×224
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),  # 转换成Tensor（PyTorch能处理的数据格式）
        # 标准化：让数据更符合模型训练要求，数值范围更稳定
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # 测试集：只做基本转换，不做数据增强
    "val": transforms.Compose([
        transforms.Resize(256),  # 缩放到256×256
        transforms.CenterCrop(224),  # 中心裁剪成224×224
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# 下载数据集到当前文件夹的"data"目录下
data_root = "./data"  # 数据集保存路径
train_dataset = datasets.Flowers17(
    root=data_root, split="train", download=True, transform=data_transform["train"]
)
val_dataset = datasets.Flowers17(
    root=data_root, split="val", download=True, transform=data_transform["val"]
)

# 创建数据加载器：批量加载数据，方便训练
batch_size = 32  # 每次加载32张图片
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 初始化网络：花分类是5类，所以num_classes=5
model = AlexNet(num_classes=5)
# 检查网络结构（可选）
print(model)

# 1. 损失函数：计算模型预测值和真实值的差距，指导模型改进
criterion = nn.CrossEntropyLoss()  # 适合多分类任务

# 2. 优化器：更新网络参数，减小损失。这里用SGD（随机梯度下降）
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # lr是学习率，控制更新速度

# 3. 训练设备：优先用GPU（速度快），没有GPU就用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)  # 把模型放到指定设备上

num_epochs = 10  # 训练轮次，新手可以先设10，后续再增加

for epoch in range(num_epochs):
    # 训练模式：开启Dropout
    model.train()
    running_loss = 0.0  # 记录每轮的损失
    
    # 遍历训练集，批量处理图片
    for i, data in enumerate(train_loader, 0):
        # 获取输入：图片（inputs）和对应的标签（labels，比如0代表玫瑰，1代表郁金香）
        inputs, labels = data[0].to(device), data[1].to(device)
        
        # 重要：每次更新前把梯度清零，避免累计
        optimizer.zero_grad()
        
        # 前向传播：模型预测
        outputs = model(inputs)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播：计算梯度，指导参数更新
        loss.backward()
        # 优化器更新参数
        optimizer.step()
        
        # 统计损失
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
        for data in val_loader:
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

# 1. 加载训练好的模型
model = AlexNet(num_classes=5)
model.load_state_dict(torch.load('alexnet_flower.pth'))
model.to(device)
model.eval()  # 切换到评估模式

# 2. 定义类别名称（对应标签0-4）
class_names = ['玫瑰', '郁金香', '雏菊', '蒲公英', '向日葵']

# 3. 处理要测试的图片（替换成你自己的图片路径，比如'./test_rose.jpg'）
img_path = './test_flower.jpg'
img = Image.open(img_path).convert('RGB')  # 打开图片并转成RGB格式

# 4. 图片预处理（和验证集的处理一致）
transform = data_transform["val"]
img_tensor = transform(img).unsqueeze(0)  # 增加一个维度（对应批量数）
img_tensor = img_tensor.to(device)

# 5. 模型预测
with torch.no_grad():
    outputs = model(img_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)  # 把输出转成概率
    _, predicted = torch.max(outputs, 1)
    pred_class = class_names[predicted[0]]
    pred_prob = probabilities[0][predicted[0]].item() * 100

# 6. 显示图片和预测结果
plt.imshow(img)
plt.title(f'预测结果：{pred_class}，概率：{pred_prob:.2f}%')
plt.axis('off')  # 隐藏坐标轴
plt.show()

