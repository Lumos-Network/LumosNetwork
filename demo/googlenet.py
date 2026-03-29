import torch  # 导入PyTorch主库
from torch import nn  # 导入神经网络模块
from torchsummary import summary  # 导入模型结构摘要工具
 
# 定义Inception模块
class Inception(nn.Module): # Inception(192, 64, 96, 128, 16, 32, 32)
    def __init__(
        self, in_channels, out1x1, out3x3red, out3x3, out5x5red, out5x5, pool_proj
    ):
        super(Inception, self).__init__()  # 调用父类构造函数
        self.ReLu = nn.ReLU()  # 定义ReLU激活函数
 
        self.branch1x1 = nn.Conv2d(in_channels, out1x1, kernel_size=1)  # 1x1卷积分支
 
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out3x3red, kernel_size=1),  # 1x1卷积降维
            nn.ReLU(),  # 激活
            nn.Conv2d(out3x3red, out3x3, kernel_size=3, padding=1),  # 3x3卷积
        )
 
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out5x5red, kernel_size=1),  # 1x1卷积降维
            nn.ReLU(),  # 激活
            nn.Conv2d(out5x5red, out5x5, kernel_size=5, padding=2),  # 5x5卷积
        )
 
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),  # 3x3最大池化
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),  # 1x1卷积
        )
 
    def forward(self, x):
        p1 = self.ReLu(self.branch1x1(x))  # 1x1卷积分支前向传播
        p2 = self.ReLu(self.branch3x3(x))  # 3x3卷积分支前向传播
        p3 = self.ReLu(self.branch5x5(x))  # 5x5卷积分支前向传播
        p4 = self.ReLu(self.branch_pool(x))  # 池化分支前向传播
        outputs = [p1, p2, p3, p4]  # 合并所有分支输出
        return torch.cat(outputs, 1)  # 在通道维度拼接
 
# 定义GoogLeNet主结构
class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(GoogLeNet, self).__init__()  # 调用父类构造函数
 
        self.b1 = nn.Sequential(
            nn.Conv2d(1, 64, 7, 2, 3),  # 7x7卷积，步长2，填充3
            nn.ReLU(),  # 激活
            nn.MaxPool2d(3, 2, 1)  # 3x3最大池化，步长2，填充1
        )
 
        self.b2 = nn.Sequential(
            nn.Conv2d(64, 64, 1),  # 1x1卷积
            nn.ReLU(),  # 激活
            nn.Conv2d(64, 192, 3, 1, 1),  # 3x3卷积
            nn.ReLU(),  # 激活
            nn.MaxPool2d(3, 2, 1),  # 3x3最大池化
        )
 
        self.b3 = nn.Sequential(
            Inception(192, 64, 96, 128, 16, 32, 32),  # 第一个Inception模块
            Inception(256, 128, 128, 192, 32, 96, 64),  # 第二个Inception模块
            nn.MaxPool2d(3, 2, 1),  # 池化
        )
 
        self.b4 = nn.Sequential(
            Inception(480, 192, 96, 208, 16, 48, 64),  # 多个Inception模块
            Inception(512, 160, 112, 224, 24, 64, 64),
            Inception(512, 128, 128, 256, 24, 64, 64),
            Inception(512, 112, 128, 288, 32, 64, 64),
            Inception(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(3, 2, 1),  # 池化
        )
 
        self.b5 = nn.Sequential(
            Inception(832, 256, 160, 320, 32, 128, 128),  # 倒数第二个Inception
            Inception(832, 384, 192, 384, 48, 128, 128),  # 最后一个Inception
        )
 
        self.b6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 自适应平均池化到1x1
            nn.Flatten(),  # 展平
            nn.Linear(1024, num_classes),  # 全连接输出
        )
 
        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")  # He初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # 偏置初始化为0
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)  # 正态分布初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # 偏置初始化为0
 
    def forward(self, x):
        x = self.b1(x)  # 第一阶段
        x = self.b2(x)  # 第二阶段
        x = self.b3(x)  # 第三阶段
        x = self.b4(x)  # 第四阶段
        x = self.b5(x)  # 第五阶段
        x = self.b6(x)  # 分类阶段
        return x  # 返回输出
 
# 测试模型结构
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择设备
    model = GoogLeNet().to(device=device)  # 实例化模型并移动到设备
    print(summary(model, (1, 224, 224)))  # 打印模型结构摘要