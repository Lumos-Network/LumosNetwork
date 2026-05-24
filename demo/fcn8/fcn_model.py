import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from dataload import NUM_CLASSES

class FCN32s(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super(FCN32s, self).__init__()
        vgg16 = models.vgg16(pretrained=pretrained)# 加载预训练的VGG16模型
        features = list(vgg16.features.children())# 获取特征提取部分
        # 根据FCN原始论文修改VGG16网络
        # 前5段卷积块保持不变
        self.features1 = nn.Sequential(*features[:5])    # conv1 + pool1
        self.features2 = nn.Sequential(*features[5:10])  # conv2 + pool2
        self.features3 = nn.Sequential(*features[10:17]) # conv3 + pool3
        self.features4 = nn.Sequential(*features[17:24]) # conv4 + pool4
        self.features5 = nn.Sequential(*features[24:31]) # conv5 + pool5
        
        # 全连接层替换为1x1卷积
        self.fc6 = nn.Conv2d(512, 4096, kernel_size=7, padding=3)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()
        
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()
        
        # 分类层
        self.score = nn.Conv2d(4096, num_classes, kernel_size=1)
        
        # 上采样层: 32倍上采样回原始图像大小
        self.upsample = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, stride=32, padding=16, bias=False)
        
        # 初始化参数
        self._initialize_weights()
        
    def forward(self, x):
        # 记录输入尺寸用于上采样
        input_size = x.size()[2:]
        
        # 编码器 (VGG16)
        x = self.features1(x)
        x = self.features2(x)
        x = self.features3(x)
        x = self.features4(x)
        x = self.features5(x)
        
        # 全连接层 (以卷积形式实现)
        x = self.relu6(self.fc6(x))
        x = self.drop6(x)
        
        x = self.relu7(self.fc7(x))
        x = self.drop7(x)
        
        # 分类
        x = self.score(x)
        
        # 上采样回原始尺寸
        x = self.upsample(x)
        # 裁剪到原始图像尺寸
        x = x[:, :, :input_size[0], :input_size[1]]
        
        return x
    
    def _initialize_weights(self):
        # 初始化反卷积层的权重为双线性上采样
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                # 双线性上采样的初始化
                m.weight.data.zero_()
                m.weight.data = self._make_bilinear_weights(m.kernel_size[0], m.out_channels)
    
    def _make_bilinear_weights(self, size, num_channels):
        """生成双线性插值的权重"""
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = torch.FloatTensor(size, size)
        for i in range(size):
            for j in range(size):
                og[i, j] = (1 - abs((i - center) / factor)) * (1 - abs((j - center) / factor))
        filter = torch.zeros(num_channels, num_channels, size, size)
        for i in range(num_channels):
            filter[i, i] = og
        return filter


class FCN16s(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super(FCN16s, self).__init__()
        # 加载预训练的VGG16模型
        vgg16 = models.vgg16(pretrained=pretrained)
        # 获取特征提取部分
        features = list(vgg16.features.children())
        
        # 分段处理VGG16特征
        self.features1 = nn.Sequential(*features[:5])    # conv1 + pool1
        self.features2 = nn.Sequential(*features[5:10])  # conv2 + pool2
        self.features3 = nn.Sequential(*features[10:17]) # conv3 + pool3
        self.features4 = nn.Sequential(*features[17:24]) # conv4 + pool4
        self.features5 = nn.Sequential(*features[24:31]) # conv5 + pool5
        
        # 全连接层替换为1x1卷积
        self.fc6 = nn.Conv2d(512, 4096, kernel_size=7, padding=3)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()
        
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()
        
        # 分类层
        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        
        # pool4的1x1卷积，用于特征融合
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        
        # 2倍上采样conv7特征
        self.upsample2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False)
        
        # 16倍上采样回原始图像大小
        self.upsample16 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=32, stride=16, padding=8, bias=False)
        
        # 初始化参数
        self._initialize_weights()
        
    def forward(self, x):
        input_size = x.size()[2:]# 记录输入尺寸用于上采样
        # 编码器 (VGG16)
        x = self.features1(x)
        x = self.features2(x)
        x = self.features3(x)
        pool4 = self.features4(x)# 保存pool4的输出用于后续融合
        x = self.features5(pool4)
        # 全连接层 (以卷积形式实现)
        x = self.relu6(self.fc6(x))
        x = self.drop6(x)
        x = self.relu7(self.fc7(x))
        x = self.drop7(x)
        x = self.score_fr(x)# 分类
        # 2倍上采样
        x = self.upsample2(x)
        # 获取pool4的分数并裁剪
        score_pool4 = self.score_pool4(pool4)
        score_pool4 = score_pool4[:, :, :x.size()[2], :x.size()[3]]
        x = x + score_pool4# 融合特征
        x = self.upsample16(x)# 16倍上采样回原始尺寸
        x = x[:, :, :input_size[0], :input_size[1]]# 裁剪到原始图像尺寸
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                # 双线性上采样的初始化
                m.weight.data.zero_()
                m.weight.data = self._make_bilinear_weights(m.kernel_size[0], m.out_channels)
    
    def _make_bilinear_weights(self, size, num_channels):
        """生成双线性插值的权重"""
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = torch.FloatTensor(size, size)
        for i in range(size):
            for j in range(size):
                og[i, j] = (1 - abs((i - center) / factor)) * (1 - abs((j - center) / factor))
        filter = torch.zeros(num_channels, num_channels, size, size)
        for i in range(num_channels):
            filter[i, i] = og
        return filter


class FCN8s(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super(FCN8s, self).__init__()
        # 加载预训练的VGG16模型
        vgg16 = models.vgg16(pretrained=pretrained)
        # 获取特征提取部分
        features = list(vgg16.features.children())
        
        # 分段处理VGG16特征
        self.features1 = nn.Sequential(*features[:5])    # conv1 + pool1
        self.features2 = nn.Sequential(*features[5:10])  # conv2 + pool2
        self.features3 = nn.Sequential(*features[10:17]) # conv3 + pool3
        self.features4 = nn.Sequential(*features[17:24]) # conv4 + pool4
        self.features5 = nn.Sequential(*features[24:31]) # conv5 + pool5
        
        # 全连接层替换为1x1卷积
        self.fc6 = nn.Conv2d(512, 4096, kernel_size=7, padding=3)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()
        
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()
        
        # 分类层
        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        
        # pool3和pool4的1x1卷积，用于特征融合
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        
        # 2倍上采样conv7特征
        self.upsample2_1 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False)
        
        # 2倍上采样融合后的特征
        self.upsample2_2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False)
        
        # 8倍上采样回原始图像大小
        self.upsample8 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, padding=4, bias=False)
        
        # 初始化参数
        self._initialize_weights()
        
    def forward(self, x):
        input_size = x.size()[2:]# 记录输入尺寸用于上采样
        # 编码器 (VGG16)
        x = self.features1(x)
        x = self.features2(x)
        pool3 = self.features3(x)# 保存pool3的输出用于后续融合
        pool4 = self.features4(pool3)# 保存pool4的输出用于后续融合
        x = self.features5(pool4)
        # 全连接层 (以卷积形式实现)
        x = self.relu6(self.fc6(x))
        x = self.drop6(x)
        x = self.relu7(self.fc7(x))
        x = self.drop7(x)
        x = self.score_fr(x)# 分类
        x = self.upsample2_1(x)# 2倍上采样
        # 获取pool4的分数并裁剪
        score_pool4 = self.score_pool4(pool4)
        score_pool4 = score_pool4[:, :, :x.size()[2], :x.size()[3]]
        x = x + score_pool4 # 第一次融合特征 (pool5上采样 + pool4)
        x = self.upsample2_2(x)# 再次2倍上采样
        # 获取pool3的分数并裁剪
        score_pool3 = self.score_pool3(pool3)
        score_pool3 = score_pool3[:, :, :x.size()[2], :x.size()[3]]
        x = x + score_pool3# 第二次融合特征 (第一次融合的上采样 + pool3)
        x = self.upsample8(x)# 8倍上采样回原始尺寸
        x = x[:, :, :input_size[0], :input_size[1]]# 裁剪到原始图像尺寸
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                # 双线性上采样的初始化
                m.weight.data.zero_()
                m.weight.data = self._make_bilinear_weights(m.kernel_size[0], m.out_channels)
    
    def _make_bilinear_weights(self, size, num_channels):
        """生成双线性插值的权重"""
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = torch.FloatTensor(size, size)
        for i in range(size):
            for j in range(size):
                og[i, j] = (1 - abs((i - center) / factor)) * (1 - abs((j - center) / factor))
        filter = torch.zeros(num_channels, num_channels, size, size)
        for i in range(num_channels):
            filter[i, i] = og
        print(filter.shape)
        return filter


def get_fcn_model(model_type='fcn8s', num_classes=NUM_CLASSES, pretrained=True):
    """
    获取FCN模型
    
    参数:
        model_type (str): 'fcn32s', 'fcn16s' 或 'fcn8s'
        num_classes (int): 类别数量
        pretrained (bool): 是否使用预训练的VGG16作为编码器
    
    返回:
        model: FCN模型
    """
    if model_type == 'fcn32s':
        return FCN32s(num_classes=num_classes, pretrained=pretrained)
    elif model_type == 'fcn16s':
        return FCN16s(num_classes=num_classes, pretrained=pretrained)
    elif model_type == 'fcn8s':
        return FCN8s(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError("Unsupported model type. Choose from 'fcn32s', 'fcn16s', or 'fcn8s'.")
