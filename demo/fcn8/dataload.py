import os # 导入os模块，用于文件路径操作
import numpy as np # 导入numpy模块，用于数值计算
import torch # 导入PyTorch主模块
from torch.utils.data import Dataset, DataLoader # 导入数据集和数据加载器
from PIL import Image # 导入PIL图像处理库
import torchvision.transforms as transforms # 导入图像变换模块
from torchvision.transforms import functional as F # 导入函数式变换模块

# VOC数据集的类别名称（21个类别，包括背景）
VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 
    'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 
    'train', 'tv/monitor'
]

# VOC数据集有21个类别 (包括背景)
NUM_CLASSES = len(VOC_CLASSES)

# 定义PIL的重采样常量（PIL库的常量在不同版本中可能不一致，这里使用数值）
PIL_NEAREST = 0  # 最近邻重采样方式，保持锐利边缘，适用于掩码
PIL_BILINEAR = 1  # 双线性重采样方式，平滑图像，适用于原始图像

# 定义VOC数据集的颜色映射 (用于可视化分割结果)
# 每个类别对应一个RGB颜色
VOC_COLORMAP = [
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
    [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
    [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
    [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128]
]

class VOCSegmentation(Dataset):
    """
    VOC2012语义分割数据集的PyTorch Dataset实现
    负责数据的加载、预处理和转换
    """
    def __init__(self, root, split='train', transform=None, target_transform=None, img_size=320):
        """
        初始化数据集
        
        参数:
            root (string): VOC数据集的根目录路径
            split (string, optional): 使用的数据集划分，可选 'train', 'val' 或 'trainval'
            transform (callable, optional): 输入图像的变换函数
            target_transform (callable, optional): 目标掩码的变换函数
            img_size (int, optional): 调整图像和掩码的大小
        """
        super(VOCSegmentation, self).__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.img_size = img_size
        # 确定图像和标签文件的路径
        voc_root = self.root
        image_dir = os.path.join(voc_root, 'JPEGImages')  # 原始图像目录
        mask_dir = os.path.join(voc_root, 'SegmentationClass')  # 语义分割标注目录
        # 获取图像文件名列表（从划分文件中读取）
        splits_dir = os.path.join(voc_root, 'ImageSets', 'Segmentation')
        split_file = os.path.join(splits_dir, self.split + '.txt')
        # 确保分割文件存在
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"找不到拆分文件: {split_file}")
        # 读取文件名列表
        with open(split_file, 'r') as f:
            file_names = [x.strip() for x in f.readlines()]
        # 构建图像和掩码的完整路径
        self.images = [os.path.join(image_dir, x + '.jpg') for x in file_names]
        self.masks = [os.path.join(mask_dir, x + '.png') for x in file_names]
        # 检查文件是否存在，打印警告但不中断程序
        for img_path in self.images:
            if not os.path.exists(img_path):
                print(f"警告: 图像文件不存在: {img_path}")
        for mask_path in self.masks:
            if not os.path.exists(mask_path):
                print(f"警告: 掩码文件不存在: {mask_path}")
        # 确保图像和掩码数量匹配
        assert len(self.images) == len(self.masks), "图像和掩码数量不匹配"
        print(f"加载了 {len(self.images)} 对图像和掩码用于{split}集")
    
    def __getitem__(self, index):
        """
        获取数据集中的一个样本
        
        参数:
            index (int): 样本索引
            
        返回:
            tuple: (图像, 掩码) 对，分别为图像张量和掩码张量
        """
        # 加载图像和掩码
        img_path = self.images[index]
        mask_path = self.masks[index]
        # 打开图像并转换为RGB格式
        img = Image.open(img_path).convert('RGB')
        # 打开掩码并转换为RGB格式（确保与colormap匹配）
        mask = Image.open(mask_path).convert('RGB')
        # 统一调整图像和掩码大小，确保尺寸一致
        # 对于图像使用双线性插值以保持平滑
        img = img.resize((self.img_size, self.img_size))
        # 对于掩码使用最近邻插值以避免引入新的类别值
        mask = mask.resize((self.img_size, self.img_size), resample=Image.NEAREST)
        # 应用图像变换（如果提供）
        if self.transform is not None:
            img = self.transform(img)
        # 处理掩码变换
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        else:
            # 将掩码转换为类别索引
            mask = np.array(mask)
            # 检查掩码的维度，确保是RGB（3通道）
            if len(mask.shape) != 3 or mask.shape[2] != 3:
                raise ValueError(f"掩码维度错误: {mask.shape}, 期望为 (H,W,3)")
            # 创建一个新的类别索引掩码
            mask_copy = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
            # 将RGB颜色映射到类别索引
            # 遍历每种颜色，将对应像素设置为类别索引
            for k, color in enumerate(VOC_COLORMAP):
                # 将每个颜色通道转换为布尔掩码
                r_match = mask[:, :, 0] == color[0]
                g_match = mask[:, :, 1] == color[1]
                b_match = mask[:, :, 2] == color[2]
                # 只有三个通道都匹配的像素才被分配为此类别
                color_match = r_match & g_match & b_match
                mask_copy[color_match] = k
            # 打印类别分布（用于调试）
            # if index == 0:  # 只打印第一个样本的信息
            #     unique_values, counts = np.unique(mask_copy, return_counts=True)
            #     print("掩码中的唯一类别索引:", unique_values)
            #     print("每个类别的像素数量:", counts)
            # 转换为PyTorch张量（长整型，用于交叉熵损失）
            mask = torch.from_numpy(mask_copy).long()
        return img, mask
    
    def __len__(self):
        """返回数据集中的样本数量"""
        return len(self.images)



def get_transforms(train=True):
    """
    获取图像变换函数
    
    参数:
        train (bool): 是否为训练集，决定是否应用数据增强
    
    返回:
        tuple: (图像变换, 目标掩码变换)
    """
    if train:
        # 训练集使用数据增强
        transform = transforms.Compose([
            # 随机水平翻转增加数据多样性
            # transforms.RandomHorizontalFlip(),
            # 随机调整亮度、对比度和饱和度
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            # 转换为张量（值范围变为[0,1]）
            transforms.ToTensor(),
            # 归一化（使用ImageNet的均值和标准差）
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # 验证集只需要基本变换
        transform = transforms.Compose([
            # 转换为张量
            transforms.ToTensor(),
            # 归一化
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # 掩码不需要标准化或者转换为Tensor (已在__getitem__中处理)
    target_transform = None
    
    return transform, target_transform

def get_data_loaders(voc_root, batch_size=4, num_workers=4, img_size=320):
    """
    创建训练和验证数据加载器
    
    参数:
        voc_root (string): VOC数据集根目录
        batch_size (int): 批次大小
        num_workers (int): 数据加载的线程数
        img_size (int): 图像的大小
    
    返回:
        tuple: (train_loader, val_loader) 训练和验证数据加载器
    """
    # 获取图像和掩码变换
    train_transform, train_target_transform = get_transforms(train=True)
    # val_transform, val_target_transform = get_transforms(train=False)
    
    # 创建训练数据集
    train_dataset = VOCSegmentation(
        root=voc_root,
        split='train',  # 使用训练集划分
        transform=train_transform,
        target_transform=train_target_transform,
        img_size=img_size
    )
    
    # 创建验证数据集
    # val_dataset = VOCSegmentation(
    #     root=voc_root,
    #     split='val',  # 使用验证集划分
    #     transform=val_transform,
    #     target_transform=val_target_transform,
    #     img_size=img_size
    # )
    
    # 创建训练数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # 随机打乱数据
        num_workers=num_workers,  # 多线程加载
        pin_memory=True,  # 数据预加载到固定内存，加速GPU传输
        drop_last=True  # 丢弃最后不足一个批次的数据
    )
    
    # 创建验证数据加载器
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,  # 不打乱数据
    #     num_workers=num_workers,
    #     pin_memory=True
    # )
    
    return train_loader

def decode_segmap(segmap):
    """
    将类别索引的分割图转换为RGB彩色图像（用于可视化）
    
    参数:
        segmap (np.array或torch.Tensor): 形状为(H,W)的分割图，值为类别索引
    
    返回:
        rgb_img (np.array): 形状为(H,W,3)的RGB彩色图像
    """
    # 确保segmap是NumPy数组
    if isinstance(segmap, torch.Tensor):
        segmap = segmap.cpu().numpy()
    
    # 检查segmap的形状，处理各种可能的输入格式
    if len(segmap.shape) > 2:
        # 如果是(B,H,W)形状，只取第一个样本
        if len(segmap.shape) == 3 and segmap.shape[0] <= 3:
            segmap = segmap[0]
        else:
            # 如果是(H,W,C)，确保C=1
            if segmap.shape[2] > 1:
                segmap = np.argmax(segmap, axis=2)
            else:
                segmap = segmap[:, :, 0]
    
    # 创建RGB图像
    rgb_img = np.zeros((segmap.shape[0], segmap.shape[1], 3), dtype=np.uint8)
    
    # 根据类别索引填充对应的颜色
    for cls_idx, color in enumerate(VOC_COLORMAP):
        # 找到属于当前类别的像素
        mask = segmap == cls_idx
        if mask.any():  # 只处理存在的类别
            # 将这些像素设置为对应的颜色
            rgb_img[mask] = color
    
    return rgb_img
