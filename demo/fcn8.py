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
from torchvision.models import vgg16

# VOC_COLORMAP = [
#     [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
#     [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
#     [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
#     [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
#     [0, 64, 128]
# ]

# def forward_hook(module, input, output):
#     flag = "input"
#     shape = None
#     data = None
#     if (flag == "input"):
#         shape = input[0].shape
#         data = input[0].tolist()
#     else:
#         shape = output.shape
#         data = output.tolist()
#     print("OutShape: {}".format(shape))
#     data_f = []
#     with open("./backup/in_py", "wb") as fp:
#         for i in range(shape[0]):
#             for j in range(shape[1]):
#                 for k in range(shape[2]):
#                     data_f += data[i][j][k]
#         for i in range(len(data_f)):
#             fp.write(struct.pack('i', int(data_f[i])))
#         fp.close()

# # grad_input 中各部分梯度的‌顺序取决于模块 forward 函数的输入顺序‌：
# # ‌nn.Linear‌（有 bias）：
# # grad_input 顺序：[bias_grad, input_grad, weight_grad]
# # ‌nn.Conv2d‌（有 bias）：
# # grad_input 顺序：[input_grad, weight_grad, bias_grad]
# # ‌nn.ReLU‌（无参数）：
# # grad_input 仅包含：[input_grad]
# def backward_hook(module, grad_input, grad_output):
#     # print("gradshape:{}".format(grad_input[0].shape))
#     input_grad = grad_input[1]
#     shape = input_grad.shape
#     print(shape)
#     grad = input_grad.tolist()
#     # print(grad)
#     data = []
#     with open("./backup/grad_py", "wb") as fp:
#         for i in range(shape[0]):
#             for j in range(shape[1]):
#                 for k in range(shape[2]):
#                     data += grad[i][j][k]
#         for i in range(len(data)):
#             fp.write(struct.pack('f', data[i]))
#         fp.close()

# class MyDataset(Dataset):
#     def __init__(self, txt_path, transform = None, target_transform = None):
#         fh = open(txt_path, 'r')
#         imgs = []
#         for line in fh:
#             pathl = "./data/VOC2012/SegmentationClass/"+line.strip().split('/')[-1]
#             imgs.append((line.strip(), pathl)) # 类别转为整型int
#             self.imgs = imgs 
#             self.transform = transform
#             self.target_transform = target_transform

#     def __getitem__(self, index):
#         pathi, pathl = self.imgs[index]
#         img = Image.open(pathi).convert('RGB') 
#         if self.transform is not None:
#             img = self.transform(img)
#         label = Image.open(pathl).convert('RGB')
#         label = label.resize((320, 320), resample=Image.NEAREST)
#         if self.target_transform is not None:
#             label = self.target_transform(label)
#         else:
#             label = np.array(label)
#             if len(label.shape) != 3 or label.shape[2] != 3:
#                 raise ValueError(f"掩码维度错误: {label.shape}, 期望为 (H,W,3)")
#             label_copy = np.zeros((320, 320), dtype=np.uint8)
#             for k, color in enumerate(VOC_COLORMAP):
#                 r_match = label[:, :, 0] == color[0]
#                 g_match = label[:, :, 1] == color[1]
#                 b_match = label[:, :, 2] == color[2]
#                 color_match = r_match & g_match & b_match
#                 label_copy[color_match] = k
#             label = torch.from_numpy(label_copy).long()
#         return img, label

#     def __len__(self):
#         return len(self.imgs)

# class FCN8(nn.Module):
#     def __init__(self, num_classes=1000):
#         super(FCN8, self).__init__()
#         self.l1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
#         self.l2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.l3 = nn.MaxPool2d(kernel_size=2, stride=2)

#         # Block 2
#         self.l4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.l5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
#         self.l6 = nn.MaxPool2d(kernel_size=2, stride=2)

#         # Block 3
#         self.l7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.l8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.l9 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.l10 = nn.MaxPool2d(kernel_size=2, stride=2)

#         # Block 4
#         self.l11 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
#         self.l12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.l13 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.l14 = nn.MaxPool2d(kernel_size=2, stride=2)

#         # Block 5
#         self.l15 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.l16 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.l17 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.l18 = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.l19 = nn.Conv2d(512, 1024, 7, 1, 3) # fc6
#         # self.l20 = nn.Dropout(0.3)
#         self.l21 = nn.Conv2d(1024, 1024, 1, 1) #fc7
#         # self.l22 = nn.Dropout(0.3)
        
#         self.l23 = nn.Conv2d(1024, num_classes, 1, 1)
#         self.l24 = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        
#         self.l25 = nn.Conv2d(512, num_classes, 1, 1)
#         self.l26 = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        
#         self.l27 = nn.Conv2d(256, num_classes, 1, 1)
#         self.l28 = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, bias=False)
        
#         self.l29 = nn.CrossEntropyLoss(ignore_index=255)
#         self._initialize_weights()

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.ConvTranspose2d):
#                 # 双线性上采样的初始化
#                 m.weight.data.zero_()
#                 m.weight.data = self._make_bilinear_weights(m.kernel_size[0], m.out_channels)
    
#     def _make_bilinear_weights(self, size, num_channels):
#         """生成双线性插值的权重"""
#         factor = (size + 1) // 2
#         if size % 2 == 1:
#             center = factor - 1
#         else:
#             center = factor - 0.5
#         og = torch.FloatTensor(size, size)
#         for i in range(size):
#             for j in range(size):
#                 og[i, j] = (1 - abs((i - center) / factor)) * (1 - abs((j - center) / factor))
#         filter = torch.zeros(num_channels, num_channels, size, size)
#         for i in range(num_channels):
#             filter[i, i] = og
#         return filter

#     def forward(self, x, labels):
#         x = self.l1(x)
#         x = torch.relu(x)
#         x = self.l2(x)
#         x = torch.relu(x)
#         x = self.l3(x)
        
#         x = self.l4(x)
#         x = torch.relu(x)
#         x = self.l5(x)
#         x = torch.relu(x)
#         x = self.l6(x)
        
#         x = self.l7(x)
#         x = torch.relu(x)
#         x = self.l8(x)
#         x = torch.relu(x)
#         x = self.l9(x)
#         x = torch.relu(x)
#         pool3 = self.l10(x) # pool3
        
#         x = self.l11(pool3)
#         x = torch.relu(x)
#         x = self.l12(x)
#         x = torch.relu(x)
#         x = self.l13(x)
#         x = torch.relu(x)
#         pool4 = self.l14(x) # pool4
        
#         x = self.l15(pool4)
#         x = torch.relu(x)
#         x = self.l16(x)
#         x = torch.relu(x)
#         x = self.l17(x)
#         x = torch.relu(x)
#         pool5 = self.l18(x) # pool5
        
#         x = self.l19(pool5) # fc6
#         x = torch.relu(x)
#         # x = self.l20(x)
#         x = self.l21(x) # fc7
#         x = torch.relu(x)
#         # x = self.l22(x)

#         x = self.l23(x)
#         up2 = self.l24(x)

#         x = self.l25(pool4)
#         x = x + up2
        
#         up4 = self.l26(x)
        
#         x = self.l27(pool3)
#         x = x + up4
        
#         x = self.l28(x)
#         x = self.l29(x, labels)
#         return x

# data_transform = transforms.Compose([
#     # transforms.Resize((320,320)),  # 随机裁剪成224×224
#     transforms.ToTensor(),  # 转换成Tensor（PyTorch能处理的数据格式）
#     # 标准化：让数据更符合模型训练要求，数值范围更稳定
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# num_epochs = 1
# batch_size = 4
# train_data = MyDataset('./data/VOCT/train.txt', transform=data_transform)
# trainloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)

# model = FCN8(num_classes=21)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# fp = open("./backup/LW_py", "wb")
# for name, param in model.named_parameters():
#     # print(f"Layer: {name}, Parameter Shape: {param.shape}") #param.shape [filters, channels, ksize, ksize]  [outputs, inputs]
#     data_f = []
#     if "l24" not in name:
#         continue
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
#     for i in range(len(data_f)):
#         fp.write(struct.pack('f', data_f[i]))
# fp.close()

# model.l2.register_forward_hook(forward_hook)
# # model.l28.register_backward_hook(backward_hook)

# device = torch.device("cpu")
# model.to(device)

# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0  # 记录每轮的损失
#     num = 0
#     for i, data in enumerate(trainloader, 0):
#         # 获取输入：图片（inputs）和对应的标签（labels，比如0代表玫瑰，1代表郁金香）
#         # print(data[0])
#         inputs, labels = data[0].to(device), data[1].to(device)
#         optimizer.zero_grad()
#         outputs = model(inputs, labels)
#         outputs.backward()
#         optimizer.step()
#         running_loss += outputs.item()
#         print("loss:{}".format(outputs.item()))
#         num += 1
#     print("AVGloss:{}".format(running_loss/num))

# # fp = open("./backup/LW_fp", "wb")
# # for name, param in model.named_parameters():
# #     # print(f"Layer: {name}, Parameter Shape: {param.shape}") #param.shape [filters, channels, ksize, ksize]  [outputs, inputs]
# #     data_f = []
# #     if ("weight" in name and len(param.shape) == 4):
# #         data = param.tolist()
# #         for i in range(param.shape[0]):
# #             for j in range(param.shape[1]):
# #                 for k in range(param.shape[2]):
# #                     data_f += data[i][j][k]
# #     if ("weight" in name and len(param.shape) == 2):
# #         data = param.tolist()
# #         for i in range(param.shape[0]):
# #             data_f += data[i]
# #     if ("weight" in name and len(param.shape) == 1):
# #         data_f += param.tolist()
# #     if ("bias" in name):
# #         data = param.tolist()
# #         data_f += data
# #     for i in range(len(data_f)):
# #         fp.write(struct.pack('f', data_f[i]))
# # fp.close()

class FCN8s(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        # 加载预训练 VGG16 作为 Backbone
        vgg = vgg16(pretrained=True)
        features = list(vgg.features)
        self.features = nn.Sequential(*features)

        # 替换全连接为卷积
        self.conv6 = nn.Conv2d(512, 4096, 7, padding=3)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d()
        self.conv7 = nn.Conv2d(4096, 4096, 1, padding=0)

        # 1x1 卷积生成得分图
        self.score_pool3 = nn.Conv2d(256, num_classes, 1)
        self.score_pool4 = nn.Conv2d(512, num_classes, 1)
        self.score_conv7 = nn.Conv2d(4096, num_classes, 1)

        # 反卷积上采样
        self.upsample2 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2, padding=1)
        self.upsample8 = nn.ConvTranspose2d(num_classes, num_classes, 16, stride=8, padding=4)

    def forward(self, x):
        # 抽取 VGG 中间层
        pool3_out = self.features[:17](x)   # 1/8
        pool4_out = self.features[:24](x)   # 1/16
        pool5_out = self.features(x)         # 1/32

        # 全卷积部分
        x = self.relu(self.conv6(pool5_out))
        x = self.dropout(x)
        x = self.relu(self.conv7(x))
        x = self.dropout(x)

        # 跳跃连接融合
        score7 = self.score_conv7(x)
        score4 = self.score_pool4(pool4_out)
        score3 = self.score_pool3(pool3_out)

        up_score7 = self.upsample2(score7)
        fuse1 = up_score7 + score4

        up_fuse1 = self.upsample2(fuse1)
        fuse2 = up_fuse1 + score3

        out = self.upsample8(fuse2)
        return out

model = FCN8s(21)

fp = open("./backup/LW_fp", "wb")
for name, param in model.named_parameters():
    # print(f"Layer: {name}, Parameter Shape: {param.shape}") #param.shape [filters, channels, ksize, ksize]  [outputs, inputs]
    if "features" not in name:
        continue
    data_f = []
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
    for i in range(len(data_f)):
        fp.write(struct.pack('f', data_f[i]))
fp.close()
