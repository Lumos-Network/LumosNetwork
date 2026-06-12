import struct
import torch
import torch.nn as nn
from torchvision import models

class DeepLabv2(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        features = list(vgg16.features.children())
        self.features = nn.Sequential(*features[:31])


# 加载预训练 ResNet50
model = DeepLabv2(num_classes=21)
fp = open("./backup/LW_deeplabv2", "wb")
data_f = []
for name, param in model.named_parameters():
    print(f"Layer: {name}, Parameter Shape: {param.shape}") #param.shape [filters, channels, ksize, ksize]  [outputs, inputs]
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

