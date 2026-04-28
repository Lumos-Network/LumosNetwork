import os
import pickle
import struct
from PIL import Image
import logging
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
from itertools import chain

class FCN8(nn.Module): 
    def __init__(self, num_classes):
        # 调用super方法调用父类nn.Module的初始化函数
        super(FCN8, self).__init__()  
        self.l1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, padding=1)
        self.l2 = nn.BatchNorm2d(num_features=96)
        self.l3 = nn.MaxPool2d(2)
        self.l4 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, padding=1)
        self.l5 = nn.BatchNorm2d(num_features=256)
        self.l6 = nn.MaxPool2d(2)
        self.l7 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.l8 = nn.BatchNorm2d(num_features=384)
        self.l9 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.l10 = nn.BatchNorm2d(num_features=384)
        self.l11 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.l12 = nn.BatchNorm2d(num_features=256)
        self.l13 = nn.MaxPool2d(2)
        
        self.l14 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.l15 = nn.BatchNorm2d(num_features=512)
        self.l16 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.l17 = nn.BatchNorm2d(num_features=512)
        self.l18 = nn.MaxPool2d(2)
        
        self.l19 = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=3, padding=1)
        self.l20 = nn.BatchNorm2d(num_features=num_classes)
        self.l21 = nn.MaxPool2d(2)
        
        self.l22 = nn.Conv2d(512 + num_classes + 256, num_classes, kernel_size=7, padding=3)
        
        self.upsample_2 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4, padding= 1, stride=2)
        self.upsample_4 = nn.ConvTranspose2d(in_channels=num_classes, out_channels=num_classes, kernel_size=4, padding= 0,stride=4)
        
        self.upsample_81 = nn.ConvTranspose2d(in_channels=512 + num_classes + 256, out_channels=512 + num_classes + 256, kernel_size=4, padding= 0,stride=4)
        self.upsample_82 = nn.ConvTranspose2d(in_channels=512 + num_classes + 256, out_channels=512 + num_classes + 256, kernel_size=4, padding= 1,stride=2)
    
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = nn.ReLU(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = nn.ReLU(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = nn.ReLU(x)
        x = self.l9(x)
        x = self.l10(x)
        x = nn.ReLU(x)
        x = self.l11(x)
        x = self.l12(x)
        x = nn.ReLU(x)
        x = self.l13(x)
        
        pool3 = x
        
        x = self.l14(x)
        x = self.l15(x)
        x = nn.ReLU(x)
        x = self.l16(x)
        x = self.l17(x)
        x = nn.ReLU(x)
        x = self.l18(x)
        
        pool4 = self.upsample_2(x)
        
        x = self.l19(x)
        x = self.l20(x)
        x = nn.ReLU(x)
        x = self.l21(x)
        
        conv7 = self.upsample_4(x)
        
        x = torch.cat([pool3, pool4, conv7], dim = 1)
        
        output = self.upsample_81(x)
        output = self.upsample_82(output)
        output = self.l22(output)
