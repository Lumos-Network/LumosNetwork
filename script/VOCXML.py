import xml.etree.ElementTree as ET
import torch
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
cls_to_idx = {cls:i for i, cls in enumerate(VOC_CLASSES)}

def parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_w = int(root.find("size/width").text)
    img_h = int(root.find("size/height").text)
    objects = []
    for obj in root.findall("object"):
        cls_name = obj.find("name").text # 类别名
        cls_id = cls_to_idx[cls_name]    # 类别索引
        bnd = obj.find("bndbox")         # 坐标
        x1 = float(bnd.find("xmin").text)
        y1 = float(bnd.find("ymin").text)
        x2 = float(bnd.find("xmax").text)
        y2 = float(bnd.find("ymax").text)
        # 计算中心、宽高（像素）
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        objects.append([cx, cy, w, h, cls_id])
    return objects, img_w, img_h

def voc_to_yolo_target(objects, orig_w, orig_h, img_size=448, S=7, B=2, C=20):
    # 初始化全0标签
    target = torch.zeros((S, S, B * 5 + C))
    cell_size = img_size / S

    for (cx, cy, w, h, cls_id) in objects:
        # 1. 坐标映射到448尺寸（原图拉伸到448）
        scale_x = img_size / orig_w
        scale_y = img_size / orig_h
        cx_scaled = cx * scale_x
        cy_scaled = cy * scale_y
        w_scaled = w * scale_x
        h_scaled = h * scale_y

        # 2. 计算网格坐标 i(col), j(row)
        grid_i = int(cx_scaled / cell_size)
        grid_j = int(cy_scaled / cell_size)
        # 防止越界
        grid_i = torch.clamp(torch.tensor(grid_i), 0, S-1).item()
        grid_j = torch.clamp(torch.tensor(grid_j), 0, S-1).item()

        # 同一网格已有物体则跳过（YOLOv1原生限制：一格一物）
        if target[grid_j, grid_i, 4] == 1.0:
            continue

        # 3. 填充相对网格的 x,y
        x_cell = (cx_scaled / cell_size) - grid_i
        y_cell = (cy_scaled / cell_size) - grid_j
        # 4. 填充相对整张图的 w,h
        w_norm = w_scaled / img_size
        h_norm = h_scaled / img_size

        # 写入第一个框的 x,y,w,h,conf
        target[grid_j, grid_i, 0] = x_cell
        target[grid_j, grid_i, 1] = y_cell
        target[grid_j, grid_i, 2] = w_norm
        target[grid_j, grid_i, 3] = h_norm
        target[grid_j, grid_i, 4] = 1.0  # obj置信度=1

        # one-hot类别
        target[grid_j, grid_i, 10 + cls_id] = 1.0
    return target

class VOCDataset(Dataset):
    def __init__(self, root, img_size=448, S=7, B=2, C=20):
        self.root = root
        self.img_dir = os.path.join(root, "JPEGImages")
        self.ann_dir = os.path.join(root, "Annotations")
        split_path = os.path.join(root, "ImageSets/Main/trainval.txt")
        with open(split_path, "r", encoding="utf-8") as f:
            self.img_ids = [line.strip() for line in f.readlines()]
        
        self.img_size = img_size
        self.S = S
        self.B = B
        self.C = C
        # YOLOv1 原图转448预处理
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        # 读取图片
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        img = Image.open(img_path).convert("RGB")
        # 读取标注
        xml_path = os.path.join(self.ann_dir, f"{img_id}.xml")
        objects, orig_w, orig_h = parse_voc_xml(xml_path)
        # 生成7x7x30标签
        target = voc_to_yolo_target(objects, orig_w, orig_h, self.img_size, self.S, self.B, self.C)
        # 图像预处理
        img = self.transform(img)
        return img, target
