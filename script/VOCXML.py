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
        cx /= img_w
        cy /= img_h
        w /= img_w
        h /= img_h
        cx *= 448
        cy *= 448
        objects.append([cx, cy, w, h, cls_id])
    return objects

# [x, y, w, h, conf, cls]
root = "./data/VOC2012/Annotations"
names = os.listdir(root)
for name in  names:
    path = root + "/" + name
    objects = parse_voc_xml(path)
    target = [0 for _ in range(7*7*6)]
    for obj in objects:
        cx = obj[0]
        cy = obj[1]
        w = obj[2]
        h = obj[3]
        cls_id = obj[4]
        scale = 448 / 7
        box_id = int((cy // scale)*7 + (cx // scale))
        print(box_id)
        if target[box_id*6+4] == 1.0:
            continue
        else:
            target[box_id*6] = cx / 448
            target[box_id*6+1] = cy / 448
            target[box_id*6+2] = w
            target[box_id*6+3] = h
            target[box_id*6+4] = 1.0
            target[box_id*6+5] = cls_id
    fp = open("./data/VOC2012/object/"+name.split(".")[0], "w")
    for i in target:
        item = str(i)
        fp.write(item+" ")
    fp.write("\n")
    fp.close()
