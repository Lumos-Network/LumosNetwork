from PIL import Image
import numpy as np
import torch

# VOC_COLORMAP = [
#     [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
#     [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
#     [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
#     [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
#     [0, 64, 128]
# ]

# path = "./backup/detect/3.png"
# img = Image.open(path)
# segmap = np.array(img)
# if isinstance(segmap, torch.Tensor):
#     segmap = segmap.cpu().numpy()

# # 检查segmap的形状，处理各种可能的输入格式
# if len(segmap.shape) > 2:
#     # 如果是(B,H,W)形状，只取第一个样本
#     if len(segmap.shape) == 3 and segmap.shape[0] <= 3:
#         segmap = segmap[0]
#     else:
#         # 如果是(H,W,C)，确保C=1
#         if segmap.shape[2] > 1:
#             segmap = np.argmax(segmap, axis=2)
#         else:
#             segmap = segmap[:, :, 0]

# # 创建RGB图像
# rgb_img = np.zeros((segmap.shape[0], segmap.shape[1], 3), dtype=np.uint8)

# # 根据类别索引填充对应的颜色
# for cls_idx, color in enumerate(VOC_COLORMAP):
#     # 找到属于当前类别的像素
#     mask = segmap == cls_idx
#     if mask.any():  # 只处理存在的类别
#         # 将这些像素设置为对应的颜色
#         rgb_img[mask] = color

# new = Image.fromarray(rgb_img)
# new.save("./backup/01.png")

path = "./backup/detect/1.png"
img = Image.open(path)
data = np.array(img)
data = data*255;
new = Image.fromarray(data)
new.save("./backup/01.png")