import cv2
import struct
import torch
import torch.nn as nn
import numpy as np

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def load_bin_3d(file_path: str, shape_3d: tuple, dtype=np.float32) -> torch.Tensor:
    """
    读取行优先平铺存储的二进制三维浮点数据
    :param file_path: 二进制文件路径
    :param shape_3d: 三维元组 (dim0, dim1, dim2)
    :param dtype: 二进制数据类型，默认float32
    :return: torch三维张量
    """
    with open(file_path, "rb") as f:
        data_bytes = f.read()
    # 字节转一维numpy数组
    arr_1d = np.frombuffer(data_bytes, dtype=dtype)
    # 还原三维shape（行优先reshape）
    arr_3d = arr_1d.reshape(shape_3d)
    # 转为torch浮点张量
    tensor_3d = torch.from_numpy(arr_3d).float()
    return tensor_3d

def plot_bbox_labels(img, bbox, label=None, cls_color=None, text_scale=0.4):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
    # plot bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), cls_color, 2)
    
    if label is not None:
        # plot title bbox
        cv2.rectangle(img, (x1, y1-t_size[1]), (int(x1 + t_size[0] * text_scale), y1), cls_color, -1)
        # put the test on the title bbox
        cv2.putText(img, label, (int(x1), int(y1 - 5)), 0, text_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    return img


def visualize(img, 
              bboxes, 
              scores, 
              labels, 
              vis_thresh, 
              class_colors, 
              class_names, 
              class_indexs=None, 
              dataset_name='voc'):
    ts = 0.4
    for i, bbox in enumerate(bboxes):
        print(scores)
        if scores[i] > vis_thresh:
            cls_id = int(labels[i])
            if dataset_name == 'coco-val':
                cls_color = class_colors[cls_id]
                cls_id = class_indexs[cls_id]
            else:
                cls_color = class_colors[cls_id]
                
            if len(class_names) > 1:
                mess = '%s: %.2f' % (class_names[cls_id], scores[i])
            else:
                cls_color = [255, 0, 0]
                mess = None
            img = plot_bbox_labels(img, bbox, mess, cls_color, text_scale=ts)

    return img

def create_grid(input_size):
    """ 
        用于生成G矩阵，其中每个元素都是特征图上的像素坐标。
    """
    # 输入图像的宽和高
    w, h = input_size, input_size
    # 特征图的宽和高
    ws, hs = w // 32, h // 32
    # 生成网格的x坐标和y坐标
    grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])

    # 将xy两部分的坐标拼起来：[H, W, 2]
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
    # [H, W, 2] -> [HW, 2] -> [HW, 2]
    grid_xy = grid_xy.view(-1, 2)
    
    return grid_xy

def decode_boxes(pred):
    """
        将txtytwth转换为常用的x1y1x2y2形式。
    """
    output = torch.zeros_like(pred)
    # 得到所有bbox 的中心点坐标和宽高
    pred[..., :2] = torch.sigmoid(pred[..., :2]) + create_grid(416)
    pred[..., 2:] = torch.exp(pred[..., 2:])

    # 将所有bbox的中心带你坐标和宽高换算成x1y1x2y2形式
    output[..., :2] = pred[..., :2] * 32 - pred[..., 2:] * 0.5
    output[..., 2:] = pred[..., :2] * 32 + pred[..., 2:] * 0.5
    
    return output


def nms(bboxes, scores):
    """"Pure Python NMS baseline."""
    x1 = bboxes[:, 0]  #xmin
    y1 = bboxes[:, 1]  #ymin
    x2 = bboxes[:, 2]  #xmax
    y2 = bboxes[:, 3]  #ymax

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []                                             
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # 计算交集的左上角点和右下角点的坐标
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # 计算交集的宽高
        w = np.maximum(1e-10, xx2 - xx1)
        h = np.maximum(1e-10, yy2 - yy1)
        # 计算交集的面积
        inter = w * h

        # 计算交并比
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # 滤除超过nms阈值的检测框
        inds = np.where(iou <= 0.5)[0]
        order = order[inds + 1]

    return keep


def postprocess(bboxes, scores):
    """
    Input:
        bboxes: [HxW, 4]
        scores: [HxW, num_classes]
    Output:
        bboxes: [N, 4]
        score:  [N,]
        labels: [N,]
    """

    labels = np.argmax(scores, axis=1)
    scores = scores[(np.arange(scores.shape[0]), labels)]
    # threshold
    keep = np.where(scores >= 0.001)
    bboxes = bboxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    print(scores)

    # NMS
    keep = np.zeros(len(bboxes), dtype=np.int16)
    for i in range(20):
        inds = np.where(labels == i)[0]
        if len(inds) == 0:
            continue
        c_bboxes = bboxes[inds]
        c_scores = scores[inds]
        c_keep = nms(c_bboxes, c_scores)
        keep[inds[c_keep]] = 1

    keep = np.where(keep > 0)
    bboxes = bboxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    return bboxes, scores, labels

img_size = 416
path = "./backup/detect/o_1"
img_path = "./data/VOC2012/JPEGImages/2009_002988.jpg"
img = cv2.imread(img_path)
img_h, img_w, img_c = img.shape

pred = load_bin_3d(path, (1, 25, 169)).reshape((1,25,13,13))
pred = pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)

conf_pred = pred[..., :1]
cls_pred = pred[..., 1:1+20]
txtytwth_pred = pred[..., 1+20:]

conf_pred = conf_pred[0]
cls_pred = cls_pred[0]
txtytwth_pred = txtytwth_pred[0]
scores = torch.sigmoid(conf_pred) * torch.softmax(cls_pred, dim=-1)

bboxes = decode_boxes(txtytwth_pred) / img_size

bboxes = torch.clamp(bboxes, 0., 1.)

scores = scores.numpy()
bboxes = bboxes.numpy()

bboxes, scores, labels = postprocess(bboxes, scores)

scale = np.array([[img_w, img_h, img_w, img_h]])
bboxes *= scale
np.random.seed(0)
class_colors = [(np.random.randint(255),
                    np.random.randint(255),
                    np.random.randint(255)) for _ in range(20)]

# 可视化检测结果
img_processed = visualize(
    img=img,
    bboxes=bboxes,
    scores=scores,
    labels=labels,
    vis_thresh=0.1,
    class_colors=class_colors,
    class_names=VOC_CLASSES,
    class_indexs=None,
    dataset_name=None
    )
cv2.imshow('detection', img_processed)
cv2.waitKey(0)

cv2.imwrite("./yolo-5.jpg", img_processed)
