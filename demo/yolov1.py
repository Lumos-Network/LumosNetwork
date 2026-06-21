import torch
import torch.nn as nn

S = 7    # 网格
B = 2    # 每个网格2个框
C = 20   # VOC类别
LAMBDA_COORD = 5.0
LAMBDA_NOOBJ = 0.5
EPS = 1e-6

def box_iou(box1, box2):
    # box: [x,y,w,h] 归一化整张图
    b1_x1 = box1[0] - box1[2]/2
    b1_y1 = box1[1] - box1[3]/2
    b1_x2 = box1[0] + box1[2]/2
    b1_y2 = box1[1] + box1[3]/2

    b2_x1 = box2[0] - box2[2]/2
    b2_y1 = box2[1] - box2[3]/2
    b2_x2 = box2[0] + box2[2]/2
    b2_y2 = box2[1] + box2[3]/2

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter = torch.clamp((inter_x2 - inter_x1), min=0) * torch.clamp((inter_y2 - inter_y1), min=0)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - inter
    iou = inter / (union + EPS)
    return iou

def yolov1_loss_darknet_style(pred, target):
    """
    pred: [batch, S*S*(B*5 + C)] 网络原始线性输出，无激活
    target: [batch, S*S*(B*5 + C)] label
    return: total_loss, loss_coord, loss_conf, loss_cls
    """
    bs = pred.shape[0]
    pred = pred.view(bs, S, S, B*5 + C)
    target = target.view(bs, S, S, B*5 + C)

    # 拆分预测：xywh conf | class
    pred_box = pred[..., :B*5].reshape(bs, S, S, B, 5)
    pred_xy = pred_box[..., :2]
    pred_wh = pred_box[..., 2:4]
    pred_conf = pred_box[..., 4]
    pred_cls = pred[..., B*5:]

    # 拆分标签
    tgt_box = target[..., :B*5].reshape(bs, S, S, B, 5)
    tgt_xy = tgt_box[..., :2]
    tgt_wh = tgt_box[..., 2:4]
    tgt_conf = tgt_box[..., 4]
    tgt_cls = target[..., B*5:]

    # ========== 1. 匹配每个GT对应的最佳框，生成obj掩码 ==========
    obj_mask = torch.zeros_like(pred_conf, dtype=torch.bool)
    tgt_conf_fill = torch.zeros_like(pred_conf)  # 匹配框target_conf=IoU

    for b in range(bs):
        for i in range(S):
            for j in range(S):
                # 当前网格真实框（只有一个框有object）
                has_obj = tgt_conf[b,i,j,0] > 0
                if not has_obj:
                    continue
                gt_xywh = torch.cat([tgt_xy[b,i,j,0], tgt_wh[b,i,j,0]])
                box0 = torch.cat([pred_xy[b,i,j,0], pred_wh[b,i,j,0]])
                box1 = torch.cat([pred_xy[b,i,j,1], pred_wh[b,i,j,1]])
                iou0 = box_iou(box0, gt_xywh)
                iou1 = box_iou(box1, gt_xywh)
                best = 0 if iou0 > iou1 else 1
                obj_mask[b,i,j,best] = True
                tgt_conf_fill[b,i,j,best] = torch.max(iou0, iou1)

    noobj_mask = ~obj_mask

    # ========== 2. 坐标损失：对齐Darknet fabs(wh)再sqrt ==========
    pred_wh_abs = torch.abs(pred_wh)
    pred_wh_sqrt = torch.sqrt(torch.clamp(pred_wh_abs, min=EPS))
    tgt_wh_sqrt = torch.sqrt(torch.clamp(tgt_wh, min=EPS))

    dx = pred_xy[obj_mask] - tgt_xy[obj_mask]
    dw = pred_wh_sqrt[obj_mask] - tgt_wh_sqrt[obj_mask]
    loss_xy = torch.sum(dx ** 2)
    loss_wh = torch.sum(dw ** 2)
    loss_coord = LAMBDA_COORD * (loss_xy + loss_wh)

    # ========== 3. 置信度损失 ==========
    loss_conf_obj = torch.sum((pred_conf[obj_mask] - tgt_conf_fill[obj_mask]) ** 2)
    loss_conf_noobj = LAMBDA_NOOBJ * torch.sum((pred_conf[noobj_mask]) ** 2)
    loss_conf = loss_conf_obj + loss_conf_noobj

    # ========== 4. 分类损失：仅存在物体网格 ==========
    grid_obj = torch.any(obj_mask, dim=-1)
    loss_cls = torch.sum((pred_cls[grid_obj] - tgt_cls[grid_obj]) ** 2)

    total = loss_coord + loss_conf + loss_cls
    return total / bs, loss_coord/bs, loss_conf/bs, loss_cls/bs