import torch.nn as nn
import torch
import struct
import numpy as np

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

class MSEWithLogitsLoss(nn.Module):
    def __init__(self, ):
        super(MSEWithLogitsLoss, self).__init__()

    def forward(self, logits, targets):
        inputs = torch.clamp(torch.sigmoid(logits), min=1e-4, max=1.0 - 1e-4)
        pos_id = (targets==1.0).float()
        neg_id = (targets==0.0).float()
        pos_loss = pos_id * (inputs - targets)**2
        neg_loss = neg_id * (inputs)**2
        loss = 5.0*pos_loss + 1.0*neg_loss
        return loss

# ========== 1. 配置你的三维维度 ==========
PRED_CONF_BIN_PATH = "./backup/pred_conf"
PRED_CLASS_BIN_PATH = "./backup/pred_class"
PRED_XY_BIN_PATH = "./backup/pred_txty"
PRED_WH_BIN_PATH = "./backup/pred_twth"

GT_CONF_BIN_PATH = "./backup/target_conf"
GT_CLASS_BIN_PATH = "./backup/target_class"
GT_XY_BIN_PATH = "./backup/target_txty"
GT_WH_BIN_PATH = "./backup/target_twth"
GT_BOX_BIN_PATH = "./backup/target_box"

# ========== 2. 加载二进制三维数据 ==========
pred_conf = load_bin_3d(PRED_CONF_BIN_PATH, (4, 169, 1))
pred_cls = load_bin_3d(PRED_CLASS_BIN_PATH, (4, 20, 169))
pred_txty = load_bin_3d(PRED_XY_BIN_PATH, (4, 169, 2))
pred_twth = load_bin_3d(PRED_WH_BIN_PATH, (4, 169, 2))

gt_obj = load_bin_3d(GT_CONF_BIN_PATH, (4, 169, 1))
gt_cls = load_bin_3d(GT_CLASS_BIN_PATH, (4, 169, 1))
gt_txty = load_bin_3d(GT_XY_BIN_PATH, (4, 169, 2))
gt_twth = load_bin_3d(GT_WH_BIN_PATH, (4, 169, 2))
gt_box_scale_weight = load_bin_3d(GT_BOX_BIN_PATH, (4, 169, 1))

# logits需要计算梯度，开启requires_grad
pred_conf.requires_grad = True
pred_cls.requires_grad = True
pred_txty.requires_grad = True
pred_twth.requires_grad = True

# 损失函数
conf_loss_function = MSEWithLogitsLoss()
cls_loss_function = nn.CrossEntropyLoss(reduction='none')
txty_loss_function = nn.BCEWithLogitsLoss(reduction='none')
twth_loss_function = nn.MSELoss(reduction='none')

pred_conf = pred_conf.reshape((4, 169))
gt_cls = gt_cls.reshape((4, 169)).long()
gt_obj = gt_obj.reshape((4, 169))
gt_box_scale_weight = gt_box_scale_weight.reshape((4, 169))

# 置信度损失
conf_loss = conf_loss_function(pred_conf, gt_obj)
conf_loss = conf_loss.sum() / 4
print(conf_loss)

# 类别损失
cls_loss = cls_loss_function(pred_cls, gt_cls) * gt_obj
cls_loss = cls_loss.sum() / 4
print(cls_loss)

# 边界框txty的损失
txty_loss = txty_loss_function(pred_txty, gt_txty).sum(-1) * gt_obj * gt_box_scale_weight
txty_loss.reshape((4, 169, 1))
txty_loss = txty_loss.sum() / 4
print(txty_loss)

# 边界框twth的损失
twth_loss = twth_loss_function(pred_twth, gt_twth).sum(-1) * gt_obj * gt_box_scale_weight
twth_loss = twth_loss.sum() / 4
print(twth_loss)
bbox_loss = txty_loss + twth_loss
print(bbox_loss)

# 总的损失
total_loss = conf_loss + cls_loss + bbox_loss
print(total_loss/8)

# # ========== 3. 构造损失并前向计算 ==========
# loss_fn = nn.MSELoss(reduction="none")
# loss = loss_fn(logits, target).sum(-1)
# loss = loss.reshape((4, 169, 1))
# loss *= gt_obj * box_scale
# loss = loss.sum() / 4
# print(loss)
# # print("Loss Value: ", loss.item())

# # ========== 4. 反向传播求梯度 ==========
# loss.backward()
# grad_pytorch = logits.grad.cpu()
# grad_data = np.array(grad_pytorch).tolist()
# print(grad_pytorch.shape)
# data = []
# ff = open("/home/libian/LumosNetwork/backup/grad_py", "wb")
# for grad_batch in grad_data:
#     for item in grad_batch:
#         for x in item:
#             ff.write(struct.pack('f', x))
# ff.close()

# # ========== 5. 手动公式验算梯度（和框架结果对比） ==========
# sigmoid_z = torch.sigmoid(logits.detach())
# grad_raw = sigmoid_z - target
# elem_total = logits.numel()
# grad_manual = grad_raw / elem_total  # reduction=mean 归一化

# print("\nPyTorch 自动梯度：\n", grad_pytorch)
# print("\n公式手动梯度：\n", grad_manual)
# print("\n最大误差（理想应≈0）：", torch.max(torch.abs(grad_pytorch - grad_manual)).item())
