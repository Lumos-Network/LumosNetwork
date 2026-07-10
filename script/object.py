import cv2
import struct

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

img_w = 768
img_h = 576
img_size = 448
path = "./backup/detect/o_1"
img_path = "./data/VOC2012/dog.jpg"
fp = open(path, "rb")
binary_data = fp.read()
fp.close()
target = []
for i in range(0, len(binary_data), 4):
    val = struct.unpack('f', binary_data[i:i+4])[0]
    target.append(val)
print(target)
img = cv2.imread(img_path)
for i in range(7*7):
    box = target[i*30:(i+1)*30]
    obj1 = box[4]
    obj2 = box[9]
    if obj1 < 0.5 and obj2 < 0.5:
        continue
    pre_box = []
    if obj1 > obj2:
        pre_box = box[:4]
    else:
        pre_box = box[5:9]
    x1 = int(box[0]*7*img_w/img_size)
    y1 = int(box[1]*7*img_h/img_size)
    w = int(box[2]*img_w/img_size)
    h = int(box[2]*img_h/img_size)
    x2 = x1 + w
    y2 = y1 + h
    print("{} {} {} {}".format(x1, y1, x2, y2))
    pre_class = target[10:]
    pro_class = max(pre_class)
    print(pro_class)
    classes = pre_class.index(pro_class)
    print(classes)
    # str_class = VOC_CLASSES[classes]
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
    text_x, text_y = x1, y1 - 10
    # text = str_class+":"+str(pro_class)
    # cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255, 255, 255), thickness=2)
cv2.imwrite("output.png", img)


# 1. 读取图片
# img = cv2.imread("test.jpg")
# if img is None:
#     print("图片读取失败，请检查路径！")
# else:
#     # 框坐标：左上角(50,50)，右下角(300,350)
#     x1, y1 = 50, 50
#     x2, y2 = 300, 350

#     # 2. 画红色矩形框，线条粗细2
#     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)

#     # 3. 在框上方标注文字
#     text = "target object"
#     # 文字放在框左上角上方一点，避免和框重叠
#     text_x, text_y = x1, y1 - 10
#     # 字体、字号、白色文字、粗细2
#     cv2.putText(
#         img,
#         text,
#         (text_x, text_y),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         fontScale=0.7,
#         color=(255, 255, 255),
#         thickness=2
#     )

#     # 显示图片
#     cv2.imshow("draw box and text", img)
#     cv2.waitKey(0)  # 按任意键关闭窗口
#     cv2.destroyAllWindows()

#     # 保存绘制后的图片
#     cv2.imwrite("output.jpg", img)