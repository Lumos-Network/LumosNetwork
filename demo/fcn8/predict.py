import os
import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

from dataload import VOC_CLASSES, VOC_COLORMAP, NUM_CLASSES, decode_segmap
from fcn_model import get_fcn_model

def parse_args():
    parser = argparse.ArgumentParser(description='FCN语义分割模型预测')
    parser.add_argument('--model-path', type=str, required=True,
                        help='预训练模型路径')
    parser.add_argument('--model-type', type=str, default='fcn8s', choices=['fcn8s', 'fcn16s', 'fcn32s'],
                        help='FCN模型类型 (fcn8s, fcn16s, fcn32s)')
    parser.add_argument('--image-path', type=str, required=True,
                        help='输入图像路径，可以是单个图像或者目录')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='结果保存目录')
    parser.add_argument('--overlay', action='store_true',
                        help='是否将分割结果与原图叠加')
    parser.add_argument('--no-cuda', action='store_true',
                        help='禁用CUDA')
    return parser.parse_args()

def preprocess_image(image_path):
    """预处理输入图像"""
    image = Image.open(image_path).convert('RGB')
    
    # 图像预处理变换
    transform = transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(320),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)  # 增加批次维度
    
    return input_batch, image

def overlay_segmentation(image, segmentation, alpha=0.7):
    """将分割结果与原图叠加"""
    # 将PIL图像转换为NumPy数组
    image_np = np.array(image)
    
    # 调整分割图大小以匹配原图
    segmentation_resized = np.array(Image.fromarray(segmentation.astype(np.uint8)).resize(
        (image_np.shape[1], image_np.shape[0]), Image.NEAREST))
    
    # 创建叠加图像
    overlay = image_np.copy()
    for i in range(3):
        overlay[:, :, i] = image_np[:, :, i] * (1 - alpha) + segmentation_resized[:, :, i] * alpha
    
    return overlay.astype(np.uint8)

def predict_image(model, image_path, device, overlay=False):
    """对单个图像进行预测"""
    # 预处理图像
    input_batch, original_image = preprocess_image(image_path)
    input_batch = input_batch.to(device)
    
    # 模型预测
    with torch.no_grad():
        output = model(input_batch)
        output = torch.nn.functional.softmax(output, dim=1)
        _, pred = torch.max(output, 1)
        pred = pred.cpu().numpy()[0]  # 取第一个样本（因为只有一个）
    
    # 将预测结果转换为彩色分割图
    segmentation_map = decode_segmap(pred)
    
    # 如果需要与原图叠加
    if overlay:
        result = overlay_segmentation(original_image, segmentation_map)
    else:
        result = segmentation_map
    
    return result, pred, original_image

def predict_and_visualize(model, image_path, output_dir, device, overlay=False):
    """预测图像并可视化结果"""
    # 如果图像路径是目录
    if os.path.isdir(image_path):
        image_files = [f for f in os.listdir(image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for image_file in image_files:
            file_path = os.path.join(image_path, image_file)
            visualize_prediction(model, file_path, output_dir, device, overlay)
    else:
        # 单个图像
        visualize_prediction(model, image_path, output_dir, device, overlay)

def visualize_prediction(model, image_path, output_dir, device, overlay=False):
    """可视化单个图像的预测结果"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 预测图像
    result, pred, original_image = predict_image(model, image_path, device, overlay)
    
    # 保存结果
    base_name = os.path.basename(image_path).split('.')[0]
    
    plt.figure(figsize=(15, 5))
    
    # 原始图像
    plt.subplot(1, 3, 1)
    plt.title('original_image')
    plt.imshow(original_image)
    plt.axis('off')
    
    # 分割图
    plt.subplot(1, 3, 2)
    plt.title('pred')
    plt.imshow(decode_segmap(pred))
    plt.axis('off')
    
    # 叠加或分割图
    plt.subplot(1, 3, 3)
    if overlay:
        plt.title('pred and original')
    else:
        plt.title('pred and original')
    plt.imshow(result)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{base_name}_result.png'))
    plt.close()
    
    # 统计各类别像素数量
    class_pixels = {}
    for i, class_name in enumerate(VOC_CLASSES):
        num_pixels = np.sum(pred == i)
        if num_pixels > 0:
            class_pixels[class_name] = num_pixels
    
    # 创建类别分布饼图
    if class_pixels:
        plt.figure(figsize=(10, 10))
        labels = list(class_pixels.keys())
        sizes = list(class_pixels.values())
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
        plt.axis('equal')
        plt.title('类别分布')
        plt.savefig(os.path.join(output_dir, f'{base_name}_class_dist.png'))
        plt.close()
    
    # 保存单独的分割图
    segmentation_img = Image.fromarray(decode_segmap(pred))
    segmentation_img.save(os.path.join(output_dir, f'{base_name}_segmentation.png'))

def main():
    # 解析参数
    args = parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建模型
    model = get_fcn_model(model_type=args.model_type, num_classes=NUM_CLASSES, pretrained=False)
    
    # 加载预训练权重（使用weights_only=False解决PyTorch 2.6的兼容性问题）
    try:
        print(f'尝试加载模型权重: {args.model_path}')
        # 先尝试新方法加载
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
        print('成功使用weights_only=False加载模型')
    except (TypeError, ValueError):
        # 如果报错，尝试旧版本方法加载
        print('尝试使用旧版本方法加载模型...')
        checkpoint = torch.load(args.model_path, map_location=device)
    
    # 加载模型权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'加载检查点: Epoch {checkpoint["epoch"]}, mIoU {checkpoint["best_miou"]:.4f}')
    else:
        model.load_state_dict(checkpoint)
        print(f'加载模型权重成功')
    
    model = model.to(device)
    model.eval()
    
    # 预测和可视化
    predict_and_visualize(model, args.image_path, args.output_dir, device, args.overlay)
    print(f'结果已保存到: {args.output_dir}')

if __name__ == '__main__':
    main() 