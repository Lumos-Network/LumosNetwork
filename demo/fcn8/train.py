import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc  
from dataload import get_data_loaders, NUM_CLASSES, decode_segmap
from fcn_model import get_fcn_model

def parse_args():
    parser = argparse.ArgumentParser(description='FCN 语义分割 PyTorch 实现')
    parser.add_argument('--voc-root', type=str, default='./data/VOC2012',
                        help='VOC数据集根目录')
    parser.add_argument('--model-type', type=str, default='fcn8s', choices=['fcn8s', 'fcn16s', 'fcn32s'],
                        help='FCN模型类型 (fcn8s, fcn16s, fcn32s)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='训练的批次大小')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练的轮数')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='学习率')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD动量')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='模型保存目录')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    return parser.parse_args()

# 评估函数
def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_corrects = 0
    total_pixels = 0
    class_iou = np.zeros(NUM_CLASSES)
    class_pixels = np.zeros(NUM_CLASSES)
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc='Evaluation'):
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item() * images.size(0)
            
            _, preds = torch.max(outputs, 1)
            
            # 计算像素准确率
            correct = (preds == targets).sum().item()
            total_corrects += correct
            total_pixels += targets.numel()
            
            # 计算每个类别的IoU
            for cls in range(NUM_CLASSES):
                pred_inds = preds == cls
                target_inds = targets == cls
                intersection = (pred_inds & target_inds).sum().item()
                union = (pred_inds | target_inds).sum().item()
                
                if union > 0:
                    class_iou[cls] += intersection / union
                    class_pixels[cls] += 1
            
            del images, targets, outputs, preds
            torch.cuda.empty_cache()  
    
    # 计算平均指标
    val_dataset_size = len(val_loader.dataset) if hasattr(val_loader.dataset, '__len__') else len(val_loader) * val_loader.batch_size
    avg_loss = total_loss / val_dataset_size
    pixel_acc = total_corrects / total_pixels
    
    # 计算每个类别的平均IoU
    for cls in range(NUM_CLASSES):
        if class_pixels[cls] > 0:
            class_iou[cls] /= class_pixels[cls]
    
    # 计算mIoU (平均交并比)
    miou = np.mean(class_iou)
    
    gc.collect()
    torch.cuda.empty_cache()
    
    return avg_loss, pixel_acc, miou, class_iou

def save_predictions(model, val_loader, device, output_dir='outputs', num_samples=5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model.eval()
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            if i >= num_samples:
                break
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            # 转换为NumPy数组用于可视化
            images_np = images.cpu().numpy()
            targets_np = targets.cpu().numpy()
            preds_np = preds.cpu().numpy()
            # 对每个样本进行可视化
            for b in range(images.size(0)):
                if b >= 3:  # 限制每个批次只保存前3个样本
                    break
                img = images_np[b].transpose(1, 2, 0)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = img * std + mean
                img = np.clip(img, 0, 1)
                target_rgb = decode_segmap(targets_np[b])
                pred_rgb = decode_segmap(preds_np[b])
                plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1)
                plt.title('Input Image')
                plt.imshow(img)
                plt.axis('off')
                plt.subplot(1, 3, 2)
                plt.title('Ground Truth')
                plt.imshow(target_rgb)
                plt.axis('off')
                plt.subplot(1, 3, 3)
                plt.title('Prediction')
                plt.imshow(pred_rgb)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'sample_{i}_{b}.png'))
                plt.close()

            del images, targets, outputs, preds
            torch.cuda.empty_cache()
    
    gc.collect()
    torch.cuda.empty_cache()

def main():
    args = parse_args()
    
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 加载数据
    train_loader, val_loader = get_data_loaders(
        args.voc_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    train_dataset_size = len(train_loader.dataset) if hasattr(train_loader.dataset, '__len__') else len(train_loader) * train_loader.batch_size
    val_dataset_size = len(val_loader.dataset) if hasattr(val_loader.dataset, '__len__') else len(val_loader) * val_loader.batch_size
    
    print(f'训练样本数: {train_dataset_size}, 验证样本数: {val_dataset_size}')
    
    # 创建模型
    model = get_fcn_model(model_type=args.model_type, num_classes=NUM_CLASSES, pretrained=True)
    print(model)
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=255)  
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # 恢复训练
    start_epoch = 0
    best_miou = 0.0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f'加载检查点: {args.resume}')
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_miou = checkpoint['best_miou']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f'从 epoch {start_epoch} 恢复训练, 最佳 mIoU: {best_miou:.4f}')
        else:
            print(f'找不到检查点: {args.resume}')
    
    # 训练循环
    print(f'开始训练 {args.model_type} 模型, 共 {args.epochs} 轮...')
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'pixel_acc': [],
        'miou': []
    }
    
    for epoch in range(start_epoch, args.epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        batch_count = 0
        
        t0 = time.time()
        for images, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}'):
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            print("输出形状:", outputs.shape)
            print("标签形状:", targets.shape)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            batch_count += 1
            
            del images, targets, outputs, loss
            
            if batch_count % 10 == 0:
                torch.cuda.empty_cache()
        
        train_loss = train_loss / train_dataset_size
        history['train_loss'].append(train_loss)
        
        # 调整学习率
        scheduler.step()
        
        # 执行垃圾回收
        gc.collect()
        torch.cuda.empty_cache()
        
        # 评估模型
        val_loss, pixel_acc, miou, class_iou = evaluate(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['pixel_acc'].append(pixel_acc)
        history['miou'].append(miou)
        
        # 打印进度
        epoch_time = time.time() - t0
        print(f'Epoch {epoch+1}/{args.epochs} - '
              f'Time: {epoch_time:.2f}s - '
              f'Train Loss: {train_loss:.4f} - '
              f'Val Loss: {val_loss:.4f} - '
              f'Pixel Acc: {pixel_acc:.4f} - '
              f'mIoU: {miou:.4f}')
        
        # 保存最佳模型
        if miou > best_miou:
            best_miou = miou
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_miou': best_miou,
            }, os.path.join(args.checkpoint_dir, f'{args.model_type}_best.pth'))
            print(f'保存最佳模型, mIoU: {best_miou:.4f}')
            
            # 生成可视化结果
            save_predictions(model, val_loader, device)
        
        # 保存最新模型
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_miou': best_miou,
        }, os.path.join(args.checkpoint_dir, f'{args.model_type}_latest.pth'))
        
        gc.collect()
        torch.cuda.empty_cache()
    

    plt.figure(figsize=(12, 10))
    

    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(history['pixel_acc'])
    plt.title('Pixel Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    

    plt.subplot(2, 2, 3)
    plt.plot(history['miou'])
    plt.title('Mean IoU')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.checkpoint_dir, f'{args.model_type}_history.png'))
    plt.close()
    
    print(f'训练完成! 最佳 mIoU: {best_miou:.4f}')

if __name__ == '__main__':
    main() 