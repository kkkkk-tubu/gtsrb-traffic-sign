# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import numpy as np
from tqdm import tqdm
import os
import time

from utils.dataset import get_data_loaders, get_class_weights
from utils.visualize import plot_training_history


class SmallCNN(nn.Module):
    """自定义小型CNN"""

    def __init__(self, num_classes=43):
        super(SmallCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def get_mobilenet_model(num_classes=43):
    """获取预训练的MobileNetV2模型"""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # 冻结前10层
    for param in model.features[:8].parameters():
        param.requires_grad = False

    # 替换分类头
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, num_classes)
    )
    return model


def get_resnet_model(num_classes=43):
    """获取预训练的ResNet18模型"""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # 冻结前几层
    for param in model.parameters():
        param.requires_grad = False

    # 解冻最后两层
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    # 替换分类头
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """验证"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validating')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({'acc': 100. * correct / total})

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def train_model(model_name='cnn', num_epochs=15, batch_size=64, img_size=64,
                use_class_weight=True, save_dir='models'):
    """主训练函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 加载数据
    train_loader, val_loader, class_names = get_data_loaders(
        batch_size=batch_size, img_size=img_size
    )

    # 创建模型
    if model_name == 'cnn':
        model = SmallCNN(num_classes=len(class_names))
        print('Using SmallCNN model')
    elif model_name == 'mobilenet':
        model = get_mobilenet_model(num_classes=len(class_names))
        print('Using MobileNetV2 model')
    elif model_name == 'resnet':
        model = get_resnet_model(num_classes=len(class_names))
        print('Using ResNet18 model')
    else:
        raise ValueError(f'Unknown model: {model_name}')

    model = model.to(device)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')

    # 损失函数和优化器
    if use_class_weight:
        class_weights = get_class_weights().to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print('Using class weights for loss')
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # 训练历史
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f'\n{"=" * 50}')
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'{"=" * 50}')

        # 训练
        train_loss, train_acc = train_one_epoch(model, train_loader,
                                                criterion, optimizer, device)
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # 调整学习率
        scheduler.step()

        # 记录
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        epoch_time = time.time() - epoch_start
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Epoch time: {epoch_time:.1f}s')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names,
                'model_name': model_name
            }, f'{save_dir}/best_model_{model_name}.pt')
            print(f'✓ Saved best model with val_acc: {val_acc:.2f}%')

    total_time = time.time() - start_time
    print(f'\n{"=" * 50}')
    print(f'Training completed!')
    print(f'Total time: {total_time / 60:.1f} minutes')
    print(f'Best validation accuracy: {best_val_acc:.2f}%')
    print(f'{"=" * 50}')

    # 绘制训练曲线
    plot_training_history(history, save_path=f'{save_dir}/training_curve_{model_name}.png')

    return model, history, class_names


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'mobilenet', 'resnet'])
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--use_class_weight', action='store_true', help='使用类别加权')
    args = parser.parse_args()

    train_model(
        model_name=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        use_class_weight=args.use_class_weight 
    )