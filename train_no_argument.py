# train_no_augment.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import time

# 导入之前的模型和训练函数
from train import SmallCNN, train_one_epoch, validate
from utils.visualize import plot_training_history


def get_no_augment_loaders(data_dir='data/GTSRB', img_size=64, batch_size=64, val_split=0.15):
    """获取无数据增强的DataLoader（仅resize和归一化）"""

    # 只有基础变换，没有数据增强！
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3403, 0.3121, 0.3214],
                             std=[0.2724, 0.2608, 0.2669])
    ])

    # 加载数据集
    train_dir = os.path.join(data_dir, 'Final_Training', 'Images')
    full_dataset = datasets.ImageFolder(root=train_dir, transform=transform)

    # 划分训练集和验证集
    val_size = int(val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=2)

    print(f"总训练样本数: {len(full_dataset)}")
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"⚠️ 注意：当前使用【无数据增强】模式")

    return train_loader, val_loader, full_dataset.classes


def train_no_augment(num_epochs=15, batch_size=64, save_dir='models'):
    """无数据增强训练"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    os.makedirs(save_dir, exist_ok=True)

    # 获取无增强的数据加载器
    train_loader, val_loader, class_names = get_no_augment_loaders(batch_size=batch_size)

    # 创建模型
    model = SmallCNN(num_classes=len(class_names)).to(device)
    print(f'模型参数量: {sum(p.numel() for p in model.parameters()):,}')

    # 损失函数和优化器
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

    print(f"\n{'=' * 60}")
    print(f"开始训练 - 无数据增强模式")
    print(f"{'=' * 60}")

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 40)

        # 训练
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
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
            torch.save(model.state_dict(), f'{save_dir}/best_model_cnn_no_augment.pt')
            print(f'✓ 保存最佳模型，验证准确率: {val_acc:.2f}%')

    total_time = time.time() - start_time
    print(f'\n{"=" * 60}')
    print(f'训练完成！')
    print(f'总耗时: {total_time / 60:.1f} 分钟')
    print(f'最佳验证准确率: {best_val_acc:.2f}%')
    print(f'{"=" * 60}')

    # 绘制训练曲线
    plot_training_history(history, save_path=f'{save_dir}/training_curve_no_augment.png')

    return model, history, best_val_acc


if __name__ == '__main__':
    train_no_augment(num_epochs=15, batch_size=64)