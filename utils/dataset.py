# utils/dataset.py
import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms
import numpy as np
from collections import Counter
import os


def get_transforms(img_size=64, is_train=True):
    """定义数据增强"""
    if is_train:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(5),
            transforms.RandomAffine(0, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.3403, 0.3121, 0.3214],
                                 std=[0.2724, 0.2608, 0.2669])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.3403, 0.3121, 0.3214],
                                 std=[0.2724, 0.2608, 0.2669])
        ])
    return transform


def get_data_loaders(data_dir='data/GTSRB', img_size=64, batch_size=64, val_split=0.15):
    """获取DataLoader"""
    train_dir = os.path.join(data_dir, 'Final_Training', 'Images')

    # 加载完整训练集
    full_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=get_transforms(img_size, is_train=True)
    )

    print(f"找到 {len(full_dataset.classes)} 个类别")
    print(f"总训练样本数: {len(full_dataset)}")

    # 划分训练集和验证集
    val_size = int(val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size]
    )

    # 验证集使用不同的transform（无增强）
    val_dataset.dataset.transform = get_transforms(img_size, is_train=False)

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    # 获取类别名称
    class_names = full_dataset.classes

    return train_loader, val_loader, class_names


def get_class_weights(data_dir='data/GTSRB'):
    """计算类别权重以处理不均衡样本"""
    train_dir = os.path.join(data_dir, 'Final_Training', 'Images')
    dataset = datasets.ImageFolder(root=train_dir)
    class_counts = Counter(dataset.targets)

    total = sum(class_counts.values())
    num_classes = len(class_counts)

    # 计算权重: 总数 / (类别数 * 该类样本数)
    class_weights = [total / (num_classes * class_counts[i]) for i in range(num_classes)]

    print("\n类别分布统计:")
    for i in range(num_classes):
        print(f"  类别 {i:03d}: {class_counts[i]:5d} 张, 权重: {class_weights[i]:.3f}")

    return torch.tensor(class_weights, dtype=torch.float32)


def get_balanced_sampler(data_dir='data/GTSRB'):
    """获取均衡采样器（可选，用于处理不均衡）"""
    train_dir = os.path.join(data_dir, 'Final_Training', 'Images')
    dataset = datasets.ImageFolder(root=train_dir)

    class_counts = Counter(dataset.targets)
    class_weights = [1.0 / class_counts[i] for i in range(len(class_counts))]
    sample_weights = [class_weights[target] for target in dataset.targets]

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    return sampler