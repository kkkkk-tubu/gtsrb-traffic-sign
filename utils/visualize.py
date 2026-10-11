# utils/visualize.py
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from torchvision.utils import make_grid
from collections import Counter
import os


def show_batch(dataloader, class_names, num_images=16, save_path=None):
    """显示一个batch的图片"""
    images, labels = next(iter(dataloader))
    images = images[:num_images]
    labels = labels[:num_images]

    # 反归一化
    mean = np.array([0.3403, 0.3121, 0.3214])
    std = np.array([0.2724, 0.2608, 0.2669])

    # 使用 make_grid 拼接成一张大图
    grid = make_grid(images, nrow=4, normalize=False)
    grid = grid.permute(1, 2, 0).numpy()
    grid = std * grid + mean
    grid = np.clip(grid, 0, 1)

    # 直接显示这张大图，不要再用子图
    plt.figure(figsize=(12, 12))
    plt.imshow(grid)

    # 添加标签（需要在正确位置添加）
    img_size = images.shape[2]  # 64
    for idx in range(num_images):
        row = idx // 4
        col = idx % 4
        # 计算文字位置（拼接图中心）
        x = col * img_size + img_size // 2
        y = row * img_size + 20  # 放在图片上方
        plt.text(x, y, class_names[labels[idx]],
                 color='white', fontsize=8, ha='center',
                 bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.3'))

    plt.axis('off')
    plt.title('Training Samples', fontsize=14)
    plt.tight_layout(pad=0)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def plot_class_distribution(data_dir='data/GTSRB', save_path=None):
    """绘制类别分布图"""
    from torchvision import datasets
    import os

    train_dir = os.path.join(data_dir, 'Final_Training', 'Images')
    dataset = datasets.ImageFolder(root=train_dir)
    class_counts = Counter(dataset.targets)
    class_names = dataset.classes

    plt.figure(figsize=(16, 6))
    bars = plt.bar(range(len(class_counts)), class_counts.values(), color='steelblue')
    plt.xticks(range(len(class_names)), class_names, rotation=90, fontsize=8)
    plt.xlabel('Class ID', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title('Class Distribution in GTSRB Dataset', fontsize=14)

    # 添加数值标签
    for bar, count in zip(bars, class_counts.values()):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 str(count), ha='center', fontsize=7)

    # 添加平均值线
    avg_count = sum(class_counts.values()) / len(class_counts)
    plt.axhline(y=avg_count, color='red', linestyle='--', label=f'Average: {avg_count:.0f}')
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return plt.gcf()


def plot_confusion_matrix(cm, class_names, save_path=None, figsize=(16, 14)):
    """绘制混淆矩阵"""
    plt.figure(figsize=figsize)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={'size': 8})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return plt.gcf()


def plot_training_history(history, save_path=None):
    """绘制训练历史曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss曲线
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy曲线
    axes[1].plot(history['train_acc'], label='Train Acc', marker='o', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Acc', marker='s', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def show_wrong_predictions(wrong_samples, class_names, n=16, save_path=None):
    """显示错误预测的样本"""
    n = min(n, len(wrong_samples))
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)

    mean = np.array([0.3403, 0.3121, 0.3214])
    std = np.array([0.2724, 0.2608, 0.2669])

    for idx, (image, true_label, pred_label) in enumerate(wrong_samples[:n]):
        row, col = idx // cols, idx % cols

        img = image.permute(1, 2, 0).numpy()
        img = std * img + mean
        img = np.clip(img, 0, 1)

        axes[row, col].imshow(img)
        axes[row, col].set_title(f'True: {class_names[true_label]}\nPred: {class_names[pred_label]}',
                                 color='red', fontsize=10)
        axes[row, col].axis('off')

    for idx in range(n, rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].axis('off')

    plt.suptitle('Wrong Predictions', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig