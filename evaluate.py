# evaluate.py
import torch
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import argparse

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils.dataset import get_data_loaders
from utils.visualize import plot_confusion_matrix, show_wrong_predictions
from train import SmallCNN, get_mobilenet_model, get_resnet_model


class GTSRBTestDataset(Dataset):
    """GTSRB官方测试集"""

    def __init__(self, test_dir, gt_file, img_size=64):
        self.test_dir = test_dir
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.3403, 0.3121, 0.3214],
                                 std=[0.2724, 0.2608, 0.2669])
        ])
        self.gt = pd.read_csv(gt_file, sep=';')
        print(f"加载官方测试集: {len(self.gt)} 张图片")

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        row = self.gt.iloc[idx]
        img_path = os.path.join(self.test_dir, row['Filename'])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        label = row['ClassId']
        return image, label


def load_model(model_path, model_type='cnn'):
    """加载模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if model_type == 'cnn':
        model = SmallCNN(num_classes=43)
    elif model_type == 'mobilenet':
        model = get_mobilenet_model(num_classes=43)
    elif model_type == 'resnet':
        model = get_resnet_model(num_classes=43)
    else:
        raise ValueError(f'Unknown model type: {model_type}')

    # 处理不同的保存格式
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        class_names = checkpoint.get('class_names', [str(i) for i in range(43)])
    else:
        model.load_state_dict(checkpoint)
        class_names = [str(i).zfill(5) for i in range(43)]

    model = model.to(device)
    model.eval()

    return model, class_names, device


def evaluate_on_validation(model, device, class_names, batch_size=64, img_size=64, save_dir='outputs',
                           model_name='model'):
    """在验证集上评估"""
    _, val_loader, _ = get_data_loaders(batch_size=batch_size, img_size=img_size)

    all_preds = []
    all_labels = []
    wrong_samples = []

    print("在验证集上推理...")
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 收集错误样本
            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    wrong_samples.append((images[i].cpu(), labels[i].item(), preds[i].item()))

    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds) * 100
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"\n验证集准确率: {accuracy:.2f}%")
    print(f"Macro F1-Score: {macro_f1:.4f}")

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, save_path=f'{save_dir}/confusion_matrix_{model_name}_val.png')

    # 错误样本
    if wrong_samples:
        show_wrong_predictions(wrong_samples, class_names, n=16,
                               save_path=f'{save_dir}/wrong_cases_{model_name}_val.png')

    return accuracy, macro_f1


def evaluate_on_testset(model, device, class_names, save_dir='outputs', model_name='model'):
    """在官方测试集上评估"""
    test_dataset = GTSRBTestDataset(
        test_dir='data/GTSRB/Final_Test/Images',
        gt_file='data/GTSRB/GT-final_test.csv',
        img_size=64
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    all_preds = []
    all_labels = []
    wrong_samples = []  # 收集错误样本

    print("在官方测试集上推理...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 收集错误样本
            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    wrong_samples.append((images[i].cpu(), labels[i].item(), preds[i].item()))

    accuracy = accuracy_score(all_labels, all_preds) * 100
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"\n官方测试集准确率: {accuracy:.2f}%")
    print(f"Macro F1-Score: {macro_f1:.4f}")

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, save_path=f'{save_dir}/confusion_matrix_{model_name}_test.png',
                          figsize=(20, 18))

    # 错误样本可视化
    if wrong_samples:
        show_wrong_predictions(wrong_samples, class_names, n=16,
                               save_path=f'{save_dir}/wrong_cases_{model_name}_test.png')

    return accuracy, macro_f1


def evaluate_model(model_path, model_type='cnn', use_testset=False, save_dir='outputs'):
    """主评估函数"""
    os.makedirs(save_dir, exist_ok=True)

    # 加载模型
    model, class_names, device = load_model(model_path, model_type)

    print(f"\n{'=' * 50}")
    print(f"评估模型: {model_type.upper()}")
    print(f"{'=' * 50}")

    if use_testset:
        # 在官方测试集上评估
        acc, f1 = evaluate_on_testset(model, device, class_names, save_dir, model_type)
    else:
        # 在验证集上评估
        acc, f1 = evaluate_on_validation(model, device, class_names, save_dir=save_dir, model_name=model_type)

    return acc, f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'mobilenet', 'resnet'])
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--testset', action='store_true', help='使用官方测试集评估（默认使用验证集）')
    parser.add_argument('--save_dir', type=str, default='outputs')

    args = parser.parse_args()

    if args.model_path is None:
        args.model_path = f'models/best_model_{args.model}.pt'

    evaluate_model(
        model_path=args.model_path,
        model_type=args.model,
        use_testset=args.testset,
        save_dir=args.save_dir
    )