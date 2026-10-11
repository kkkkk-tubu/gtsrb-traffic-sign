# 交通标志识别与轻量化部署

基于 GTSRB 数据集的交通标志识别实验，包含数据预处理与增强策略的消融对比研究，以及模型轻量化部署实践。

> 《人工智能开发平台实训》课程项目 · 团队项目（4 人）

## 项目简介

交通标志识别（TSR）是自动驾驶感知层的核心技术，负责从视觉数据中提取限速、禁令、指示等道路标志信息。本项目在 GTSRB 基准数据集上完成了从数据预处理、模型训练到轻量化部署的完整流程。

## 数据集

**GTSRB**（German Traffic Sign Recognition Benchmark，2011 年发布）

| 项目 | 说明 |
|---|---|
| 类别数 | 43 类德国交通标志 |
| 训练集 | 39,209 张 |
| 测试集 | 12,630 张 |
| 图像特点 | 覆盖不同光照、角度、天气条件下的真实道路场景 |

**关键挑战：类别严重不均衡**

样本分布呈现明显长尾效应，类别间数量差距达 **10.7 倍**（最大类 2,250 张 vs 最小类 210 张，平均 912 张），易导致模型对少数类特征学习不足。

## 数据预处理

### 1. 统一尺寸

所有训练样本统一缩放至 **64×64** 像素，在保留核心特征的前提下平衡计算精度与推理速度。

### 2. 多策略数据增强（消融实验）

设计三组对照实验：

| 策略 | 内容 | 结果 |
|---|---|---|
| 无增强 | 原图直接训练 | **96.87%（最优）** |
| 温和增强 | 随机翻转、轻微裁剪（Rotation 5°/ColorJitter 0.15） | 95.74% |
| 较强增强 | 亮度扰动、旋转（Rotation 15°/ColorJitter 0.3/RandomAffine 0.1） | 95.80% |

> 三组都用 SmallCNN，在 GTSRB 官方测试集（12,630 张）上评测。

### 3. 像素级归一化

采用定制化均值与标准差对像素值标准化，加速模型收敛并稳定梯度：

```python
mean = [0.3403, 0.3121, 0.3214]
std  = [0.2724, 0.2608, 0.2669]
```

### 4. 类别加权机制

长尾分布会让模型偏向多数类。工程中对损失函数施加**逆频率类别加权**：

```python
# utils/dataset.py
total = sum(class_counts.values())
num_classes = len(class_counts)
class_weights = [total / (num_classes * class_counts[i]) for i in range(num_classes)]

criterion = nn.CrossEntropyLoss(weight=class_weights)
```

即 `w_c = N_total / (C × N_c)`——等价于「平均样本数 ÷ 该类样本数」，样本越少的类权重越高。
本数据集平均每类 912 张，故 `w_c ≈ 912 / N_c`：最小类（210 张）的权重约为最大类（2,250 张）的 **10.7 倍**，正好抵消长尾差距。

## 核心实验发现

> **反直觉结论：不进行数据增强的模型准确率最高（96.87%），引入增强后准确率反而下降约 1%。**

原因分析：

1. **数据特性天然丰富**——GTSRB 本身已覆盖多样的光照、角度与模糊变化，样本自然变异性足以支撑模型学习
2. **过度增强破坏特征**——交通标志形状与符号高度规范化，旋转、缩放、裁剪会破坏关键语义信息
3. **引发分布偏移**——不当增强引入偏离真实环境的特征分布，导致训练集与测试集不一致

这一结论为团队的模型选型提供了关键实验依据。

## 实验结果

| 模型 | 准确率 | 参数量 | ONNX 推理耗时 |
|---|---|---|---|
| SmallCNN（基线） | 96.87% | ~6.5M | 0.26 ms |
| MobileNetV2（迁移学习） | 94.09% | — | 0.53 ms |

**部署优化**：ONNX 格式转换后推理提速 4.19–11.53 倍，模型体积压缩至 0.25MB（缩减 26 倍），适配嵌入式边缘计算平台。

## 误差分析

- **限速标志混淆（约 40%）**——相似数字（30/80、40/90）轮廓特征相近，局部纹理区分度不足
- **光照/对比度异常（约 25%）**——夜间、逆光场景下特征提取失效
- **模糊与遮挡（约 20%）**——拍摄距离过远或运动模糊导致关键特征缺失
- **少数类欠拟合（约 15%）**——低频标志样本量不足

## 我负责的部分

**数据预处理与增强**（对应 `src/utils/dataset.py`）

- `get_transforms()`——统一尺寸至 64×64，定义训练/验证两套变换（验证集不施加增强）
- `get_data_loaders()`——ImageFolder 加载、按 85%/15% 划分训练/验证、DataLoader 组装
- `get_class_weights()` / `get_balanced_sampler()`——长尾分布的两条处理路径（加权损失 / 加权采样）
- **设计并完成三组数据增强消融对照实验**，得出「无增强最优」的结论

## 项目结构

```
.
├── README.md
├── requirements.txt
├── main.py                    # 统一入口（train / evaluate / export / compare）
├── train.py                   # 模型定义（SmallCNN / MobileNetV2 / ResNet18）与训练循环
├── train_no_argument.py       # 无增强消融实验
├── evaluate.py                # 模型评估（验证集 / 官方测试集）
├── infer_onnx.py              # ONNX 导出与推理测速
├── generate_plots.py          # 类别分布、样例图等可视化
├── utils/
│   ├── dataset.py             # 数据预处理与增强
│   └── visualize.py           # 混淆矩阵、训练曲线、错误样本等绘图
├── models/                    # 训练权重与导出的 ONNX 模型
├── outputs/                   # 实验图表
└── data/GTSRB/                # 数据集（需按官方说明自行下载）
```

## 如何运行

```bash
pip install -r requirements.txt

# 训练（默认 15 epoch）
python main.py --mode train --model cnn --epochs 15
python main.py --mode train --model mobilenet --epochs 15

# 无增强消融实验
python train_no_argument.py

# 在官方测试集（12,630 张）上评估
python evaluate.py --model cnn --testset

# 导出 ONNX 并测速
python main.py --mode export --model cnn
```

> 数据集需自行下载：GTSRB 官方页面 → 解压为 `data/GTSRB/Final_Training/Images/00000…00042`。

## 技术栈

Python · PyTorch · torchvision · NumPy · Matplotlib · scikit-learn · ONNX Runtime

## 参考

- [GTSRB 官方数据集](https://benchmark.ini.rub.de/gtsrb_dataset.html)
- MobileNetV2: Sandler et al., *MobileNetV2: Inverted Residuals and Linear Bottlenecks*, CVPR 2018
