# generate_plots.py
import matplotlib

matplotlib.use('Agg')  # 解决某些环境下的显示问题

from utils.visualize import plot_class_distribution, show_batch
from utils.dataset import get_data_loaders
import os


def generate_all_plots():
    """生成所有必要的可视化图表"""
    os.makedirs('outputs', exist_ok=True)

    # 1. 类别分布图
    print("生成类别分布图...")
    fig1 = plot_class_distribution(data_dir='data/GTSRB',
                                   save_path='outputs/class_distribution.png')
    print("✓ 类别分布图已保存: outputs/class_distribution.png")

    # 2. 样本展示图
    print("🖼生成样本展示图...")
    train_loader, _, class_names = get_data_loaders(batch_size=16)
    fig2 = show_batch(train_loader, class_names, num_images=16,
                      save_path='outputs/sample_images.png')
    print("✓ 样本展示图已保存: outputs/sample_images.png")

    print("\n所有图表生成完成！")
    print("   文件位置:")
    print("   - outputs/class_distribution.png")
    print("   - outputs/sample_images.png")


if __name__ == '__main__':
    generate_all_plots()