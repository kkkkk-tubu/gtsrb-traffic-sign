# infer_onnx.py
import torch
import onnx
import onnxruntime as ort
import numpy as np
import time
from PIL import Image
import torchvision.transforms as transforms
import os
import glob

from tqdm import tqdm


def export_to_onnx(model_path, model_type='cnn', output_path='models/model.onnx',
                   img_size=64, batch_size=1):
    """将PyTorch模型导出为ONNX格式"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if model_type == 'cnn':
        from train import SmallCNN
        model = SmallCNN(num_classes=len(checkpoint['class_names']))
    elif model_type == 'mobilenet':
        from train import get_mobilenet_model
        model = get_mobilenet_model(num_classes=len(checkpoint['class_names']))
    else:
        from train import get_resnet_model
        model = get_resnet_model(num_classes=len(checkpoint['class_names']))

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 创建虚拟输入
    dummy_input = torch.randn(batch_size, 3, img_size, img_size).to(device)

    # 导出ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"✓ Model exported to {output_path}")

    # 验证ONNX模型
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model validation passed")

    # 获取模型大小
    model_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ ONNX model size: {model_size:.2f} MB")

    return output_path


def test_inference_time_pytorch(model_path, model_type='cnn', img_size=64, num_runs=100):
    """测试PyTorch推理时间"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if model_type == 'cnn':
        from train import SmallCNN
        model = SmallCNN(num_classes=len(checkpoint['class_names']))
    elif model_type == 'mobilenet':
        from train import get_mobilenet_model
        model = get_mobilenet_model(num_classes=len(checkpoint['class_names']))
    else:
        from train import get_resnet_model
        model = get_resnet_model(num_classes=len(checkpoint['class_names']))

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 准备测试输入
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)

    # 预热
    for _ in range(10):
        _ = model(dummy_input)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # 计时
    start_time = time.time()
    for _ in range(num_runs):
        _ = model(dummy_input)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    elapsed_time = time.time() - start_time
    avg_time = elapsed_time / num_runs * 1000  # 毫秒

    print(f"PyTorch ({model_type.upper()}) - Avg inference time: {avg_time:.2f} ms")

    # 计算吞吐量
    throughput = num_runs / elapsed_time
    print(f"  - Throughput: {throughput:.1f} images/sec")

    return avg_time


def test_inference_time_onnx(onnx_path, num_runs=100):
    """测试ONNX推理时间"""
    # 创建ONNX Runtime session
    ort_session = ort.InferenceSession(onnx_path)

    # 准备输入
    dummy_input = np.random.randn(1, 3, 64, 64).astype(np.float32)

    # 预热
    for _ in range(10):
        ort_session.run(None, {'input': dummy_input})

    # 计时
    start_time = time.time()
    for _ in range(num_runs):
        ort_session.run(None, {'input': dummy_input})

    elapsed_time = time.time() - start_time
    avg_time = elapsed_time / num_runs * 1000  # 毫秒

    print(f"ONNX Runtime - Avg inference time: {avg_time:.2f} ms")

    # 计算吞吐量
    throughput = num_runs / elapsed_time
    print(f"  - Throughput: {throughput:.1f} images/sec")

    return avg_time


def predict_single_image_onnx(onnx_path, image_path, img_size=64, class_names=None):
    """使用ONNX模型预测单张图片"""
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3403, 0.3121, 0.3214],
                             std=[0.2724, 0.2608, 0.2669])
    ])

    # 加载图片
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # 添加batch维度

    # ONNX推理
    ort_session = ort.InferenceSession(onnx_path)
    inputs = {ort_session.get_inputs()[0].name: image_tensor.numpy()}
    outputs = ort_session.run(None, inputs)

    # 获取预测结果
    predictions = outputs[0][0]
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    return predicted_class, confidence, predictions


def batch_predict_onnx(onnx_path, image_folder, img_size=64, num_images=None):
    """批量预测文件夹中的图片"""
    # 获取图片列表
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.ppm']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_folder, '**', ext), recursive=True))

    if num_images:
        image_paths = image_paths[:num_images]

    print(f"Found {len(image_paths)} images")

    results = []
    for img_path in tqdm(image_paths, desc="Predicting"):
        pred_class, confidence, _ = predict_single_image_onnx(onnx_path, img_path, img_size)
        results.append({
            'image': os.path.basename(img_path),
            'path': img_path,
            'predicted_class': pred_class,
            'confidence': confidence
        })

    return results


if __name__ == '__main__':
    print("=" * 50)
    print("ONNX Export and Inference Test")
    print("=" * 50)

    # 导出模型
    print("\n1. Exporting models to ONNX...")

    if os.path.exists('models/best_model_cnn.pt'):
        export_to_onnx('models/best_model_cnn.pt', 'cnn', 'models/model_cnn.onnx')

    if os.path.exists('models/best_model_mobilenet.pt'):
        export_to_onnx('models/best_model_mobilenet.pt', 'mobilenet', 'models/model_mobilenet.onnx')

    # 测试推理时间
    print("\n2. Testing inference time...")
    print("-" * 40)

    results = []

    if os.path.exists('models/best_model_cnn.pt'):
        time_pytorch = test_inference_time_pytorch('models/best_model_cnn.pt', 'cnn')
        time_onnx = test_inference_time_onnx('models/model_cnn.onnx')
        results.append(('SmallCNN', time_pytorch, time_onnx))

    if os.path.exists('models/best_model_mobilenet.pt'):
        time_pytorch = test_inference_time_pytorch('models/best_model_mobilenet.pt', 'mobilenet')
        time_onnx = test_inference_time_onnx('models/model_mobilenet.onnx')
        results.append(('MobileNet', time_pytorch, time_onnx))

    # 对比结果
    if results:
        print("\n3. Inference Time Comparison (ms per image):")
        print("-" * 60)
        print(f"{'Model':<15} {'PyTorch (ms)':<15} {'ONNX (ms)':<15} {'Speedup':<10}")
        print("-" * 60)
        for name, pytorch_time, onnx_time in results:
            speedup = pytorch_time / onnx_time if onnx_time > 0 else 1
            print(f"{name:<15} {pytorch_time:<15.2f} {onnx_time:<15.2f} {speedup:<10.2f}x")