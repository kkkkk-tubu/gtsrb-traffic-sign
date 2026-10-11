# main.py
import os
import argparse


def print_header(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='GTSRB Traffic Sign Recognition')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['train', 'evaluate', 'export', 'all', 'compare'],
                        help='Run mode')
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['cnn', 'mobilenet', 'resnet', 'both'],
                        help='Model to use')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

    args = parser.parse_args()

    # 创建必要的目录
    for dir_name in ['models', 'outputs', 'data/GTSRB']:
        os.makedirs(dir_name, exist_ok=True)

    # 检查数据集是否存在
    train_path = 'data/GTSRB/Final_Training/Images'
    if not os.path.exists(train_path):
        print(f"\n⚠️ Warning: Dataset not found at {train_path}")
        print("Please ensure you have:")
        print("  1. Downloaded GTSRB_Final_Training_Images.zip")
        print("  2. Extracted to data/GTSRB/Final_Training/Images/")
        print("\nDataset structure should be:")
        print("  data/GTSRB/Final_Training/Images/")
        print("    ├── 00000/")
        print("    ├── 00001/")
        print("    └── ... (up to 00042/)")

        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return

    # 训练模式
    if args.mode == 'train' or args.mode == 'all':
        print_header("TRAINING MODELS")

        models_to_train = []
        if args.model == 'both':
            models_to_train = ['cnn', 'mobilenet']
        else:
            models_to_train = [args.model]

        for model_name in models_to_train:
            print(f"\n>>> Training {model_name.upper()}...")
            from train import train_model
            train_model(
                model_name=model_name,
                num_epochs=args.epochs,
                batch_size=args.batch_size
            )

    # 评估模式
    if args.mode == 'evaluate' or args.mode == 'all':
        print_header("EVALUATING MODELS")

        from evaluate import evaluate_model

        models_to_eval = []
        if args.model == 'both':
            models_to_eval = [('cnn', 'models/best_model_cnn.pt'),
                              ('mobilenet', 'models/best_model_mobilenet.pt')]
        else:
            models_to_eval = [(args.model, f'models/best_model_{args.model}.pt')]

        for model_name, model_path in models_to_eval:
            if os.path.exists(model_path):
                evaluate_model(model_path, model_name)
            else:
                print(f"⚠️ Model not found: {model_path}")

    # 模型对比模式
    if args.mode == 'compare':
        print_header("MODEL COMPARISON")

        from evaluate import compare_models

        model_paths = {}
        for model_name in ['cnn', 'mobilenet', 'resnet']:
            path = f'models/best_model_{model_name}.pt'
            if os.path.exists(path):
                model_paths[model_name] = path

        if model_paths:
            compare_models(model_paths)
        else:
            print("No models found for comparison. Please train models first.")

    # ONNX导出和测试
    if args.mode == 'export' or args.mode == 'all':
        print_header("ONNX EXPORT AND INFERENCE TEST")

        from infer_onnx import export_to_onnx, test_inference_time_pytorch, test_inference_time_onnx

        models_to_export = []
        if args.model == 'both':
            models_to_export = [('cnn', 'models/best_model_cnn.pt', 'models/model_cnn.onnx'),
                                ('mobilenet', 'models/best_model_mobilenet.pt', 'models/model_mobilenet.onnx')]
        else:
            models_to_export = [(args.model, f'models/best_model_{args.model}.pt',
                                 f'models/model_{args.model}.onnx')]

        for model_name, model_path, onnx_path in models_to_export:
            if os.path.exists(model_path):
                print(f"\n>>> Processing {model_name.upper()}...")
                export_to_onnx(model_path, model_name, onnx_path)
                test_inference_time_pytorch(model_path, model_name)
                test_inference_time_onnx(onnx_path)
            else:
                print(f"⚠️ Model not found: {model_path}")

    print_header("PROJECT COMPLETED")
    print("\nOutput files:")
    print("  - Models: models/")
    print("  - Evaluation results: outputs/")
    print("  - ONNX models: models/model_*.onnx")
    print("  - Training curves: models/training_curve_*.png")
    print("  - Confusion matrices: outputs/confusion_matrix_*.png")


if __name__ == '__main__':
    main()