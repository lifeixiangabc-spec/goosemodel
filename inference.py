"""
推理脚本 - 使用训练好的模型预测鹅体尺测量值
"""
import os
import sys
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from models.model import MultiViewGooseTransformer


def load_image(path):
    """加载并预处理图像"""
    img = Image.open(path).convert('RGB')
    return np.array(img)


def predict_single_sample(model, sample_dir, device, num_views=4):
    """
    对单个样本进行预测
    
    参数:
        model: 训练好的模型
        sample_dir: 样本目录路径
        device: 计算设备
        num_views: 视角数量
    
    返回:
        预测的体尺测量值
    """
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2()
    ])
    
    images = []
    for i in range(1, num_views + 1):
        img_path = os.path.join(sample_dir, f'render_angle_{i}.png')
        if os.path.exists(img_path):
            img_np = load_image(img_path)
            transformed = transform(image=img_np)
            images.append(transformed['image'])
            print(f"  加载图像: {img_path}")
        else:
            raise FileNotFoundError(f"找不到视角 {i} 的图像文件: {img_path}")
    
    images = torch.stack(images, dim=0).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        predictions = model(images)
    
    return predictions.cpu().numpy()[0]


def main(sample_dir='data/goose_6 copy'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    model_path = 'outputs/manylow2_exp/final_model.pth'
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return
    
    print(f"加载模型: {model_path}")
    model = MultiViewGooseTransformer(num_measurements=3)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print("模型加载成功!")
    
    if not os.path.exists(sample_dir):
        print(f"错误: 样本目录不存在: {sample_dir}")
        return
    
    # 检查是否是目录，如果是目录则处理所有子目录
    if os.path.isdir(sample_dir):
        sample_dirs = [os.path.join(sample_dir, d) for d in os.listdir(sample_dir) if os.path.isdir(os.path.join(sample_dir, d))]
        print(f"\n找到 {len(sample_dirs)} 个样本目录")
        
        for dir_path in sample_dirs:
            print(f"\n预测样本: {dir_path}")
            try:
                predictions = predict_single_sample(model, dir_path, device)
                
                measurement_names = ['体长', '颈长', '胸宽']
                
                print("\n" + "="*50)
                print("预测结果:")
                print("="*50)
                for i, (name, pred) in enumerate(zip(measurement_names, predictions)):
                    print(f"  {name}: {pred:.2f} cm")
                print("="*50)
                
                result_file = os.path.join(dir_path, 'prediction_result.txt')
                with open(result_file, 'w', encoding='utf-8') as f:
                    f.write("鹅体尺测量预测结果\n")
                    f.write("="*50 + "\n")
                    f.write(f"样本目录: {dir_path}\n")
                    f.write("="*50 + "\n")
                    for name, pred in zip(measurement_names, predictions):
                        f.write(f"{name}: {pred:.2f} cm\n")
                    f.write("="*50 + "\n")
                    f.write(f"数值: {' '.join([f'{p:.2f}' for p in predictions])}\n")
                print(f"\n结果已保存到: {result_file}")
            except Exception as e:
                print(f"  预测失败: {e}")
    else:
        # 处理单个样本
        print(f"\n预测样本: {sample_dir}")
        predictions = predict_single_sample(model, sample_dir, device)
        
        measurement_names = ['体长', '颈长', '胸宽']
        
        print("\n" + "="*50)
        print("预测结果:")
        print("="*50)
        for i, (name, pred) in enumerate(zip(measurement_names, predictions)):
            print(f"  {name}: {pred:.2f} cm")
        print("="*50)
        
        result_file = os.path.join(sample_dir, 'prediction_result.txt')
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write("鹅体尺测量预测结果\n")
            f.write("="*50 + "\n")
            f.write(f"样本目录: {sample_dir}\n")
            f.write("="*50 + "\n")
            for name, pred in zip(measurement_names, predictions):
                f.write(f"{name}: {pred:.2f} cm\n")
            f.write("="*50 + "\n")
            f.write(f"数值: {' '.join([f'{p:.2f}' for p in predictions])}\n")
        print(f"\n结果已保存到: {result_file}")


if __name__ == '__main__':
    sample_dir = sys.argv[1] if len(sys.argv) > 1 else 'manytest'
    main(sample_dir)
