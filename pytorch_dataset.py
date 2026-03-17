"""多视角鹅数据集

用于加载和处理多视角鹅图像数据及其体尺测量标签。
"""
import os
import logging
from PIL import Image
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)


class MultiViewGooseDataset(Dataset):
    """多视角鹅数据集类
    
    参数:
        root_dir (str): 数据集根目录路径
        num_views (int): 每个样本的视角数量，默认4
        transform (callable, optional): 图像变换函数
        use_albumentations (bool): 是否使用Albumentations进行数据增强，默认True
    """
    
    VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    def __init__(self, root_dir, num_views=4, transform=None, use_albumentations=True, 
                 image_prefix='view', image_suffix='.jpg', label_file='label.txt'):
        self.root_dir = root_dir
        self.num_views = num_views
        self.use_albumentations = use_albumentations
        self.image_prefix = image_prefix
        self.image_suffix = image_suffix
        self.label_file = label_file
        
        if not os.path.isdir(root_dir):
            raise ValueError(f"数据集目录不存在: {root_dir}")
        
        self.samples = sorted([
            d for d in os.listdir(root_dir) 
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        
        if len(self.samples) == 0:
            raise ValueError(f"数据集目录为空: {root_dir}")
        
        self._validate_dataset()
        
        if transform is not None:
            self.transform = transform
        elif use_albumentations:
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.GaussNoise(p=0.1),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0
                ),
                ToTensorV2()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def _validate_dataset(self):
        invalid_samples = []
        for sample in self.samples:
            sample_dir = os.path.join(self.root_dir, sample)
            
            for i in range(1, self.num_views + 1):
                img_path = os.path.join(sample_dir, f'{self.image_prefix}{i}{self.image_suffix}')
                if not os.path.exists(img_path):
                    invalid_samples.append(f"{sample}/{self.image_prefix}{i}{self.image_suffix}")
            
            label_path = os.path.join(sample_dir, self.label_file)
            if not os.path.exists(label_path):
                invalid_samples.append(f"{sample}/{self.label_file}")
        
        if invalid_samples:
            logger.warning(f"发现缺失文件: {invalid_samples[:5]}{'...' if len(invalid_samples) > 5 else ''}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir = os.path.join(self.root_dir, self.samples[idx])
        images = []
        
        for i in range(1, self.num_views + 1):
            img_path = os.path.join(sample_dir, f'{self.image_prefix}{i}{self.image_suffix}')
            
            try:
                img = Image.open(img_path).convert('RGB')
            except FileNotFoundError:
                raise FileNotFoundError(f"图像文件不存在: {img_path}")
            except Exception as e:
                raise RuntimeError(f"无法加载图像 {img_path}: {e}")
            
            if self.use_albumentations:
                img_np = np.array(img)
                transformed = self.transform(image=img_np)
                img = transformed['image']
            else:
                img = self.transform(img)
                
            images.append(img)
        
        images = torch.stack(images, dim=0)

        label_path = os.path.join(sample_dir, self.label_file)
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    raise ValueError(f"标签文件为空: {label_path}")
                measurements = [float(x) for x in content.split()]
        except FileNotFoundError:
            raise FileNotFoundError(f"标签文件不存在: {label_path}")
        except ValueError as e:
            raise ValueError(f"标签格式错误 {label_path}: {e}")
        
        measurements = torch.tensor(measurements, dtype=torch.float32)

        return images, measurements


def create_dataloaders(root_dir, batch_size=8, num_views=4, train_ratio=0.8, num_workers=0):
    """创建训练和验证数据加载器
    
    参数:
        root_dir: 数据集根目录
        batch_size: 批量大小
        num_views: 视角数量
        train_ratio: 训练集比例
        num_workers: 数据加载线程数
    
    返回:
        train_loader, val_loader
    """
    from torch.utils.data import random_split
    
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(p=0.1),
        A.Rotate(limit=15, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2()
    ])
    
    full_dataset = MultiViewGooseDataset(root_dir, num_views=num_views, transform=train_transform)
    
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    val_dataset.dataset = MultiViewGooseDataset(root_dir, num_views=num_views, transform=val_transform)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader
