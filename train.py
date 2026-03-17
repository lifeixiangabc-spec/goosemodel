"""鹅体尺测量模型训练脚本

该脚本用于训练多视角鹅体尺测量模型，使用Transformer融合多个视角的特征。
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from pytorch_dataset import MultiViewGooseDataset
from models.model import MultiViewGooseTransformer


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_samples = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]", leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * imgs.size(0)
        num_samples += imgs.size(0)
        
        avg_loss = total_loss / num_samples
        pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'batch_loss': f'{loss.item():.4f}'})
    
    return total_loss / num_samples


def validate(model, dataloader, criterion, device, epoch, total_epochs):
    """验证模型"""
    model.eval()
    total_loss = 0
    num_samples = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs} [Val]", leave=False)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * imgs.size(0)
            num_samples += imgs.size(0)
            
            avg_loss = total_loss / num_samples
            pbar.set_postfix({'val_loss': f'{avg_loss:.4f}'})
    
    return total_loss / num_samples


def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    """保存检查点"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }, path)


def load_checkpoint(model, optimizer, scheduler, path):
    """加载检查点"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']


def main(args):
    """主训练函数"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"训练设备: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("加载数据集...")
    full_dataset = MultiViewGooseDataset(
        root_dir=args.data_dir, 
        num_views=args.num_views,
        use_albumentations=True,
        image_prefix=args.image_prefix,
        image_suffix=args.image_suffix,
        label_file=args.label_file
    )
    
    train_size = int(args.train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"训练集: {len(train_dataset)} 样本, 验证集: {len(val_dataset)} 样本")
    
    print("初始化模型...")
    model = MultiViewGooseTransformer(
        num_measurements=args.num_measurements,
        backbone_name=args.backbone
    ).to(device)
    
    model.print_model_info()
    
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs, 
        eta_min=args.lr * 0.01
    )
    
    early_stopping = EarlyStopping(patience=args.patience, min_delta=1e-4)
    
    writer = SummaryWriter(os.path.join(args.output_dir, 'runs', args.experiment_name))
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        if os.path.exists(args.resume):
            print(f"从检查点恢复: {args.resume}")
            start_epoch, _ = load_checkpoint(model, optimizer, scheduler, args.resume)
        else:
            print(f"检查点文件不存在: {args.resume}")
    
    print("开始训练...")
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs)
        val_loss = validate(model, val_loader, criterion, device, epoch, args.epochs)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                os.path.join(args.output_dir, 'best_model.pth')
            )
            print(f"  保存最佳模型 (Val Loss: {val_loss:.4f})")
        
        if epoch % args.save_interval == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
            )
        
        if early_stopping(val_loss):
            print(f"早停触发！最佳验证损失: {early_stopping.best_score:.4f}")
            break
    
    save_checkpoint(
        model, optimizer, scheduler, args.epochs - 1, val_loss,
        os.path.join(args.output_dir, 'final_model.pth')
    )
    print(f"训练完成！最终模型已保存到 {args.output_dir}/final_model.pth")
    
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='鹅体尺测量模型训练')
    
    parser.add_argument('--data_dir', type=str, default='data/', help='数据集目录')
    parser.add_argument('--output_dir', type=str, default='outputs', help='输出目录')
    parser.add_argument('--experiment_name', type=str, default='goose_measurements', help='实验名称')
    
    parser.add_argument('--backbone', type=str, default='resnet50', 
                        choices=['resnet50', 'efficientnet_b4'], help='主干网络')
    parser.add_argument('--num_views', type=int, default=4, help='视角数量')
    parser.add_argument('--num_measurements', type=int, default=5, help='体尺测量值数量')
    parser.add_argument('--image_prefix', type=str, default='view', help='图像文件前缀')
    parser.add_argument('--image_suffix', type=str, default='.jpg', help='图像文件后缀')
    parser.add_argument('--label_file', type=str, default='label.txt', help='标签文件名')
    
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批量大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    
    parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载线程数')
    
    parser.add_argument('--save_interval', type=int, default=5, help='保存检查点间隔')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    main(args)
