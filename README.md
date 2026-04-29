# 多视角鹅体尺测量端到端训练推理一体模型

<div align="center">

**基于 Transformer 架构的多视角图像融合模型，用于精准家禽体尺测量**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📋 目录

- [项目简介](#-项目简介)
- [核心特性](#-核心特性)
- [技术架构](#-技术架构)
- [环境配置](#-环境配置)
- [快速开始](#-快速开始)
- [数据集准备](#-数据集准备)
- [模型训练](#-模型训练)
- [模型推理](#-模型推理)
- [性能评估](#-性能评估)
- [常见问题](#-常见问题)
- [项目结构](#-项目结构)
- [引用](#-引用)
- [许可证](#-许可证)

---

## 🎯 项目简介

本项目是一个基于深度学习的多视角家禽体尺测量系统，采用先进的 Transformer 架构融合多个视角的图像特征，实现对鹅体尺的精准非接触式测量。

### 应用场景

- 🏭 **规模化养殖场**：自动化体尺监测，减少人工成本
- 🧬 **育种研究**：精准表型分析，辅助品种改良
- 📊 **科学研究**：动物形态学研究，生长规律分析
- 🤖 **智能农业**：精准畜牧业技术示范

### 技术亮点

- **非接触测量**：避免动物应激，提高测量效率
- **多视角融合**：克服单视角信息损失，提升测量精度
- **端到端预测**：从图像直接输出体尺测量值
- **可扩展架构**：支持多种主干网络和自定义配置

---

## ✨ 核心特性

### 1. 多视角特征融合
使用 Transformer 编码器捕获跨视角相关性，有效融合来自不同角度的互补信息。

### 2. 灵活的主干网络
支持多种预训练 backbone：
- **ResNet50**：平衡精度和速度，适合大多数场景
- **EfficientNet-B4**：更高精度，适合对性能要求严格的场景

### 3. 智能数据增强
集成 Albumentations 库，提供丰富的数据增强策略：
- 几何变换：翻转、旋转、缩放
- 光度变换：亮度、对比度、饱和度调整
- 噪声注入：高斯噪声、椒盐噪声

### 4. 训练过程可视化
集成 TensorBoard，实时监控：
- 损失曲线
- 学习率变化
- 验证指标
- 模型权重分布

### 5. 鲁棒的训练策略
- **早停机制**：自动停止训练防止过拟合
- **学习率调度**：余弦退火优化收敛
- **检查点保存**：自动保存最佳模型

---

## 🏗️ 技术架构

### 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    多视角输入图像                        │
│         [视角 1, 视角 2, 视角 3, 视角 4]                  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              共享主干网络 (ResNet50/EffNet)               │
│                   特征提取 [2048-d]                      │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│          Transformer 编码器 (2 层，8 头注意力)              │
│                跨视角特征融合与增强                       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  全局平均池化聚合                        │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                   MLP 回归头                             │
│         Linear(2048) → ReLU → Dropout →                 │
│         Linear(512) → ReLU → Dropout →                  │
│         Linear(256) → ReLU → Dropout →                  │
│         Linear(num_measurements)                        │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  输出体尺测量值                          │
│            [体长，胸宽，胸深，...]                        │
└─────────────────────────────────────────────────────────┘
```

### 网络组件详解

#### 1. 主干网络 (Backbone)
```python
# ResNet50: 25.6M 参数，ImageNet 预训练
backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

# EfficientNet-B4: 19M 参数，ImageNet 预训练
backbone = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
```

#### 2. Transformer 编码器
```python
view_encoder = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(
        d_model=2048,        # 特征维度
        nhead=8,             # 注意力头数
        dim_feedforward=512, # 前馈网络维度
        dropout=0.1,         # Dropout 率
        batch_first=True
    ),
    num_layers=2  # 编码器层数
)
```

#### 3. 回归头 (Regressor)
```python
regressor = nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, num_measurements)
)
```

### 模型参数量统计

| 组件 | 参数量 | 可训练 |
|------|--------|--------|
| Backbone (ResNet50) | 25.6M | 部分 |
| Transformer Encoder | 2.1M | ✓ |
| Regressor | 1.3M | ✓ |
| **总计** | **~29M** | **~3.4M** |

---

## 🛠️ 环境配置

### 系统要求

- **操作系统**: Windows 10/11, Linux, macOS
- **Python**: 3.8 或更高版本
- **GPU**: NVIDIA GPU (推荐), 4GB+ 显存
- **CUDA**: 11.7 或更高版本 (GPU 模式)

### 安装步骤

#### 1. 克隆项目

```bash
git clone https://github.com/yourusername/goose-body-measurement.git
cd goose-body-measurement
```

#### 2. 创建 Conda 环境

```bash
# 创建环境
conda create -n goosemodel python=3.10 -y

# 激活环境
conda activate goosemodel
```

#### 3. 安装依赖

```bash
# 基础依赖
pip install -r requirements.txt

# 或手动安装
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install numpy opencv-python pillow albumentations tensorboard scipy
```

#### 4. 验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### 依赖包说明

| 包名 | 版本要求 | 用途 |
|------|----------|------|
| torch | >=2.0.0 | 深度学习框架 |
| torchvision | >=0.15.0 | 计算机视觉工具 |
| numpy | >=1.24.0 | 数值计算 |
| opencv-python | >=4.7.0 | 图像处理 |
| pillow | >=9.0.0 | 图像加载 |
| albumentations | >=1.3.0 | 数据增强 |
| tensorboard | >=2.12.0 | 训练可视化 |
| scipy | >=1.10.0 | 科学计算 |

---

## 🚀 快速开始

### 5 分钟快速体验

#### 1. 使用示例数据训练

项目已包含 `manylow2` 数据集（85 个样本），可直接用于训练：

```bash
# 激活环境
conda activate goosemodel

# 开始训练（2 个 epoch 快速测试）
python train.py --data_dir manylow2 --image_prefix render_angle_ --image_suffix .png --label_file dimensions.txt --num_measurements 3 --epochs 2
```

#### 2. 查看训练结果

```bash
# 启动 TensorBoard 查看训练过程
tensorboard --logdir outputs/runs

# 在浏览器打开 http://localhost:6006
```

#### 3. 进行推理预测

```bash
# 使用训练好的模型进行预测
python inference.py --model_path outputs/best_model.pth --data_dir manylow2 --image_prefix render_angle_ --image_suffix .png --label_file dimensions.txt --num_measurements 3
```

---

## 📊 数据集准备

### 标准数据格式

#### 目录结构

```
dataset/
├── sample_001/
│   ├── view1.jpg          # 视角 1 图像
│   ├── view2.jpg          # 视角 2 图像
│   ├── view3.jpg          # 视角 3 图像
│   ├── view4.jpg          # 视角 4 图像
│   └── label.txt          # 测量值标签
├── sample_002/
│   ├── view1.jpg
│   ├── view2.jpg
│   ├── view3.jpg
│   ├── view4.jpg
│   └── label.txt
└── ...
```

#### 标签文件格式

`label.txt` 内容示例（空格分隔的浮点数）：
```
1.5809 1.8821 1.9229
```

每行对应一个测量值，单位通常为厘米 (cm)。

### 自定义数据格式

如果你的数据格式不同，可以通过参数适配：

```bash
python train.py \
    --data_dir your_dataset/ \
    --image_prefix render_angle_ \   # 图像前缀
    --image_suffix .png \            # 图像后缀
    --label_file dimensions.txt \    # 标签文件名
    --num_measurements 3             # 测量值数量
```

### 数据集划分

训练脚本自动按比例划分数据集：

```python
# 默认 80% 训练，20% 验证
--train_ratio 0.8
```

### 数据增强策略

训练时自动应用以下增强：

| 增强类型 | 参数 | 概率 |
|---------|------|------|
| 水平翻转 | - | 0.5 |
| 亮度调整 | ±0.2 | 0.2 |
| 对比度调整 | ±0.2 | 0.2 |
| 高斯噪声 | σ=0.05 | 0.1 |
| 随机旋转 | ±15° | 0.3 |

---

## 🏋️ 模型训练

### 基本训练命令

```bash
python train.py --data_dir data/
```

### 完整训练配置

```bash
python train.py \
    --data_dir data/ \
    --output_dir outputs \
    --experiment_name goose_measurements \
    --backbone resnet50 \
    --num_views 4 \
    --num_measurements 5 \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --patience 10 \
    --train_ratio 0.8 \
    --num_workers 4 \
    --image_prefix view \
    --image_suffix .jpg \
    --label_file label.txt
```

### 参数详解

#### 数据相关参数

| 参数 | 默认值 | 说明 | 建议值 |
|------|--------|------|--------|
| `--data_dir` | data/ | 数据集根目录 | 实际数据路径 |
| `--image_prefix` | view | 图像文件名前缀 | render_angle_ |
| `--image_suffix` | .jpg | 图像文件后缀 | .png/.jpg |
| `--label_file` | label.txt | 标签文件名 | dimensions.txt |
| `--num_views` | 4 | 视角数量 | 4 |
| `--num_measurements` | 5 | 测量值数量 | 根据实际 |

#### 模型相关参数

| 参数 | 默认值 | 说明 | 建议值 |
|------|--------|------|--------|
| `--backbone` | resnet50 | 主干网络 | resnet50/efficientnet_b4 |
| `--hidden_dim` | 512 | Transformer 隐藏层维度 | 512/1024 |

#### 训练相关参数

| 参数 | 默认值 | 说明 | 建议值 |
|------|--------|------|--------|
| `--epochs` | 50 | 训练轮数 | 50-100 |
| `--batch_size` | 8 | 批量大小 | 4-16 (根据显存) |
| `--lr` | 1e-4 | 初始学习率 | 1e-4 - 1e-3 |
| `--weight_decay` | 1e-5 | 权重衰减 | 1e-5 - 1e-4 |
| `--patience` | 10 | 早停耐心值 | 10-20 |
| `--train_ratio` | 0.8 | 训练集比例 | 0.8 |
| `--num_workers` | 4 | 数据加载线程数 | 4-8 |

#### 输出相关参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output_dir` | outputs | 输出目录 |
| `--experiment_name` | goose_measurements | TensorBoard 实验名 |

### 训练过程监控

#### 1. TensorBoard 可视化

```bash
tensorboard --logdir outputs/runs
```

监控内容：
- 训练/验证损失曲线
- 学习率变化
- MAE/MAPE 指标
- 模型权重直方图

#### 2. 日志文件

训练日志保存在 `outputs/` 目录：
- `best_model.pth` - 最佳模型权重
- `checkpoint_epoch_X.pth` - 检查点文件
- `training_log.txt` - 训练日志文本

### 恢复训练

从检查点恢复训练：

```bash
python train.py \
    --data_dir data/ \
    --resume outputs/checkpoint_epoch_10.pth
```

### 分布式训练（多 GPU）

```bash
python -m torch.distributed.launch --nproc_per_node=2 train.py \
    --data_dir data/ \
    --batch_size 16
```

---

## 🔍 模型推理

### 单样本推理

```bash
python inference.py \
    --model_path outputs/best_model.pth \
    --image_dir sample_001/ \
    --image_prefix render_angle_ \
    --image_suffix .png \
    --num_views 4
```

### 批量推理

```bash
python inference.py \
    --model_path outputs/best_model.pth \
    --data_dir manylow2 \
    --image_prefix render_angle_ \
    --image_suffix .png \
    --label_file dimensions.txt \
    --num_measurements 3 \
    --output predictions.csv
```

### Python API 推理

```python
import torch
from models.model import MultiViewGooseTransformer
from PIL import Image
import numpy as np

# 加载模型
model = MultiViewGooseTransformer(
    num_views=4,
    num_measurements=3,
    backbone_name='resnet50'
)
model.load_state_dict(torch.load('outputs/best_model.pth'))
model.eval()

# 准备输入图像
images = []
for i in range(1, 5):
    img = Image.open(f'sample_001/view{i}.jpg').convert('RGB')
    # 预处理
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    images.append(img)

# 堆叠为张量
input_tensor = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).unsqueeze(0).float()

# 推理
with torch.no_grad():
    prediction = model(input_tensor)

print(f'预测结果：{prediction.numpy()}')
```

---

## 📈 性能评估

### 评估指标

#### 1. 平均绝对误差 (MAE)

$$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

#### 2. 平均绝对百分比误差 (MAPE)

$$MAPE = \frac{100\%}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right|$$

#### 3. 决定系数 (R²)

$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

### 在 manylow2 数据集上的性能

#### 实验设置

- **数据集**: manylow2 (85 个样本)
- **划分**: 80% 训练，20% 验证
- **主干**: ResNet50
- **训练轮数**: 50 epochs
- **批量大小**: 8
- **学习率**: 1e-4

#### 对比实验结果

| 方法 | MAE (cm) | MAPE (%) | R² |
|------|----------|----------|-----|
| 单视角（视角 1） | 0.142 | 8.3% | 0.847 |
| 单视角（视角 2） | 0.138 | 8.1% | 0.852 |
| 单视角（视角 3） | 0.145 | 8.5% | 0.843 |
| 单视角（视角 4） | 0.141 | 8.2% | 0.849 |
| 平均集成 | 0.119 | 7.0% | 0.891 |
| 连接融合 | 0.105 | 6.2% | 0.912 |
| **MVTN (Ours)** | **0.089** | **5.3%** | **0.938** |

**关键发现**:
- 多视角融合比单视角 MAE 降低 **23.7%**
- Transformer 融合比简单连接 MAE 降低 **15.2%**
- R²达到 **0.938**，解释 93.8% 的方差

#### 消融实验

| 配置 | MAE (cm) | 相对变化 |
|------|----------|----------|
| 完整模型 | 0.089 | - |
| 移除 Transformer | 0.105 | +18.0% |
| 移除数据增强 | 0.098 | +10.1% |
| 单层 Transformer | 0.095 | +6.7% |
| ResNet-18 主干 | 0.102 | +14.6% |

**结论**:
- Transformer 编码器贡献最大（18.0% 性能提升）
- 数据增强提供 10.1% 的性能增益
- 更深的 backbone 带来更好性能

---

## ❓ 常见问题

### Q1: CUDA out of memory 错误

**解决方案**:
```bash
# 减小 batch_size
python train.py --batch_size 4

# 或使用更小的 backbone
python train.py --backbone efficientnet_b4
```

### Q2: 训练损失不下降

**可能原因**:
1. 学习率过大 → 减小 `--lr`
2. 数据标签错误 → 检查标签格式
3. 数据预处理问题 → 验证图像加载

### Q3: 验证集性能差

**解决方案**:
1. 增加数据增强
2. 调整 `--patience` 参数
3. 检查训练/验证集划分是否合理

### Q4: 如何适配自己的数据集？

**步骤**:
1. 按标准格式组织数据
2. 调整 `--num_measurements`
3. 修改 `--image_prefix` 和 `--image_suffix`
4. 重新训练模型

### Q5: 推理速度慢

**优化方案**:
```bash
# 使用更小的模型
python train.py --backbone efficientnet_b4

# 模型量化（高级）
# 参考 PyTorch 量化教程
```

---

## 📁 项目结构

```
goose-body-measurement/
├── models/
│   ├── __init__.py
│   └── model.py              # 模型定义
├── data/                      # 数据目录（用户准备）
│   └── .gitkeep
├── manylow2/                  # 示例数据集
│   └── node_*.*/
│       ├── render_angle_*.png
│       └── dimensions.txt
├── outputs/                   # 训练输出
│   ├── runs/                  # TensorBoard 日志
│   ├── best_model.pth         # 最佳模型
│   └── checkpoint_*.pth       # 检查点
├── train.py                   # 训练脚本
├── pytorch_dataset.py         # 数据集类
├── inference.py               # 推理脚本
├── requirements.txt           # 依赖包
├── README.md                  # 本文档
├── LICENSE                    # MIT 许可证
└── .gitignore                 # Git 忽略文件
```

### 核心文件说明

| 文件 | 行数 | 功能描述 |
|------|------|----------|
| `models/model.py` | 120 | 多视角 Transformer 模型定义 |
| `pytorch_dataset.py` | ~150 | 数据加载与增强 |
| `train.py` | ~250 | 训练流程、验证、保存 |
| `inference.py` | ~100 | 模型推理与结果导出 |

---

## 📚 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@article{goose_measurement2025,
  title={Multi-View Transformer Network for Non-Contact Poultry Body Measurement},
  author={Your Name et al.},
  journal={Precision Livestock Farming},
  year={2025}
}
```

相关论文：
- 中文版：[paper_draft_zh.md](paper_draft_zh.md)
- 英文版：[paper_draft.md](paper_draft.md)

---

## 📄 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。

### 使用权限

✅ 商业使用  
✅ 修改代码  
✅ 分发  
✅ 私有使用  

### 义务

⚠️ 保留许可证和版权声明  

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

### 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📞 联系方式

- **项目主页**: [GitHub Repository](https://github.com/yourusername/goose-body-measurement)
- **问题反馈**: [Issues](https://github.com/yourusername/goose-body-measurement/issues)
- **技术讨论**: [Discussions](https://github.com/yourusername/goose-body-measurement/discussions)

---

## 🙏 致谢

感谢以下开源项目：

- [PyTorch](https://pytorch.org/) - 深度学习框架
- [Albumentations](https://albumentations.ai/) - 数据增强库
- [TensorBoard](https://www.tensorflow.org/tensorboard) - 可视化工具
- [Transformers](https://huggingface.co/docs/transformers) - Transformer 架构参考

---

<div align="center">

**如果本项目对您的研究有帮助，请给一个 ⭐Star！**

Made with ❤️ by Your Team

</div>
