# 多视角鹅体尺测量模型 (Multi-View Goose Body Measurement)

基于 Transformer 架构的多视角图像融合模型，用于预测鹅的体尺测量值。缺少一些推理文件可联系我来补充

## 特性

- 多视角图像融合：使用 Transformer 编码器融合多个视角的图像特征
- 灵活的主干网络：支持 ResNet50 和 EfficientNet-B4
- 数据增强：集成 Albumentations 数据增强库
- 训练监控：支持 TensorBoard 可视化
- 早停机制：自动停止训练防止过拟合

## 环境要求

- Python >= 3.8
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- CUDA (推荐)

## 安装

```bash
# 克隆仓库
git clone https://github.com/lifeixiangabc-spec/goose-body-measurement.git
cd goose-body-measurement

# 创建虚拟环境
conda create -n goosemodel python=3.10
conda activate goosemodel

# 安装依赖
pip install -r requirements.txt
```

## 数据集格式

数据集应按以下结构组织：

```
data/
├── sample_001/
│   ├── view1.jpg          # 视角1图像
│   ├── view2.jpg          # 视角2图像
│   ├── view3.jpg          # 视角3图像
│   ├── view4.jpg          # 视角4图像
│   └── label.txt          # 测量值标签（空格分隔）
├── sample_002/
│   └── ...
└── ...
```

`label.txt` 文件格式：
```
1.5809 1.8821 1.9229
```

每行为测量值，用空格分隔。

### 自定义数据格式

如果你的数据格式不同，可以通过参数指定：

```bash
python train.py \
    --data_dir your_data/ \
    --image_prefix render_angle_ \
    --image_suffix .png \
    --label_file dimensions.txt \
    --num_measurements 3
```

## 训练

### 基本训练

```bash
python train.py --data_dir data/
```

### 完整参数

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

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | data/ | 数据集目录 |
| `--output_dir` | outputs | 输出目录 |
| `--backbone` | resnet50 | 主干网络 (resnet50/efficientnet_b4) |
| `--num_views` | 4 | 视角数量 |
| `--num_measurements` | 5 | 测量值数量 |
| `--epochs` | 50 | 训练轮数 |
| `--batch_size` | 8 | 批量大小 |
| `--lr` | 1e-4 | 学习率 |
| `--patience` | 10 | 早停耐心值 |
| `--resume` | None | 恢复训练的检查点路径 |

## 模型结构

```
MultiViewGooseTransformer
├── Backbone (ResNet50/EfficientNet-B4)  # 特征提取
├── TransformerEncoder                    # 多视角特征融合
│   └── 2 layers, 8 heads
└── Regressor                             # 测量值回归
    ├── Linear(2048 -> 512)
    ├── ReLU + Dropout
    ├── Linear(512 -> 256)
    ├── ReLU + Dropout
    └── Linear(256 -> num_measurements)
```

## 项目结构

```
goose-body-measurement/
├── models/
│   └── model.py          # 模型定义
├── data/                  # 数据目录
├── train.py              # 训练脚本
├── pytorch_dataset.py    # 数据集类
├── requirements.txt      # 依赖
├── README.md             # 说明文档
└── LICENSE               # 许可证
```

## 监控训练

```bash
tensorboard --logdir outputs/runs
```

## 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。
