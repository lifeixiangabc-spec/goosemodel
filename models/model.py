"""多视角鹅体尺测量模型

该模型使用Transformer架构融合多个视角的图像特征，用于预测鹅的体尺测量值。
"""
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, efficientnet_b4, EfficientNet_B4_Weights


class MultiViewGooseTransformer(nn.Module):
    """多视角鹅体尺测量Transformer模型
    
    参数:
        num_views (int): 视角数量，默认4
        num_measurements (int): 体尺测量值数量，默认5
        hidden_dim (int): 隐藏层维度，默认512
        backbone_name (str): 主干网络名称，可选'resnet50'或'efficientnet_b4'
    """
    
    SUPPORTED_BACKBONES = ['resnet50', 'efficientnet_b4']
    
    def __init__(self, num_views=4, num_measurements=5, hidden_dim=512, backbone_name='resnet50'):
        super().__init__()
        self.num_views = num_views
        self.backbone_name = backbone_name
        
        if backbone_name not in self.SUPPORTED_BACKBONES:
            raise ValueError(f"不支持的主干网络: {backbone_name}，请选择 {self.SUPPORTED_BACKBONES}")
        
        self.feature_extractor, self.feature_dim = self._create_backbone(backbone_name)
        
        self.view_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.feature_dim,
                nhead=8,
                dim_feedforward=hidden_dim,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_measurements)
        )
        
        self._init_weights()
    
    def _create_backbone(self, backbone_name):
        """创建主干网络特征提取器"""
        if backbone_name == 'resnet50':
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
            feature_dim = backbone.fc.in_features
        elif backbone_name == 'efficientnet_b4':
            backbone = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
            feature_extractor = nn.Sequential(
                *list(backbone.features.children()),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            feature_dim = backbone.classifier[1].in_features
        
        return feature_extractor, feature_dim
    
    def _init_weights(self):
        """初始化回归头权重"""
        for m in self.regressor.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播
        
        参数:
            x (torch.Tensor): 输入张量，形状为 [batch_size, num_views, 3, H, W]
            
        返回:
            torch.Tensor: 预测的体尺测量值，形状为 [batch_size, num_measurements]
        """
        batch_size, num_views, c, h, w = x.shape
        
        x = x.view(batch_size * num_views, c, h, w)
        features = self.feature_extractor(x)
        
        if len(features.shape) == 4:
            features = features.view(features.size(0), -1)
        
        features = features.view(batch_size, num_views, -1)
        
        encoded = self.view_encoder(features)
        aggregated = encoded.mean(dim=1)
        
        output = self.regressor(aggregated)
        return output
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'backbone': self.backbone_name,
            'feature_dim': self.feature_dim,
            'num_views': self.num_views,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'num_measurements': self.regressor[-1].out_features
        }
        
        return info
    
    def print_model_info(self):
        """打印模型信息"""
        info = self.get_model_info()
        print(f"模型信息:")
        print(f"  主干网络: {info['backbone']}")
        print(f"  特征维度: {info['feature_dim']}")
        print(f"  视角数量: {info['num_views']}")
        print(f"  输出维度: {info['num_measurements']}")
        print(f"  总参数量: {info['total_params']:,}")
        print(f"  可训练参数: {info['trainable_params']:,}")
