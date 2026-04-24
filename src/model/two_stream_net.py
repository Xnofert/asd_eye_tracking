from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


def _build_branch_rgb() -> nn.Module:
    """构建 RGB 语义分支：ResNet-50 去掉 avgpool 和 fc，保留空间特征图。
    输入: (B, 3, 224, 224) → 输出: (B, 2048, 7, 7)
    """
    root_dir = Path(__file__).resolve().parent.parent.parent
    # resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)  # 尝试从服务器下载预训练权重
    resnet = models.resnet50(weights=str(root_dir / "outputs" / "resnet50" / "resnet50-11ad3fa6.pth"))  # 本地权重
    # children()[:-2] 去掉最后的 AdaptiveAvgPool2d 和 Linear
    modules = list(resnet.children())[:-2]
    return nn.Sequential(*modules)


class GazeCNN(nn.Module):
    """轻量级 5 层 CNN，提取热力图空间注意力分布特征。
    每层 stride=2 逐步降采样，通道数由 channels 参数控制。
    """

    # 各层固定的 (kernel_size, padding) 配置
    LAYER_CONFIGS = [
        (7, 3),  # Layer 1: 大感受野
        (5, 2),  # Layer 2
        (3, 1),  # Layer 3
        (3, 1),  # Layer 4
        (3, 1),  # Layer 5
    ]

    def __init__(self, channels: list[int] | None = None):
        super().__init__()
        channels = channels or [32, 64, 128, 256, 256]

        layers: list[nn.Module] = []
        in_ch = 1  # 热力图单通道输入
        for out_ch, (k, p) in zip(channels, self.LAYER_CONFIGS):
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=2, padding=p, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ])
            in_ch = out_ch

        self.features = nn.Sequential(*layers)
        self.out_channels = channels[-1]
        self._init_weights()

    def _init_weights(self):
        """He 正态分布初始化卷积层，BatchNorm 权重=1 偏置=0。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class TwoStreamNet(nn.Module):
    """双路特征提取 + 后期融合分类网络。

    数据流 (默认配置):
      RGB     (B,3,224,224) → ResNet-50 → (B,2048,7,7) → GAP → (B,2048) → Proj → (B,256)  ─┐
      Heatmap (B,1,224,224) → GazeCNN  → (B,256,7,7)  → GAP → (B,256)                      ─┤
                                                                  Concat → (B,512)
                                                            Linear+ReLU → (B,512)
                                                                Dropout → (B,512)
                                                                 Linear → (B,1)  logits

    通过 model_cfg 可自定义 gaze_channels、rgb_proj_dim、classifier_hidden_dim、dropout。
    """

    RGB_FEAT_DIM = 2048   # ResNet-50 layer4 输出通道数

    def __init__(self, model_cfg: dict | None = None):
        super().__init__()
        model_cfg = model_cfg or {}
        gaze_channels = model_cfg.get("gaze_channels", [32, 64, 128, 256, 256])
        hidden_dim = model_cfg.get("classifier_hidden_dim", 512)
        dropout = model_cfg.get("dropout", 0.5)
        rgb_proj_dim = model_cfg.get("rgb_proj_dim", 256)

        self.branch_rgb = _build_branch_rgb()
        self.branch_gaze = GazeCNN(channels=gaze_channels)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 将 RGB 特征从 2048 维投影到与热力图分支对等的维度，防止淹没热力图信号
        gaze_feat_dim = self.branch_gaze.out_channels
        self.rgb_proj = nn.Sequential(
            nn.Linear(self.RGB_FEAT_DIM, rgb_proj_dim),
            nn.ReLU(inplace=True),
        )

        # 融合后的全连接分类头
        fused_dim = rgb_proj_dim + gaze_feat_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self._init_classifier()

    def _init_classifier(self):
        for m in list(self.rgb_proj.modules()) + list(self.classifier.modules()):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, rgb: torch.Tensor, heatmap: torch.Tensor) -> torch.Tensor:
        """前向传播，输出原始 logits（未经 sigmoid），配合 BCEWithLogitsLoss 使用。"""
        feat_rgb = self.pool(self.branch_rgb(rgb)).flatten(1)
        feat_rgb = self.rgb_proj(feat_rgb)
        feat_gaze = self.pool(self.branch_gaze(heatmap)).flatten(1)
        fused = torch.cat([feat_rgb, feat_gaze], dim=1)
        return self.classifier(fused)

    def freeze_branch_rgb(self):
        """冻结 ResNet-50 分支权重（训练前 N 个 epoch 仅训练 GazeCNN + 分类头）。"""
        for param in self.branch_rgb.parameters():
            param.requires_grad = False

    def unfreeze_branch_rgb(self):
        """解冻 ResNet-50 分支，开始端到端微调。"""
        for param in self.branch_rgb.parameters():
            param.requires_grad = True
