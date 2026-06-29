"""
models/densenet_cbam.py
=======================
Definisi arsitektur DenseNet121 + CBAM.
Kode ini merupakan konversi LANGSUNG dari notebook penelitian (Cell 8),
tanpa perubahan logika apapun.

Referensi:
  - Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.
  - Huang et al., "Densely Connected Convolutional Networks", CVPR 2017.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet121_Weights

from config import DROPOUT, CBAM_REDUCTION, CBAM_KERNEL


# ── CBAM Sub-Modules ──────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    """
    Channel Attention Module (Woo et al. 2018).
    Menggunakan average-pool dan max-pool secara paralel,
    lalu dijumlahkan sebelum sigmoid.
    """
    def __init__(self, channels: int, reduction: int = CBAM_REDUCTION):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.size(0), x.size(1)
        avg = self.mlp(self.avg_pool(x).view(b, c))
        mx  = self.mlp(self.max_pool(x).view(b, c))
        return self.sigmoid(avg + mx).view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (Woo et al. 2018).
    Concat channel-wise avg dan max, lalu conv 7×7.
    """
    def __init__(self, kernel_size: int = CBAM_KERNEL):
        super().__init__()
        padding = kernel_size // 2
        self.conv    = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1)[0].unsqueeze(1)
        concat  = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(concat))


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (Woo et al., ECCV 2018).
    Urutan: Channel Attention → Spatial Attention (sequential).
    """
    def __init__(self, channels: int,
                 reduction: int = CBAM_REDUCTION,
                 kernel_size: int = CBAM_KERNEL):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x


# ── Model Utama ───────────────────────────────────────────────────────────────

class DenseNet121CBAM(nn.Module):
    """
    DenseNet121 + CBAM untuk klasifikasi biner stroke.
    CBAM disisipkan setelah seluruh feature extractor DenseNet121
    (output 1024 channel dari denseblock4).

    Arsitektur head:
        Dropout(0.35) → Linear(1024→512) → BN → ReLU
        → Dropout(0.175) → Linear(512→1)

    Output: logit skalar (sebelum sigmoid).
    Gunakan torch.sigmoid() saat inference untuk mendapat probabilitas.
    """

    def __init__(self, dropout: float = DROPOUT, freeze_backbone: bool = False):
        super().__init__()

        # Backbone DenseNet121 pretrained ImageNet
        base = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        self.features = base.features   # output: (B, 1024, 7, 7) untuk input 224×224

        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM(1024)
        self.gap  = nn.AdaptiveAvgPool2d(1)   # Global Average Pooling → (B, 1024, 1, 1)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, 1),
        )

        # Freeze backbone jika diminta (tidak digunakan saat inference)
        if freeze_backbone:
            for name, param in self.features.named_parameters():
                frozen_blocks = [
                    "conv0", "norm0",
                    "denseblock1", "transition1",
                    "denseblock2", "transition2",
                ]
                if any(name.startswith(blk) for blk in frozen_blocks):
                    param.requires_grad = False

    def unfreeze(self):
        """Unfreeze semua parameter (digunakan saat fine-tuning)."""
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        feat = self.features(x)      # (B, 1024, H', W')
        feat = self.relu(feat)

        # CBAM attention
        feat = self.cbam(feat)       # (B, 1024, H', W')

        # Pooling + flatten
        feat = self.gap(feat)        # (B, 1024, 1, 1)
        feat = feat.view(feat.size(0), -1)   # (B, 1024)

        # Classification head
        out = self.head(feat)        # (B, 1)
        return out