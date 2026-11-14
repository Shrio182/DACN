# vit_encoder.py
# Vision Transformer Encoder thuần túy (không dùng CNN pretrained, theo IEEE TMI 2022).
# Luồng xử lý:
# 1. Cắt ảnh thành các patch vuông (patch_size = 32x32), làm phẳng và chiếu tuyến tính lên không gian d_model (512).
# 2. Thêm vector học được [CLS] token vào đầu chuỗi + cộng vector mã hóa vị trí (Positional Embedding).
# 3. Cho qua 3 tầng Transformer Encoder (Multi-Head Self-Attention + Feed Forward Network + LayerNorm).
# 4. Hỗ trợ xử lý đồng thời cặp ảnh 2 góc chụp (Frontal + Lateral) với trọng số chia sẻ (Shared Weights).

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    # Tách ảnh thành các vùng cục bộ (Patches) và chiếu tuyến tính lên chiều ẩn d_model
    def __init__(self, img_size: int, patch_size: int, in_channels: int, d_model: int):
        super().__init__()
        assert img_size % patch_size == 0, "img_size phải chia hết cho patch_size"
        self.num_patches = (img_size // patch_size) ** 2
        # Chiếu trực tiếp flattened patch (P*P*C) -> d_model theo Fig.1 của bài báo
        self.proj = nn.Linear(patch_size * patch_size * in_channels, d_model)
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Đầu vào: x có kích thước (B, C, H, W)
        # Đầu ra: (B, num_patches, d_model), với 448x448 patch 32 -> num_patches = 196
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)               # (B, C, nH, nW, p, p)
        x = x.reshape(B, C, -1, p * p).permute(0, 2, 1, 3)  # (B, num_patches, C, p*p)
        x = x.reshape(B, -1, C * p * p)                     # (B, num_patches, C*p*p)
        return self.proj(x)


class TransformerEncoderLayer(nn.Module):
    # Khối Transformer Encoder tiêu chuẩn: Self-Attention -> Add&Norm -> FFN -> Add&Norm
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-Head Self-Attention với residual connection và post-layer norm
        a, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + a)
        x = self.norm2(x + self.ffn(x))
        return x


class ViTEncoder(nn.Module):
    # Bộ trích xuất đặc trưng thị giác hoàn chỉnh từ cặp ảnh X-quang
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbedding(cfg.img_size, cfg.patch_size, cfg.in_channels, cfg.d_model)
        num_patches = self.patch_embed.num_patches
        # [CLS] token đại diện thông tin tổng quan của ảnh (dùng cho phân loại bệnh lý MLC & ITM)
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        # Vector nhúng vị trí không gian cho 196 patches + 1 [CLS] token
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, cfg.d_model) * 0.02)
        # 3 lớp Transformer Encoder xếp chồng
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(cfg.d_model, cfg.nhead, cfg.dim_feedforward, cfg.dropout)
            for _ in range(cfg.num_encoder_layers)
        ])
        self.norm = nn.LayerNorm(cfg.d_model)

    def encode_single(self, x: torch.Tensor):
        # Trích xuất đặc trưng cho một góc chụp đơn lẻ: (B, C, H, W) -> regional (B, N, d), cls (B, d)
        B = x.size(0)
        x = self.patch_embed(x)                                    # (B, N, d)
        cls = self.cls_token.expand(B, -1, -1)                     # (B, 1, d)
        x = torch.cat([cls, x], dim=1) + self.pos_embed            # (B, N+1, d)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x[:, 1:], x[:, 0]                                   # regional features, cls token

    def forward(self, images: torch.Tensor):
        # Đầu vào:
        # - Ảnh đơn: (B, C, H, W) hoặc Cặp ảnh 2 góc: (B, V, C, H, W) với V=2 (Frontal + Lateral)
        # Đầu ra:
        # - regional_features: (B, V*N, d_model) — chứa 392 token đặc trưng vùng cho Cross-Attention
        # - cls_global: (B, d_model) — trung bình cộng vector CLS để phân loại bệnh lý và so khớp ITM
        if images.dim() == 4:
            images = images.unsqueeze(1)                            # (B, 1, C, H, W)
        B, V = images.shape[:2]
        regionals, clss = [], []
        # Chạy qua cùng 1 ViT Encoder (chia sẻ trọng số giữa các góc nhìn)
        for v in range(V):
            reg, cls = self.encode_single(images[:, v])
            regionals.append(reg)
            clss.append(cls)
        # Ghép đặc trưng vùng của các view lại với nhau (ví dụ: 196 + 196 = 392 vùng)
        regional_features = torch.cat(regionals, dim=1)             # (B, V*N, d)
        cls_global = torch.stack(clss, dim=0).mean(dim=0)           # (B, d)
        return regional_features, cls_global
