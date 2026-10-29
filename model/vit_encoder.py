"""
vit_encoder.py
--------------
Vision Transformer encoder (không dùng CNN — "pure transformer" như paper).
- Linear projection of flattened patches + position embedding + <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token
- 3 lớp transformer encoder (MHSA + Add&Norm + FFN + Add&Norm)
- Hỗ trợ cả ảnh đơn (MIMIC-CXR) và cặp 2 view (IU-Xray: frontal + lateral)
"""
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_channels: int, d_model: int):
        super().__init__()
        assert img_size % patch_size == 0, "img_size phải chia hết cho patch_size"
        self.num_patches = (img_size // patch_size) ** 2
        # "Linear Projection of Flattened Patches" (theo đúng Fig.1)
        self.proj = nn.Linear(patch_size * patch_size * in_channels, d_model)
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, num_patches, d_model)
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)          # (B,C,nH,nW,p,p)
        x = x.reshape(B, C, -1, p * p).permute(0, 2, 1, 3)  # (B, num_patches, C, p*p)
        x = x.reshape(B, -1, C * p * p)                # (B, num_patches, C*p*p)
        return self.proj(x)


class TransformerEncoderLayer(nn.Module):
    """1 khối: Multi-Head Attention -> Add&Norm -> Feed Forward -> Add&Norm (post-norm)."""
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
        a, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + a)
        x = self.norm2(x + self.ffn(x))
        return x


class ViTEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbedding(cfg.img_size, cfg.patch_size, cfg.in_channels, cfg.d_model)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, cfg.d_model) * 0.02)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(cfg.d_model, cfg.nhead, cfg.dim_feedforward, cfg.dropout)
            for _ in range(cfg.num_encoder_layers)
        ])
        self.norm = nn.LayerNorm(cfg.d_model)

    def encode_single(self, x: torch.Tensor):
        """x: (B, C, H, W) -> regional_features (B, N, d), cls (B, d)."""
        B = x.size(0)
        x = self.patch_embed(x)                                   # (B, N, d)
        cls = self.cls_token.expand(B, -1, -1)                    # (B, 1, d)
        x = torch.cat([cls, x], dim=1) + self.pos_embed           # (B, N+1, d)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x[:, 1:], x[:, 0]                                  # regional, cls

    def forward(self, images: torch.Tensor):
        """
        images: (B, C, H, W) ảnh đơn  HOẶC  (B, V, C, H, W) nhiều view (IU-Xray V=2).
        Trả về:
            regional_features: (B, V*N, d_model)  — đầu vào cross-attention của decoder
            cls_global:        (B, d_model)       — dùng cho MLC head & ITM head
        """
        if images.dim() == 4:
            images = images.unsqueeze(1)                           # (B, 1, C, H, W)
        B, V = images.shape[:2]
        regionals, clss = [], []
        for v in range(V):
            reg, cls = self.encode_single(images[:, v])
            regionals.append(reg)
            clss.append(cls)
        regional_features = torch.cat(regionals, dim=1)            # (B, V*N, d)
        cls_global = torch.stack(clss, dim=0).mean(dim=0)          # (B, d)
        return regional_features, cls_global
