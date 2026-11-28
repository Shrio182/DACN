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

    def forward(self, x: torch.Tensor, capture_attention: bool = False) -> torch.Tensor:
        # Multi-Head Self-Attention với residual connection và post-layer norm
        a, attention = self.attn(
            x,
            x,
            x,
            need_weights=capture_attention,
            average_attn_weights=True,
        )
        if capture_attention:
            self.last_attention = attention
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
        for index, layer in enumerate(self.layers):
            x = layer(x, capture_attention=(index == len(self.layers) - 1))
        x = self.norm(x)
        return x[:, 1:], x[:, 0]                                   # regional features, cls token

    @staticmethod
    def _suppress_border_attention(grid: torch.Tensor, border_scale: float = 0.15) -> torch.Tensor:
        """Làm dịu 1 hàng/cột viền ngoài cùng (attention sink ở nền đen/mô mềm ngoài lồng ngực)."""
        if grid.size(-1) < 3 or grid.size(-2) < 3:
            return grid
        attenuated = grid.clone()
        interior_mean = attenuated[..., 1:-1, 1:-1].mean(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        cap = 0.5 * interior_mean
        mask = torch.ones_like(attenuated)
        mask[..., 0, :] = border_scale
        mask[..., -1, :] = border_scale
        mask[..., :, 0] = border_scale
        mask[..., :, -1] = border_scale
        attenuated = attenuated * mask
        # Cap outer ring so sharp border sinks cannot exceed lung interior scale
        border_capped = torch.minimum(attenuated, cap)
        is_border = mask < 1.0
        attenuated = torch.where(is_border, border_capped, attenuated)
        return attenuated / attenuated.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)

    @staticmethod
    def _gaussian_blur_grid(grid: torch.Tensor) -> torch.Tensor:
        """Làm mượt nhẹ bản đồ chú ý 2D bằng bộ lọc Gaussian 3x3 trước khi nội suy."""
        b, v, h, w = grid.shape
        kernel = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
            device=grid.device,
            dtype=grid.dtype,
        ) / 16.0
        kernel = kernel.view(1, 1, 3, 3)
        flat = grid.reshape(b * v, 1, h, w)
        padded = torch.nn.functional.pad(flat, (1, 1, 1, 1), mode="replicate")
        smoothed = torch.nn.functional.conv2d(padded, kernel)
        return smoothed.reshape(b, v, h, w)

    @staticmethod
    def _minmax_norm_grid(grid: torch.Tensor) -> torch.Tensor:
        g_min = grid.amin(dim=(-2, -1), keepdim=True)
        g_max = grid.amax(dim=(-2, -1), keepdim=True)
        return (grid - g_min) / (g_max - g_min + 1e-8)

    @torch.no_grad()
    def get_attention_heatmap(
        self,
        images: torch.Tensor,
        img_size: int = 448,
        cross_attention: torch.Tensor = None,
        return_patch_grid: bool = False,
    ):
        """
        Trích trọng số Self-Attention từ [CLS] tới lưới patch ở tầng Encoder cuối (B,V,H,W),
        triệt tiêu attention sink ở viền ngoài cùng, ưu tiên kết hợp Decoder Cross-Attention nếu có,
        và làm mượt Gaussian nhẹ trước khi nội suy.
        """
        if images.dim() == 4:
            images = images.unsqueeze(1)
        if images.dim() != 5:
            raise ValueError("Expected (B,C,H,W) or (B,V,C,H,W)")

        batch_size, view_count = images.shape[:2]
        heatmaps = []
        for view_index in range(view_count):
            view = images[:, view_index]
            patches = self.patch_embed(view)
            cls = self.cls_token.expand(batch_size, -1, -1)
            tokens = torch.cat((cls, patches), dim=1) + self.pos_embed

            for index, layer in enumerate(self.layers):
                tokens = layer(
                    tokens,
                    capture_attention=(index == len(self.layers) - 1),
                )
            tokens = self.norm(tokens)

            cls_attention = self.layers[-1].last_attention[:, 0, 1:]
            cls_attention = cls_attention / cls_attention.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
            side = int(cls_attention.size(-1) ** 0.5)
            if side * side != cls_attention.size(-1):
                raise RuntimeError("Patch count cannot form a square attention grid")
            grid = cls_attention.reshape(batch_size, 1, side, side)
            grid = self._suppress_border_attention(grid)
            heatmaps.append(grid)

        patch_grid = torch.cat(heatmaps, dim=1)  # (B, V, side, side)

        if cross_attention is not None:
            cross = cross_attention.to(device=patch_grid.device, dtype=patch_grid.dtype)
            if cross.dim() == 3:
                cross = cross.unsqueeze(1)
            if cross.size(1) == 1 and view_count > 1:
                cross = cross.expand(-1, view_count, -1, -1)
            cross = self._suppress_border_attention(cross)
            patch_grid = 0.65 * self._minmax_norm_grid(cross) + 0.35 * self._minmax_norm_grid(patch_grid)
            patch_grid = patch_grid / patch_grid.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)

        smoothed_grid = self._gaussian_blur_grid(patch_grid)
        upsampled = torch.nn.functional.interpolate(
            smoothed_grid,
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        )
        if return_patch_grid:
            return upsampled, smoothed_grid
        return upsampled

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
