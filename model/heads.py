"""
heads.py
--------
Hai head phụ trợ trong paper:
1. MLC head (Multi-Label Classification): Norm & Linear Projection từ CLS token
   -> dự đoán nhãn bệnh.
2. ITM head (Image-Text Matching): tính per-token similarity giữa ảnh & văn bản,
   áp dụng temporal weighting theo vị trí token (Eq.10-11 paper),
   trả về similarity matrix (B, B) dùng cho triplet loss (Eq.8-9).

   Temporal weighting: lambda_i = (N_r - i + 1) ** (-alpha)
   Weighted similarity: S(I,T) = sum(lambda_i * s_i) / sum(lambda_i)
   với alpha=0.5, N_r = số token hợp lệ, i = vị trí token.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLCHead(nn.Module):
    def __init__(self, d_model: int, num_labels: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, num_labels)

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(cls_token))   # (B, num_labels) logits


class ITMHead(nn.Module):
    """ITM head với per-token similarity + temporal weighting (Eq.10-11).

    Quy trình:
    1. Project img_cls → itm_dim, project mỗi txt token → itm_dim
    2. Tính per-token similarity s_i = dot(img_proj, txt_proj_i)
    3. Áp dụng temporal weight: lambda_i = (N_r - i + 1) ** (-alpha)
    4. Weighted similarity: S(I,T) = sum(lambda_i * s_i) / sum(lambda_i)
    5. Kết quả: similarity matrix (B, B) cho triplet loss
    """
    def __init__(self, d_model: int, itm_dim: int, temporal_alpha: float = 0.5):
        super().__init__()
        self.img_proj = nn.Linear(d_model, itm_dim)
        self.txt_proj = nn.Linear(d_model, itm_dim)
        self.temporal_alpha = temporal_alpha

    def forward(
        self,
        img_cls: torch.Tensor,        # (B, d_model) — CLS global từ encoder
        txt_features: torch.Tensor,    # (B, T, d_model) — per-token fused features từ decoder
        padding_mask: torch.Tensor,    # (B, T) — True = token hợp lệ, False = padding
    ) -> torch.Tensor:
        """Trả về similarity matrix (B, B) — đường chéo là cặp đúng (positive)."""
        B, T, _ = txt_features.shape

        # 1. Linear projection + L2 normalize
        img = F.normalize(self.img_proj(img_cls), dim=-1)          # (B, itm_dim)
        txt = F.normalize(self.txt_proj(txt_features), dim=-1)     # (B, T, itm_dim)

        # 2. Per-token similarity cho MỌI cặp (image_i, text_j):
        #    s[i, j, t] = dot(img[i], txt[j, t])
        per_token_sim = torch.einsum('id,jtd->ijt', img, txt)     # (B, B, T)

        # 3. Temporal weighting: lambda_i = (N_r - i + 1) ** (-alpha)
        #    N_r = số token hợp lệ của mỗi text sample j
        N_r = padding_mask.float().sum(dim=1)                      # (B,)
        positions = torch.arange(T, device=txt.device).float()     # (T,)

        # lambda_weights[j, t] = (N_r[j] - t + 1) ** (-alpha) nếu token hợp lệ, 0 nếu padding
        # Clamp để tránh giá trị âm hoặc 0 khi t >= N_r
        raw_weights = (N_r.unsqueeze(1) - positions.unsqueeze(0) + 1.0)  # (B, T)
        raw_weights = raw_weights.clamp(min=1e-8) ** (-self.temporal_alpha)
        lambda_weights = raw_weights * padding_mask.float()        # (B, T) — zero out padding

        # 4. Chuẩn hoá: lambda_norm[j, t] = lambda[j,t] / sum_t(lambda[j,t])
        lambda_sum = lambda_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)  # (B, 1)
        lambda_norm = lambda_weights / lambda_sum                  # (B, T)

        # 5. Weighted similarity: S[i, j] = sum_t(lambda_norm[j, t] * s[i, j, t])
        #    lambda_norm shape (B_txt, T) -> unsqueeze(0) -> (1, B_txt, T) for broadcasting
        sim_matrix = (per_token_sim * lambda_norm.unsqueeze(0)).sum(dim=2)  # (B, B)

        return sim_matrix
