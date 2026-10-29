"""
heads.py
--------
Hai head phụ trợ trong paper:
1. MLC head (Multi-Label Classification): Norm & Linear Projection từ CLS token
   -> dự đoán 14 nhãn bệnh CheXpert.
2. ITM head (Image-Text Matching): chiếu ảnh & văn bản về cùng không gian,
   dùng in-batch negatives (mỗi ảnh ghép cặp với báo cáo của ảnh khác làm negative).
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
    def __init__(self, d_model: int, itm_dim: int, temperature: float = 0.07):
        super().__init__()
        self.img_proj = nn.Linear(d_model, itm_dim)
        self.txt_proj = nn.Linear(d_model, itm_dim)
        self.temperature = temperature

    def forward(self, img_cls: torch.Tensor, txt_pooled: torch.Tensor) -> torch.Tensor:
        """
        img_cls:   (B, d_model)
        txt_pooled:(B, d_model)
        Trả về similarity matrix (B, B) — đường chéo là cặp đúng (positive).
        """
        img = F.normalize(self.img_proj(img_cls), dim=-1)
        txt = F.normalize(self.txt_proj(txt_pooled), dim=-1)
        return (img @ txt.t()) / self.temperature   # (B, B)
