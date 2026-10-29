"""
memory_layernorm.py
-------------------
Memory-driven LayerNorm — cơ chế đặc trưng của paper.
Khác với LayerNorm thông thường, một vector memory (có thể học) điều khiển
hệ số scale/shift của normalization, giúp decoder "nhớ" thông tin bối cảnh.

Lưu ý: paper mô tả khái niệm nhưng không công khai code. Dưới đây là cách diễn
giải hợp lý và phổ biến. Bạn (hoặc AI assistant) có thể tinh chỉnh sau.
"""
import torch
import torch.nn as nn


class MemoryDrivenLayerNorm(nn.Module):
    def __init__(self, d_model: int, memory_dim: int = 64, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(d_model, eps=eps)
        # Vector memory dùng chung cho toàn bộ layer (có thể học)
        self.memory = nn.Parameter(torch.randn(memory_dim) * 0.02)
        self.to_gamma = nn.Linear(memory_dim, d_model)
        self.to_beta = nn.Linear(memory_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = self.ln(x)
        gamma = self.to_gamma(self.memory)  # (d_model,)
        beta = self.to_beta(self.memory)    # (d_model,)
        return x * (1.0 + gamma) + beta
