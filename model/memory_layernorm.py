# memory_layernorm.py
# Memory-Driven LayerNorm (Cơ chế chuẩn hóa lớp điều khiển bởi bộ nhớ)
# Kiến trúc cốt lõi được nhóm tác giả IEEE TMI 2022 đề xuất:
# Thay vì giữ tham số scale (gamma) và shift (beta) tĩnh như LayerNorm cổ điển,
# một vector bộ nhớ có thể học (Memory Parameter) sẽ tự động điều biến hệ số
# scale/shift thích ứng theo từng ngữ cảnh biểu diễn đặc thù của báo cáo y khoa.

import torch
import torch.nn as nn


class MemoryDrivenLayerNorm(nn.Module):
    # Khối chuẩn hóa thích ứng với vector memory học được
    def __init__(self, d_model: int, memory_dim: int = 64, eps: float = 1e-5):
        super().__init__()
        # Chuẩn hóa giá trị trung bình = 0 và độ lệch chuẩn = 1
        self.ln = nn.LayerNorm(d_model, eps=eps)
        # Vector memory dùng chung cho cả khối (có thể tối ưu qua lan truyền ngược)
        self.memory = nn.Parameter(torch.randn(memory_dim) * 0.02)
        # Hai tầng tuyến tính chiếu vector memory sang không gian gamma và beta
        self.to_gamma = nn.Linear(memory_dim, d_model)
        self.to_beta = nn.Linear(memory_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Đầu vào: x có kích thước (B, T, d_model)
        # Đầu ra: x đã được chuẩn hóa và điều biến bởi gamma & beta học từ memory
        x = self.ln(x)
        gamma = self.to_gamma(self.memory)  # (d_model,)
        beta = self.to_beta(self.memory)    # (d_model,)
        # Phép biến đổi Affine điều biến: y = x * (1 + gamma) + beta
        return x * (1.0 + gamma) + beta
