# heads.py
# Các đầu dự đoán phụ trợ (Auxiliary Heads) cho mô hình giám sát đa tiêu chí:
# 1. MLC Head (Multi-Label Classification): Dự đoán xác suất các nhãn bệnh lý (MeSH tags).
# 2. ITM Head (Image-Text Matching): Ghép nối sự tương đồng giữa ảnh và từng từ trong báo cáo,
#    kèm cơ chế trọng số thời gian theo vị trí từ (Temporal Weighting theo Công thức 10-11).

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLCHead(nn.Module):
    # Đầu phân loại đa nhãn bệnh lý từ vector CLS toàn cục của ảnh X-quang
    def __init__(self, d_model: int, num_labels: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, num_labels)

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        # Đầu vào: cls_token (B, d_model) trích xuất từ ViT Encoder
        # Đầu ra: Logits chưa qua hàm sigmoid có kích thước (B, num_labels)
        return self.proj(self.norm(cls_token))


class ITMHead(nn.Module):
    # Đầu so khớp ảnh - văn bản với trọng số thời gian (Temporal-Weighted ITM)
    # Quy trình toán học theo Công thức 10 và 11 của bài báo gốc:
    # 1. Chiếu vector CLS ảnh và đặc trưng từng từ văn bản sang không gian tương đồng d_itm (512).
    # 2. Chuẩn hóa L2 rồi tính độ tương đồng cosine từng từ: s_i = dot(img, txt_i).
    # 3. Tính trọng số thời gian tăng dần về cuối câu: lambda_i = (N_r - i + 1) ** (-alpha).
    #    (Các từ ở cuối câu thường là kết luận bệnh lý quan trọng nhất nên nhận trọng số cao).
    # 4. Trả về ma trận tương đồng (B, B) để tối ưu Triplet Loss.
    def __init__(self, d_model: int, itm_dim: int, temporal_alpha: float = 0.5):
        super().__init__()
        self.img_proj = nn.Linear(d_model, itm_dim)
        self.txt_proj = nn.Linear(d_model, itm_dim)
        self.temporal_alpha = temporal_alpha

    def forward(
        self,
        img_cls: torch.Tensor,        # (B, d_model) - vector CLS toàn cục của ảnh
        txt_features: torch.Tensor,    # (B, T, d_model) - đặc trưng từng token từ Decoder
        padding_mask: torch.Tensor,    # (B, T) - True là token hợp lệ, False là token padding
    ) -> torch.Tensor:
        B, T, _ = txt_features.shape

        # Bước 1: Chiếu tuyến tính và chuẩn hóa L2 trên trục vector
        img = F.normalize(self.img_proj(img_cls), dim=-1)          # (B, itm_dim)
        txt = F.normalize(self.txt_proj(txt_features), dim=-1)     # (B, T, itm_dim)

        # Bước 2: Tính tích vô hướng tương đồng giữa mọi cặp ảnh i và văn bản j
        # sim_per_token[i, j, t] là độ tương đồng giữa ảnh i và token t của văn bản j
        sim_per_token = torch.einsum("bd,jtd->bjt", img, txt)      # (B, B, T)

        # Bước 3: Tính trọng số thời gian lambda_i = (N_r - i + 1)^(-alpha)
        # N_r là số token thực tế trong câu của mẫu j (loại trừ các token padding)
        valid_lengths = padding_mask.sum(dim=-1)                   # (B,)
        pos = torch.arange(T, device=txt_features.device).unsqueeze(0).expand(B, -1)  # (B, T)
        dist_to_end = (valid_lengths.unsqueeze(1) - pos).float()   # N_r - i + 1

        # Gán trọng số 0 cho các vị trí padding hoặc vượt quá độ dài câu
        valid_pos_mask = (dist_to_end >= 1.0) & padding_mask       # (B, T)
        dist_to_end_safe = dist_to_end.clamp(min=1.0)
        temporal_weights = dist_to_end_safe.pow(-self.temporal_alpha) * valid_pos_mask.float()

        # Bước 4: Tính trung bình có trọng số độ tương đồng ảnh-văn bản: S(I, T) = sum(w * s) / sum(w)
        w = temporal_weights.unsqueeze(0).expand(B, -1, -1)         # (B, B, T)
        w_sum = w.sum(dim=-1).clamp(min=1e-6)                       # (B, B)
        sim_matrix = (sim_per_token * w).sum(dim=-1) / w_sum        # (B, B)

        # Trả về ma trận đối sánh: đường chéo chính (i == j) là các cặp khớp đúng (Positive pairs)
        return sim_matrix
