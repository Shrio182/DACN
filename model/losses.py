# losses.py
# Ba hàm mất mát cốt lõi trong hệ thống giám sát đa tiêu chí (IEEE TMI 2022):
# 1. RG Loss (Report Generation): Cross-Entropy có đánh trọng số TF-IDF cho từ vựng (Công thức 4-5).
# 2. MLC Loss (Multi-Label Classification): Phân loại nhị phân đa nhãn bệnh lý (Công thức 6).
# 3. ITM Loss (Image-Text Matching): Triplet Loss với lề margin và chuẩn hóa L2 (Công thức 8-9).

import torch
import torch.nn.functional as F


def term_weighted_cross_entropy(
    logits: torch.Tensor,        # (B, T, V) logits dự đoán từ decoder
    targets: torch.Tensor,       # (B, T) nhãn từ ground-truth
    term_weight_table: torch.Tensor,  # (V,) bảng trọng số TF-IDF tính từ tập dữ liệu
    pad_idx: int,
) -> torch.Tensor:
    # Hàm mất mát sinh từ vựng có đánh trọng số (Term-Weighted Cross-Entropy)
    # Các từ hiếm chỉ dấu bệnh lý (như pneumonia, cardiomegaly, effusion) có trọng số cao hơn,
    # giúp mô hình không bị thiên lệch vào các từ dừng phổ biến (the, is, no, are).
    B, T, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    targets_flat = targets.reshape(-1)
    
    # Tính Cross-Entropy cơ sở cho từng token
    loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=pad_idx, reduction="none")
    
    # Lấy trọng số tương ứng của từng từ mục tiêu
    weights = term_weight_table[targets_flat]
    mask = (targets_flat != pad_idx).float()
    
    # Tính trung bình tổn thất có trọng số trên các token hợp lệ
    return (loss * weights * mask).sum() / mask.sum().clamp(min=1)


def mlc_bce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # Hàm mất mát phân loại bệnh lý đa nhãn (Multi-Label Classification Loss)
    # labels: ma trận nhị phân (B, num_labels) mang giá trị 0 hoặc 1
    return F.binary_cross_entropy_with_logits(logits, labels)


def itm_triplet_loss(
    sim_matrix: torch.Tensor,   # (B, B) ma trận độ tương đồng cosine giữa ảnh và văn bản
    epsilon: float = 0.05,      # lề cách biệt Triplet Margin (theo bài báo: epsilon = 0.05)
    mu: float = 0.025,          # hệ số chuẩn hóa Frobenius norm (theo bài báo: mu = 0.025)
) -> torch.Tensor:
    # Hàm tổn thất đối sánh ảnh - văn bản sử dụng kĩ thuật Hard Negative Mining:
    # Với mỗi cặp đúng (i == j, đường chéo chính), tìm mẫu âm tính khó nhất:
    # - Ảnh đúng đi với văn bản sai của bệnh nhân khác (Negative Text)
    # - Văn bản đúng đi với ảnh sai của bệnh nhân khác (Negative Image)
    # Tổng tổn thất = Triplet Margin Loss + Chuẩn hóa L2 ma trận tương đồng
    B = sim_matrix.size(0)
    
    # Cặp dương tính: các phần tử nằm trên đường chéo chính
    pos = sim_matrix.diagonal()  # (B,)

    # Tạo mặt nạ loại trừ đường chéo chính để tìm cặp âm tính khó nhất
    mask = torch.eye(B, device=sim_matrix.device).bool()
    sim_neg = sim_matrix.masked_fill(mask, -float("inf"))

    # Khai phá mẫu âm tính khó nhất (Hardest Negative)
    neg_txt = sim_neg.max(dim=1).values  # ảnh i đi với văn bản j khác khó nhất
    neg_img = sim_neg.max(dim=0).values  # văn bản i đi với ảnh j khác khó nhất

    # Tính tổn thất Triplet: max(0, margin - pos + neg)
    loss_txt = F.relu(epsilon - pos + neg_txt).mean()
    loss_img = F.relu(epsilon - pos + neg_img).mean()
    triplet_loss = (loss_txt + loss_img) / 2.0

    # Chuẩn hóa Frobenius norm để ổn định ma trận biểu diễn đặc trưng
    reg_loss = mu * (sim_matrix ** 2).sum() / (B * B)

    return triplet_loss + reg_loss
