"""
losses.py
---------
Ba thành phần loss của paper IEEE TMI 2022:
1. RG loss  : cross-entropy sinh báo cáo, có term-weighting (Eq.4–5).
2. MLC loss : binary cross-entropy dự đoán nhãn bệnh (Eq.6).
3. ITM loss : image-text matching với triplet margin ε=0.05 + L2-reg μ=0.025 (Eq.8–9).
              Nhân với temporal weight α=0.5 tăng dần theo epoch (Eq.10–11).
"""
import math
import torch
import torch.nn.functional as F


def term_weighted_cross_entropy(
    logits: torch.Tensor,        # (B, T, V)
    targets: torch.Tensor,       # (B, T)
    term_weight_table: torch.Tensor,  # (V,) — tính từ tần số từ trong dataset
    pad_idx: int,
) -> torch.Tensor:
    """RG loss với term-weighting (Eq.4–5): từ hiếm/bệnh lý được nhân trọng số cao hơn."""
    B, T, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    targets_flat = targets.reshape(-1)
    loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=pad_idx, reduction="none")
    weights = term_weight_table[targets_flat]
    mask = (targets_flat != pad_idx).float()
    return (loss * weights * mask).sum() / mask.sum().clamp(min=1)


def mlc_bce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """MLC loss (Eq.6): binary cross-entropy cho 14 nhãn bệnh CheXpert.
    labels: (B, L) ∈ {0,1}.
    """
    return F.binary_cross_entropy_with_logits(logits, labels)


def itm_triplet_loss(
    sim_matrix: torch.Tensor,   # (B, B) — similarity của mọi cặp image-text
    epsilon: float = 0.05,      # ε: triplet margin (paper Eq.8)
    mu: float = 0.025,          # μ_ITM: L2-regularization (paper Eq.9)
) -> torch.Tensor:
    """ITM loss với triplet margin + L2-regularization (Eq.8–9).

    Với mỗi anchor (image_i, text_i):
      - positive: sim(image_i, text_i)  — đường chéo
      - negative_img: max_{j≠i} sim(image_j, text_i)  — ảnh sai ghép với text đúng
      - negative_txt: max_{j≠i} sim(image_i, text_j)  — text sai ghép với ảnh đúng

    L_ITM = mean[max(0, ε - sim_pos + sim_neg_img) + max(0, ε - sim_pos + sim_neg_txt)]
          + μ × ||sim_matrix||²_F
    """
    B = sim_matrix.size(0)
    # Positive: đường chéo
    pos = sim_matrix.diagonal()  # (B,)

    # Negative: hard negative — max ngoài đường chéo
    # Tạo mask loại bỏ đường chéo
    mask = torch.eye(B, device=sim_matrix.device).bool()
    sim_off_diag = sim_matrix.masked_fill(mask, float("-inf"))

    neg_img = sim_off_diag.max(dim=0).values   # (B,) — best wrong image for each text
    neg_txt = sim_off_diag.max(dim=1).values   # (B,) — best wrong text for each image

    # Triplet loss (Eq.8)
    loss_img = F.relu(epsilon - pos + neg_img)
    loss_txt = F.relu(epsilon - pos + neg_txt)
    triplet = (loss_img + loss_txt).mean()

    # L2 regularization trên sim_matrix (Eq.9)
    l2_reg = mu * sim_matrix.pow(2).mean()

    return triplet + l2_reg


def itm_temporal_weight(epoch: int, alpha: float = 0.5, max_val: float = 5.0) -> float:
    """Temporal weighting (Eq.10–11): trọng số ITM tăng từ 0 lên max_val theo sigmoid.
    w(t) = max_val × (2σ(α·t) − 1)  với σ là sigmoid.
    α=0.5, max_val=λ_ITM=5.0 theo paper.
    """
    return max_val * (2.0 / (1.0 + math.exp(-alpha * epoch)) - 1.0)
