"""
config.py
---------
Toàn bộ siêu tham số của model và quá trình train tập trung ở đây.
Tuân thủ đúng paper IEEE TMI 2022 (Table I & Eq.12).
"""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    # --- Ảnh đầu vào (paper: 448×448, patch 32) ---
    img_size: int = 448            # paper: resize về 448×448
    patch_size: int = 32           # paper: patch size = 32 → (448/32)²=196 patches
    in_channels: int = 1           # ảnh X-quang grayscale = 1

    # --- Kích thước model (paper: d=512, 8 heads, 3 lớp) ---
    d_model: int = 512             # chiều ẩn chung của encoder & decoder
    nhead: int = 8                 # paper: 8 attention heads
    num_encoder_layers: int = 3    # paper: 3 lớp Vision Encoder
    num_decoder_layers: int = 3    # paper: 3 lớp Text Decoder
    dim_feedforward: int = 2048
    dropout: float = 0.1

    # --- Báo cáo văn bản ---
    max_seq_len: int = 60          # paper: độ dài báo cáo tối đa = 60 từ
    vocab_size: int = 1000         # sẽ cập nhật từ vocab.json sau preprocess
    pad_idx: int = 0               # idx của token <pad>

    # --- Nhãn bệnh (Multi-Label Classification) ---
    num_mlc_labels: int = 14       # 14 bệnh theo bộ nhãn CheXpert

    # --- Memory-driven LayerNorm (đặc trưng của paper) ---
    memory_dim: int = 64           # chiều của vector memory điều khiển LN

    # --- Image-Text Matching (ITM) ---
    itm_dim: int = 512
    itm_temperature: float = 0.07

    # --- IU-Xray: luôn dùng 2 view (frontal + lateral) ---
    use_paired_views: bool = True   # paper: IU-Xray dùng cả 2 góc chụp


@dataclass
class TrainConfig:
    batch_size: int = 16            # paper: mini-batch = 16
    epochs: int = 30                # paper: train 30 epochs
    lr_encoder: float = 5e-5        # paper: lr riêng cho vision encoder
    lr_rest: float = 1e-4           # paper: lr cho các tham số còn lại
    weight_decay: float = 1e-4

    # Trọng số các thành phần loss (Equation 12 trong paper)
    lambda_rg: float = 1.0          # λ_RG = 1.0
    lambda_mlc: float = 2.0         # λ_MLC = 2.0 (paper Table I)
    lambda_itm_max: float = 5.0     # λ_ITM = 5.0 (paper Table I)

    # Temporal weighting cho ITM: w(t) = λ_ITM × sigmoid(α × t)
    itm_ramp_k: float = 0.5         # α = 0.5 (paper: hệ số temporal-weighting)

    # ITM loss hyperparameters (paper Eq.8 & Eq.9)
    itm_epsilon: float = 0.05       # ε = 0.05: biên độ triplet margin
    itm_mu: float = 0.025           # μ_ITM = 0.025: L2-regularization cho ITM
