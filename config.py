# config.py
# Tập hợp toàn bộ siêu tham số cấu hình của mô hình và quá trình huấn luyện
# Tuân thủ chặt chẽ theo Bảng I và Công thức 12 trong bài báo IEEE TMI 2022.

from dataclasses import dataclass


@dataclass
class ModelConfig:
    # --- Cấu hình ảnh đầu vào (IEEE TMI 2022: 448x448, patch 32x32) ---
    img_size: int = 448            # Kích thước ảnh chuẩn hóa: 448 x 448
    patch_size: int = 32           # Kích thước mỗi patch vuông: 32 x 32 -> (448/32)^2 = 196 patches/view
    in_channels: int = 1           # Ảnh X-quang Grayscale 1 kênh

    # --- Kích thước mô hình Transformer (d_model = 512, 8 heads, 3 tầng) ---
    d_model: int = 512             # Chiều vector ẩn chung cho cả Encoder và Decoder
    nhead: int = 8                 # Số lượng cơ chế tập trung (Attention Heads)
    num_encoder_layers: int = 3    # 3 lớp Vision Transformer Encoder
    num_decoder_layers: int = 3    # 3 lớp Memory-Augmented Text Decoder
    dim_feedforward: int = 2048    # Chiều ẩn của mạng FFN
    dropout: float = 0.1           # Tỉ lệ Dropout chống quá khớp (Overfitting)

    # --- Cấu hình văn bản báo cáo ---
    max_seq_len: int = 60          # Độ dài chuỗi từ tối đa của báo cáo y khoa (60 từ)
    vocab_size: int = 1000         # Kích thước từ vựng (tự động cập nhật theo vocab.json)
    pad_idx: int = 0               # Mã ID của token đệm <pad>

    # --- Phân loại bệnh lý đa nhãn (Multi-Label Classification) ---
    # IU-Xray sử dụng 65 nhãn MeSH root tags trích xuất thực tế từ dữ liệu
    num_mlc_labels: int = 65

    # --- Chuẩn hóa lớp điều khiển bởi bộ nhớ (Memory-Driven LayerNorm) ---
    memory_dim: int = 64           # Chiều của vector bộ nhớ điều khiển tham số chuẩn hóa

    # --- Ghép nối ảnh - văn bản (Image-Text Matching) ---
    itm_dim: int = 512             # Chiều không gian biểu diễn chung giữa ảnh và từ vựng
    use_paired_views: bool = True   # IU-Xray luôn sử dụng cặp 2 góc chụp (Frontal + Lateral)

    # Trọng số thời gian tăng dần về cuối báo cáo theo Công thức 10-11: w(t) = 2*sigmoid(alpha*t) - 1
    temporal_alpha: float = 0.5


@dataclass
class TrainConfig:
    batch_size: int = 16            # Kích thước mini-batch = 16
    epochs: int = 30                # Huấn luyện trong 30 epochs
    lr_encoder: float = 5e-5        # Tốc độ học riêng cho Vision Transformer Encoder
    lr_rest: float = 1e-4           # Tốc độ học cho Text Decoder và các đầu phụ trợ
    weight_decay: float = 1e-4      # Hệ số suy giảm trọng số L2

    # Trọng số các thành phần hàm mất mát theo Công thức 12 trong bài báo:
    # L_total = lambda_rg * L_RG + lambda_mlc * L_MLC + lambda_itm * L_ITM
    lambda_rg: float = 1.0          # Trọng số mất mát sinh báo cáo
    lambda_mlc: float = 2.0         # Trọng số mất mát phân loại bệnh lý
    lambda_itm_max: float = 5.0     # Trọng số mất mát ghép cặp ảnh-văn bản

    # Siêu tham số cho hàm mất mát ITM Triplet Loss (Công thức 8 và 9)
    itm_epsilon: float = 0.05       # Biên cách biệt Triplet Margin
    itm_mu: float = 0.025           # Hệ số chuẩn hóa Frobenius norm
