# decoder.py
# Memory-Augmented Transformer Decoder (Mô-đun giải mã văn bản tăng cường bộ nhớ)
# Theo cấu trúc đề xuất trong IEEE TMI 2022 (Hình 1):
# 1. Nhúng từ vựng (Word Embedding) + Nhúng vị trí thứ tự câu (Positional Embedding).
# 2. Mặt nạ nhân quả (Causal Mask) đảm bảo giải mã tự hồi quy (từ sau không nhìn thấy từ trước).
# 3. Cơ chế Cross-Attention giúp liên kết trực tiếp từng từ vựng với 392 vùng đặc trưng ảnh X-quang.
# 4. Thay thế toàn bộ LayerNorm truyền thống bằng Memory-Driven LayerNorm (MLN).
# 5. Lấy trung bình cộng hidden states của tất cả các lớp Decoder (Decoders Average Output).

import torch
import torch.nn as nn
from .memory_layernorm import MemoryDrivenLayerNorm


class MemoryAugmentedDecoderLayer(nn.Module):
    # Một tầng Decoder: Causal Self-Attention -> Cross-Attention -> Feed-Forward
    # Mỗi bước chuẩn hóa đều được điều biến bởi Memory-Driven LayerNorm
    def __init__(self, d_model, nhead, dim_ff, dropout, memory_dim):
        super().__init__()
        # Self-attention có che mặt nạ nhân quả
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Cross-attention tương tác với đặc trưng vùng thị giác của ảnh
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Mạng lan truyền xuôi FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        # Các khối chuẩn hóa tích hợp bộ nhớ học được
        self.norm1 = MemoryDrivenLayerNorm(d_model, memory_dim)
        self.norm2 = MemoryDrivenLayerNorm(d_model, memory_dim)
        self.norm3 = MemoryDrivenLayerNorm(d_model, memory_dim)

    def forward(self, x, memory, causal_mask):
        # 1. Masked Self-Attention trên chuỗi từ đã sinh
        a, _ = self.self_attn(x, x, x, attn_mask=causal_mask, need_weights=False)
        x = self.norm1(x + a)
        # 2. Cross-Attention truy vấn đặc trưng thị giác từ ViT Encoder
        a, _ = self.cross_attn(x, memory, memory, need_weights=False)
        x = self.norm2(x + a)
        # 3. Chiếu qua FFN và chuẩn hóa MLN
        x = self.norm3(x + self.ffn(x))
        return x


class MemoryAugmentedDecoder(nn.Module):
    # Bộ giải mã sinh báo cáo hoàn chỉnh tích hợp vector bộ nhớ
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Lớp nhúng từ vựng
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_idx)
        # Vector nhúng vị trí theo độ dài tối đa báo cáo (60 từ)
        self.pos_embed = nn.Parameter(torch.randn(1, cfg.max_seq_len, cfg.d_model) * 0.02)
        # 3 lớp Memory-Augmented Decoder xếp chồng
        self.layers = nn.ModuleList([
            MemoryAugmentedDecoderLayer(
                cfg.d_model, cfg.nhead, cfg.dim_feedforward, cfg.dropout, cfg.memory_dim
            )
            for _ in range(cfg.num_decoder_layers)
        ])
        self.final_norm = MemoryDrivenLayerNorm(cfg.d_model, cfg.memory_dim)
        # Đầu chiếu tuyến tính ra không gian từ điển để tính xác suất từ tiếp theo
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size)

    @staticmethod
    def _causal_mask(T: int, device):
        # Tạo ma trận tam giác trên với giá trị True tại các vị trí cần che (tương lai)
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, tgt_ids: torch.Tensor, memory: torch.Tensor):
        # Đầu vào:
        # - tgt_ids: (B, T) chuỗi mã token đầu vào (đã dịch phải 1 bước)
        # - memory:  (B, V*N, d_model) đặc trưng vùng ảnh từ ViT Encoder
        # Đầu ra:
        # - logits:       (B, T, vocab_size) phân phối xác suất từ cho hàm loss RG
        # - txt_features: (B, T, d_model) đặc trưng từng token phục vụ ghép cặp ảnh-văn bản ITM
        B, T = tgt_ids.shape
        x = self.token_embed(tgt_ids) + self.pos_embed[:, :T, :]
        mask = self._causal_mask(T, x.device)

        layer_outs = []
        for layer in self.layers:
            x = layer(x, memory, mask)
            layer_outs.append(x)
            
        # Cơ chế Decoders Average Output: lấy trung bình biểu diễn của cả 3 tầng decoder
        x_avg = torch.stack(layer_outs, dim=0).mean(dim=0)
        x = self.final_norm(x_avg)

        # Tính logits phân phối từ trên toàn bộ từ điển
        logits = self.lm_head(x)                                    # (B, T, vocab_size)
        txt_features = x                                            # (B, T, d_model)
        return logits, txt_features
