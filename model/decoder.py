"""
decoder.py
----------
Memory-augmented Transformer Decoder (theo Fig.1 bên phải).
- Word embeddings + positional embedding
- Masked self-attention (causal mask)
- Cross-attention tới regional features của ViT encoder
- Mọi LayerNorm thay bằng MemoryDrivenLayerNorm (pre-norm style theo sơ đồ)
- Nhiều decoder layer; paper gợi ý trung bình hóa output các lớp (decoders average output)
"""
import torch
import torch.nn as nn
from .memory_layernorm import MemoryDrivenLayerNorm


class MemoryAugmentedDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, dropout, memory_dim):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model), nn.Dropout(dropout),
        )
        self.norm1 = MemoryDrivenLayerNorm(d_model, memory_dim)
        self.norm2 = MemoryDrivenLayerNorm(d_model, memory_dim)
        self.norm3 = MemoryDrivenLayerNorm(d_model, memory_dim)

    def forward(self, x, memory, causal_mask):
        # Masked self-attention
        a, _ = self.self_attn(x, x, x, attn_mask=causal_mask, need_weights=False)
        x = self.norm1(x + a)
        # Cross-attention tới image regions
        a, _ = self.cross_attn(x, memory, memory, need_weights=False)
        x = self.norm2(x + a)
        # Feed forward
        x = self.norm3(x + self.ffn(x))
        return x


class MemoryAugmentedDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_idx)
        self.pos_embed = nn.Parameter(torch.randn(1, cfg.max_seq_len, cfg.d_model) * 0.02)
        self.layers = nn.ModuleList([
            MemoryAugmentedDecoderLayer(
                cfg.d_model, cfg.nhead, cfg.dim_feedforward, cfg.dropout, cfg.memory_dim
            )
            for _ in range(cfg.num_decoder_layers)
        ])
        self.final_norm = MemoryDrivenLayerNorm(cfg.d_model, cfg.memory_dim)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size)

    @staticmethod
    def _causal_mask(T: int, device):
        # True = vị trí bị che (không được nhìn thấy)
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, tgt_ids: torch.Tensor, memory: torch.Tensor):
        """
        tgt_ids: (B, T) — báo cáo đã shift-right (input cho decoder)
        memory:  (B, V*N, d_model) — regional features từ encoder
        Trả về:
            logits:      (B, T, vocab_size)
            txt_pooled:  (B, d_model) — biểu diễn văn bản cho ITM head
        """
        B, T = tgt_ids.shape
        x = self.token_embed(tgt_ids) + self.pos_embed[:, :T, :]
        mask = self._causal_mask(T, x.device)

        layer_outs = []
        for layer in self.layers:
            x = layer(x, memory, mask)
            layer_outs.append(x)
        # "decoders average output": trung bình hóa hidden state các lớp (Fig.1)
        x_avg = torch.stack(layer_outs, dim=0).mean(dim=0)
        x = self.final_norm(x_avg)

        logits = self.lm_head(x)                                    # (B, T, V)
        txt_pooled = x.mean(dim=1)                                  # (B, d) — dùng cho ITM
        return logits, txt_pooled
