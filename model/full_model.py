"""
full_model.py
-------------
Mô hình hoàn chỉnh: Multicriteria Supervised Transformer.
Gộp ViT encoder + memory-augmented decoder + 3 head loss (RG, MLC, ITM).
Forward trả về tổng loss và dict các thành phần loss để theo dõi.
"""
import torch
import torch.nn as nn

from .vit_encoder import ViTEncoder
from .decoder import MemoryAugmentedDecoder
from .heads import MLCHead, ITMHead
from .losses import (
    term_weighted_cross_entropy,
    mlc_bce_loss,
    itm_triplet_loss,
)


class MulticriteriaTransformer(nn.Module):
    def __init__(self, mcfg, tcfg):
        super().__init__()
        self.mcfg = mcfg
        self.tcfg = tcfg
        self.encoder = ViTEncoder(mcfg)
        self.decoder = MemoryAugmentedDecoder(mcfg)
        self.mlc_head = MLCHead(mcfg.d_model, mcfg.num_mlc_labels)
        # ITM head: per-token similarity + temporal weighting (Eq.10-11)
        self.itm_head = ITMHead(mcfg.d_model, mcfg.itm_dim, mcfg.temporal_alpha)

    def forward(
        self,
        images: torch.Tensor,             # (B,C,H,W) hoặc (B,V,C,H,W)
        report_ids: torch.Tensor,         # (B, T) — báo cáo gốc, chưa shift
        mlc_labels: torch.Tensor = None,  # (B, L) — nhãn bệnh (None khi inference)
        term_weight_table: torch.Tensor = None,  # (V,) — None khi inference
    ):
        # --- Encoder ---
        regional_features, cls_global = self.encoder(images)

        # --- Decoder (shift-right: input = report[:, :-1], target = report[:, 1:]) ---
        dec_input = report_ids[:, :-1]
        logits, txt_features = self.decoder(dec_input, regional_features)
        # logits: (B, T-1, V) — cho RG loss
        # txt_features: (B, T-1, d_model) — per-token fused features cho ITM

        total_loss = torch.tensor(0.0, device=images.device)
        loss_dict = {}

        # 1) RG loss + term-weighting (chỉ tính khi có bảng trọng số)
        if term_weight_table is not None:
            l_rg = term_weighted_cross_entropy(
                logits, report_ids[:, 1:], term_weight_table, self.mcfg.pad_idx
            )
            total_loss = total_loss + self.tcfg.lambda_rg * l_rg
            loss_dict["rg"] = l_rg.item()

        # 2) MLC loss
        if mlc_labels is not None:
            mlc_logits = self.mlc_head(cls_global)
            l_mlc = mlc_bce_loss(mlc_logits, mlc_labels)
            total_loss = total_loss + self.tcfg.lambda_mlc * l_mlc
            loss_dict["mlc"] = l_mlc.item()

        # 3) ITM loss (Eq.8–11)
        #    - ITM head tính per-token similarity với temporal weighting (Eq.10-11)
        #    - Padding mask để loại trừ padding tokens khỏi phép tính
        #    - Triplet loss + L2-reg (Eq.8-9)
        #    - λ_ITM = 5.0 cố định (không dùng epoch-based scheduler)
        padding_mask = (dec_input != self.mcfg.pad_idx)             # (B, T-1), True = valid
        sim = self.itm_head(cls_global, txt_features, padding_mask)
        l_itm = itm_triplet_loss(sim, epsilon=self.tcfg.itm_epsilon, mu=self.tcfg.itm_mu)
        total_loss = total_loss + self.tcfg.lambda_itm_max * l_itm
        loss_dict["itm"] = l_itm.item()
        loss_dict["w_itm"] = self.tcfg.lambda_itm_max

        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict

    @torch.no_grad()
    def generate(self, images: torch.Tensor, bos_idx: int, eos_idx: int):
        """Sinh báo cáo bằng greedy decoding (skeleton cho inference/demo)."""
        device = images.device
        regional_features, cls_global = self.encoder(images)
        B = images.size(0) if images.dim() == 4 else images.size(0)
        ys = torch.full((B, 1), bos_idx, dtype=torch.long, device=device)
        for _ in range(self.mcfg.max_seq_len - 1):
            logits, _ = self.decoder(ys, regional_features)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B,1)
            ys = torch.cat([ys, next_token], dim=1)
            if (next_token == eos_idx).all():
                break
        return ys
