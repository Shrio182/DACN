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
    def predict_mlc(self, images: torch.Tensor) -> torch.Tensor:
        """Dự đoán xác suất các nhãn bệnh từ CLS token toàn cục.
        Trả về: probs tensor shape (B, num_mlc_labels) trong khoảng [0, 1].
        """
        _, cls_global = self.encoder(images)
        logits = self.mlc_head(cls_global)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        bos_idx: int,
        eos_idx: int,
        max_len: int = None,
        beam_size: int = 1,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Sinh báo cáo với hỗ trợ Greedy Decoding và Beam Search kết hợp Repetition Penalty."""
        device = images.device
        regional_features, cls_global = self.encoder(images)
        max_len = max_len or self.mcfg.max_seq_len
        B = images.size(0)

        if beam_size <= 1:
            ys = torch.full((B, 1), bos_idx, dtype=torch.long, device=device)
            finished = torch.zeros(B, dtype=torch.bool, device=device)

            for _ in range(max_len - 1):
                logits, _ = self.decoder(ys, regional_features)
                next_logits = logits[:, -1, :].clone()

                if repetition_penalty != 1.0:
                    for b in range(B):
                        unique_tokens = torch.unique(ys[b])
                        for token_id in unique_tokens:
                            if token_id in (bos_idx, eos_idx, self.mcfg.pad_idx):
                                continue
                            val = next_logits[b, token_id]
                            if val < 0:
                                next_logits[b, token_id] = val * repetition_penalty
                            else:
                                next_logits[b, token_id] = val / repetition_penalty

                next_token = next_logits.argmax(dim=-1, keepdim=True)
                next_token = torch.where(finished.unsqueeze(1), torch.full_like(next_token, eos_idx), next_token)
                ys = torch.cat([ys, next_token], dim=1)
                finished = finished | (next_token.squeeze(1) == eos_idx)
                if finished.all():
                    break
            return ys
        else:
            all_best_seqs = []
            for b in range(B):
                reg_b = regional_features[b:b+1]
                beams = [(torch.tensor([[bos_idx]], dtype=torch.long, device=device), 0.0, False)]

                for _ in range(max_len - 1):
                    if all(done for _, _, done in beams):
                        break

                    candidates = []
                    for seq, score, done in beams:
                        if done:
                            candidates.append((seq, score, True))
                            continue

                        logits, _ = self.decoder(seq, reg_b)
                        next_logits = logits[:, -1, :].squeeze(0).clone()

                        if repetition_penalty != 1.0:
                            unique_tokens = torch.unique(seq[0])
                            for token_id in unique_tokens:
                                if token_id in (bos_idx, eos_idx, self.mcfg.pad_idx):
                                    continue
                                val = next_logits[token_id]
                                if val < 0:
                                    next_logits[token_id] = val * repetition_penalty
                                else:
                                    next_logits[token_id] = val / repetition_penalty

                        log_probs = torch.log_softmax(next_logits, dim=-1)
                        topk_log_probs, topk_tokens = torch.topk(log_probs, beam_size)

                        for k in range(beam_size):
                            tok = topk_tokens[k].item()
                            lp = topk_log_probs[k].item()
                            new_seq = torch.cat([seq, torch.tensor([[tok]], dtype=torch.long, device=device)], dim=1)
                            new_score = score + lp
                            new_done = (tok == eos_idx)
                            candidates.append((new_seq, new_score, new_done))

                    # Chiều dài phạt (length penalty)
                    candidates.sort(key=lambda x: x[1] / (len(x[0][0]) ** 0.7), reverse=True)
                    beams = candidates[:beam_size]

                all_best_seqs.append(beams[0][0])

            max_batch_len = max(s.size(1) for s in all_best_seqs)
            padded_batch = []
            for s in all_best_seqs:
                if s.size(1) < max_batch_len:
                    pad = torch.full((1, max_batch_len - s.size(1)), eos_idx, dtype=torch.long, device=device)
                    s = torch.cat([s, pad], dim=1)
                padded_batch.append(s)
            return torch.cat(padded_batch, dim=0)
