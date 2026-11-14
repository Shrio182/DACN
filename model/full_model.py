# full_model.py
# Mô hình hoàn chỉnh: Multicriteria Supervised Transformer (IEEE TMI 2022).
# Tích hợp toàn diện:
# - Vision Transformer Encoder (from scratch) xử lý cặp ảnh Frontal + Lateral.
# - Memory-Augmented Text Decoder sinh báo cáo tự hồi quy.
# - Ba nhánh giám sát: Sinh từ (RG), Phân loại bệnh lý (MLC), So khớp ảnh-văn bản (ITM).
# - Cơ chế giải mã Visual-Grounded Contrastive Guidance phá vỡ Mode Collapse (Language Prior Bias).

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
        self.itm_head = ITMHead(mcfg.d_model, mcfg.itm_dim, mcfg.temporal_alpha)

    def forward(
        self,
        images: torch.Tensor,             # (B, C, H, W) hoặc (B, V, C, H, W)
        report_ids: torch.Tensor,         # (B, T) - chuỗi token báo cáo đầy đủ
        mlc_labels: torch.Tensor = None,  # (B, num_labels) - nhãn bệnh lý
        term_weight_table: torch.Tensor = None,  # (vocab_size,) - trọng số TF-IDF
    ):
        # 1. Trích xuất đặc trưng thị giác qua Vision Transformer
        regional_features, cls_global = self.encoder(images)

        # 2. Chuẩn bị dữ liệu cho Decoder theo nguyên tắc Teacher Forcing:
        dec_input = report_ids[:, :-1]
        logits, txt_features = self.decoder(dec_input, regional_features)

        total_loss = torch.tensor(0.0, device=images.device)
        loss_dict = {}

        # Tiêu chí 1: Term-Weighted Report Generation Loss (L_RG)
        if term_weight_table is not None:
            l_rg = term_weighted_cross_entropy(
                logits, report_ids[:, 1:], term_weight_table, self.mcfg.pad_idx
            )
            total_loss = total_loss + self.tcfg.lambda_rg * l_rg
            loss_dict["rg"] = l_rg.item()

        # Tiêu chí 2: Multi-Label Classification Loss (L_MLC) từ vector CLS toàn cục
        if mlc_labels is not None:
            mlc_logits = self.mlc_head(cls_global)
            l_mlc = mlc_bce_loss(mlc_logits, mlc_labels)
            total_loss = total_loss + self.tcfg.lambda_mlc * l_mlc
            loss_dict["mlc"] = l_mlc.item()

        # Tiêu chí 3: Temporal-Weighted Image-Text Matching Loss (L_ITM)
        if term_weight_table is not None:
            padding_mask = (dec_input != self.mcfg.pad_idx)
            sim_matrix = self.itm_head(cls_global, txt_features, padding_mask)
            l_itm = itm_triplet_loss(sim_matrix, self.tcfg.itm_epsilon, self.tcfg.itm_mu)
            total_loss = total_loss + self.tcfg.lambda_itm_max * l_itm
            loss_dict["itm"] = l_itm.item()

        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict

    @torch.no_grad()
    def predict_mlc(self, images: torch.Tensor) -> torch.Tensor:
        # Dự đoán phân phối xác suất (0 đến 1) các nhãn bệnh lý từ ảnh X-quang
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
        repetition_penalty: float = 1.2,
        visual_guidance: float = 0.8,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        # Thuật toán sinh chuỗi báo cáo y khoa tự hồi quy (Autoregressive Generation)
        # Tích hợp Visual-Grounded Contrastive Guidance:
        # score = logits_vis + visual_guidance * (logits_vis - logits_null)
        # Giúp triệt tiêu thiên lệch mẫu câu bình thường mặc định và ép mô hình
        # phản ánh chính xác các biến đổi thị giác thực tế trên ảnh X-quang.
        device = images.device
        regional_features, _ = self.encoder(images)
        max_len = max_len or self.mcfg.max_seq_len
        B = images.size(0)

        # Tính đặc trưng ảnh rỗng (Null Image Features) phục vụ hướng dẫn đối sánh
        if visual_guidance > 0.0:
            null_images = torch.zeros_like(images)
            null_features, _ = self.encoder(null_images)
        else:
            null_features = None

        def get_adjusted_logits(ys_seq, reg_feat, null_feat):
            logits_vis, _ = self.decoder(ys_seq, reg_feat)
            next_l = logits_vis[:, -1, :].clone()

            if visual_guidance > 0.0 and null_feat is not None:
                logits_null, _ = self.decoder(ys_seq, null_feat)
                next_l_null = logits_null[:, -1, :].clone()
                # Khuếch đại sự khác biệt có nguồn gốc từ tín hiệu thị giác
                diff = next_l - next_l_null
                next_l = next_l + visual_guidance * diff

            if temperature != 1.0 and temperature > 0:
                next_l = next_l / temperature

            return next_l

        # Chế độ 1: Greedy Decoding với Visual Guidance
        if beam_size <= 1:
            ys = torch.full((B, 1), bos_idx, dtype=torch.long, device=device)
            finished = torch.zeros(B, dtype=torch.bool, device=device)

            for _ in range(max_len - 1):
                next_logits = get_adjusted_logits(ys, regional_features, null_features)

                # Áp dụng phạt lặp từ
                if repetition_penalty != 1.0:
                    for b in range(B):
                        unique_tokens = torch.unique(ys[b])
                        for token_id in unique_tokens:
                            if token_id in (bos_idx, eos_idx, self.mcfg.pad_idx):
                                continue
                            val = next_logits[b, token_id]
                            if val > 0:
                                next_logits[b, token_id] = val / repetition_penalty
                            else:
                                next_logits[b, token_id] = val * repetition_penalty

                next_token = next_logits.argmax(dim=-1, keepdim=True)
                next_token = torch.where(finished.unsqueeze(1), torch.full_like(next_token, eos_idx), next_token)
                ys = torch.cat([ys, next_token], dim=1)
                finished = finished | (next_token.squeeze(1) == eos_idx)
                if finished.all():
                    break
            return ys

        # Chế độ 2: Beam Search với Visual Guidance
        else:
            all_best_seqs = []
            for b in range(B):
                reg_b = regional_features[b:b+1]
                null_b = null_features[b:b+1] if null_features is not None else None
                beams = [(torch.tensor([[bos_idx]], dtype=torch.long, device=device), 0.0, False)]

                for _ in range(max_len - 1):
                    if all(done for _, _, done in beams):
                        break

                    candidates = []
                    for seq, score, done in beams:
                        if done:
                            candidates.append((seq, score, True))
                            continue

                        next_logits = get_adjusted_logits(seq, reg_b, null_b)
                        l_step = next_logits[0].clone()

                        if repetition_penalty != 1.0:
                            unique_tokens = torch.unique(seq[0])
                            for token_id in unique_tokens:
                                if token_id in (bos_idx, eos_idx, self.mcfg.pad_idx):
                                    continue
                                val = l_step[token_id]
                                if val > 0:
                                    l_step[token_id] = val / repetition_penalty
                                else:
                                    l_step[token_id] = val * repetition_penalty

                        log_probs = torch.log_softmax(l_step, dim=-1)
                        topk_log_probs, topk_indices = torch.topk(log_probs, beam_size)

                        for k in range(beam_size):
                            token = topk_indices[k].unsqueeze(0).unsqueeze(0)
                            new_seq = torch.cat([seq, token], dim=1)
                            new_score = score + topk_log_probs[k].item()
                            is_done = (token.item() == eos_idx)
                            candidates.append((new_seq, new_score, is_done))

                    # Chọn ra k ứng viên có điểm log-likelihood chuẩn hóa theo độ dài
                    candidates.sort(key=lambda x: x[1] / max(1, x[0].size(1) ** 0.7), reverse=True)
                    beams = candidates[:beam_size]

                best_seq = beams[0][0]
                all_best_seqs.append(best_seq)

            # Đệm padding các chuỗi để ghép thành batch
            max_out_len = max(s.size(1) for s in all_best_seqs)
            padded_batch = torch.full((B, max_out_len), eos_idx, dtype=torch.long, device=device)
            for b, s in enumerate(all_best_seqs):
                padded_batch[b, :s.size(1)] = s[0]
            return padded_batch
