"""
test_forward.py — kiểm tra nhanh model chạy được với tensor giả (không cần dataset).
Chạy: python test_forward.py
"""
import torch
from config import ModelConfig, TrainConfig
from model.full_model import MulticriteriaTransformer


def main():
    mcfg, tcfg = ModelConfig(), TrainConfig()
    mcfg.vocab_size = 100
    B, T = 4, mcfg.max_seq_len

    model = MulticriteriaTransformer(mcfg, tcfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Số tham số: {n_params/1e6:.2f}M")

    # Dữ liệu giả
    images = torch.randn(B, mcfg.in_channels, mcfg.img_size, mcfg.img_size)
    report_ids = torch.randint(1, mcfg.vocab_size, (B, T))
    mlc_labels = torch.randint(0, 2, (B, mcfg.num_mlc_labels)).float()
    term_weight_table = torch.ones(mcfg.vocab_size)

    # Forward + backward
    loss, loss_dict = model(images, report_ids, mlc_labels, term_weight_table, epoch=5)
    loss.backward()
    print("Loss dict:", {k: round(v, 4) for k, v in loss_dict.items()})

    # Greedy decode (inference)
    out = model.generate(images, bos_idx=1, eos_idx=2)
    print("Output shape khi generate:", out.shape)
    print("OK — forward, backward, generate đều chạy được.")


if __name__ == "__main__":
    main()
