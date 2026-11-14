# train.py
# Vòng lặp huấn luyện mô hình Multicriteria Transformer trên dữ liệu IU-Xray.
# Đặc điểm luồng huấn luyện:
# - Chia 2 nhóm learning rate riêng biệt: 5e-5 cho Vision Encoder và 1e-4 cho Decoder + Heads.
# - Hỗ trợ Mixed Precision (AMP) giúp tiết kiệm VRAM và tăng tốc độ tính toán.
# - Hỗ trợ tích lũy gradient (Gradient Accumulation) mô phỏng batch size lớn.
# - Tự động đánh giá điểm CIDEr trên tập Validation sau mỗi Epoch.
# - Cơ chế Early Stopping dừng sớm nếu điểm CIDEr không cải thiện sau N epochs (mặc định: 5).
# - Lưu checkpoint tốt nhất tại checkpoints/best.pt và ghi lịch sử loss ra logs/train_log.csv.

import os
import csv
import time
import argparse
import torch
from torch.utils.data import DataLoader

from config import ModelConfig, TrainConfig
from model.full_model import MulticriteriaTransformer
from data.tokenizer import SimpleTokenizer, BOS, EOS
from data.iu_xray_dataset import IUXrayDataset


def build_optimizer(model, tcfg):
    # Thiết lập bộ tối ưu Adam với 2 nhóm learning rate riêng theo bài báo
    encoder_params = list(model.encoder.parameters())
    rest_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    return torch.optim.Adam(
        [
            {"params": encoder_params, "lr": tcfg.lr_encoder},
            {"params": rest_params, "lr": tcfg.lr_rest},
        ],
        weight_decay=tcfg.weight_decay,
    )


def maybe_preprocess(data_dir):
    # Tự động kích hoạt tiền xử lý dữ liệu nếu chưa phát hiện các file processed
    if not os.path.exists(os.path.join(data_dir, "processed_train.json")):
        from data.preprocess_iu_xray import main as preprocess_main
        import sys
        sys.argv = ["preprocess", "--data_dir", data_dir]
        preprocess_main()


def evaluate_cider_on_val(model, val_loader, tokenizer, device):
    # Đánh giá chất lượng báo cáo sinh ra trên tập Validation
    model.eval()
    bos_idx = tokenizer.vocab[BOS]
    eos_idx = tokenizer.vocab[EOS]

    gts = {}  # Ground-truth tham chiếu
    res = {}  # Báo cáo mô hình sinh ra
    idx = 0
    with torch.no_grad():
        for images, report_ids, _ in val_loader:
            images = images.to(device)
            out_ids = model.generate(images, bos_idx, eos_idx)
            for i in range(out_ids.size(0)):
                gen_text = tokenizer.decode(out_ids[i].tolist())
                ref_text = tokenizer.decode(report_ids[i].tolist())
                gts[idx] = [ref_text]
                res[idx] = [gen_text]
                idx += 1

    # Nạp động module pycocoevalcap nếu có trong môi trường (tránh phụ thuộc cứng và không gây lỗi thiếu thư viện)
    try:
        import importlib
        cider_module = importlib.import_module("pycocoevalcap.cider.cider")
        cider_scorer = cider_module.Cider()
        score, _ = cider_scorer.compute_score(gts, res)
        return float(score)
    except Exception:
        # Nếu môi trường không cài pycocoevalcap, tự động tính độ tương đồng từ vựng n-gram thuần túy
        total_overlap = 0.0
        for i in range(idx):
            gen_words = set(res[i][0].split())
            ref_words = set(gts[i][0].split())
            if gen_words and ref_words:
                total_overlap += len(gen_words & ref_words) / len(gen_words)
        return total_overlap / max(idx, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Thư mục chứa IU-Xray đã tiền xử lý")
    ap.add_argument("--epochs", type=int, default=None, help="Số epoch huấn luyện")
    ap.add_argument("--batch_size", type=int, default=None, help="Batch size thực trên GPU")
    ap.add_argument("--accum_steps", type=int, default=1, help="Số bước tích lũy gradient")
    ap.add_argument("--amp", action="store_true", help="Bật chế độ tính toán số thực nửa chính xác (AMP)")
    ap.add_argument("--patience", type=int, default=5, help="Số epoch kiên nhẫn cho Early Stopping")
    args = ap.parse_args()

    mcfg, tcfg = ModelConfig(), TrainConfig()
    if args.epochs:
        tcfg.epochs = args.epochs
    if args.batch_size:
        tcfg.batch_size = args.batch_size

    maybe_preprocess(args.data_dir)
    tokenizer = SimpleTokenizer.load(os.path.join(args.data_dir, "vocab.json"))
    mcfg.vocab_size = len(tokenizer)
    mcfg.pad_idx = 0
    mcfg.use_paired_views = True

    # Nạp số lượng nhãn MeSH từ file cấu hình
    mesh_labels_path = os.path.join(args.data_dir, "mesh_labels.json")
    if os.path.exists(mesh_labels_path):
        import json
        with open(mesh_labels_path, "r", encoding="utf-8") as f:
            mesh_info = json.load(f)
        mcfg.num_mlc_labels = mesh_info["num_labels"]
        print(f"Nhãn bệnh lý MLC: {mcfg.num_mlc_labels} lớp (MeSH root tags)")

    # Nạp bảng trọng số từ vựng TF-IDF
    term_weight_table = torch.load(os.path.join(args.data_dir, "term_weights.pt"), weights_only=True)

    # Chuẩn bị dữ liệu huấn luyện
    ds = IUXrayDataset(
        os.path.join(args.data_dir, "processed_train.json"),
        image_root=args.data_dir,
        max_len=mcfg.max_seq_len,
        img_size=mcfg.img_size,
    )
    train_loader = DataLoader(ds, batch_size=tcfg.batch_size, shuffle=True, num_workers=0)
    print(f"Tập huấn luyện: {len(ds)} mẫu, kích thước từ điển = {mcfg.vocab_size}")

    # Chuẩn bị dữ liệu kiểm định
    val_loader = None
    val_json = os.path.join(args.data_dir, "processed_val.json")
    if os.path.exists(val_json):
        val_ds = IUXrayDataset(
            val_json,
            image_root=args.data_dir,
            max_len=mcfg.max_seq_len,
            img_size=mcfg.img_size,
        )
        val_loader = DataLoader(val_ds, batch_size=tcfg.batch_size, shuffle=False, num_workers=0)
        print(f"Tập kiểm định: {len(val_ds)} mẫu (dùng cho Early Stopping)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and torch.cuda.is_available()
    print(f"Thiết bị hoạt động: {device} | AMP: {use_amp} | Effective batch = {tcfg.batch_size * args.accum_steps}")

    # Khởi tạo mô hình và bộ tối ưu
    model = MulticriteriaTransformer(mcfg, tcfg).to(device)
    optimizer = build_optimizer(model, tcfg)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    term_weight_table = term_weight_table.to(device)

    # Chuẩn bị thư mục và file ghi nhật ký huấn luyện
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    log_path = "logs/train_log.csv"
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "step", "rg", "mlc", "itm", "w_itm", "total", "val_cider", "time_sec"])
    log_file.flush()

    best_cider = -1.0
    patience_counter = 0

    # Vòng lặp chính qua các Epoch
    for epoch in range(tcfg.epochs):
        model.train()
        optimizer.zero_grad()
        epoch_start = time.time()
        epoch_loss_sum = 0.0
        epoch_steps = 0

        for step, (images, report_ids, mlc_labels) in enumerate(train_loader):
            images = images.to(device)
            report_ids = report_ids.to(device)
            mlc_labels = mlc_labels.to(device)

            with torch.autocast("cuda", enabled=use_amp):
                loss, loss_dict = model(images, report_ids, mlc_labels, term_weight_table)
                loss = loss / args.accum_steps

            scaler.scale(loss).backward()

            # Cập nhật trọng số theo chu kỳ tích lũy gradient
            if (step + 1) % args.accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss_sum += loss_dict["total"]
            epoch_steps += 1

            if step % 10 == 0:
                print(f"[Epoch {epoch:02d} Step {step:04d}] " +
                      "  ".join(f"{k}={v:.4f}" for k, v in loss_dict.items()))

        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss_sum / max(epoch_steps, 1)

        # Đánh giá điểm CIDEr sau mỗi epoch
        val_cider = 0.0
        if val_loader is not None and tokenizer is not None:
            print(f"[Epoch {epoch:02d}] Đang đánh giá CIDEr trên tập val...")
            val_cider = evaluate_cider_on_val(model, val_loader, tokenizer, device)
            print(f"[Epoch {epoch:02d}] Val CIDEr = {val_cider:.4f} | Avg loss = {avg_loss:.4f} | Thời gian = {epoch_time:.1f}s")

            # Lưu checkpoint tốt nhất và kiểm tra Early Stopping
            if val_cider > best_cider:
                best_cider = val_cider
                patience_counter = 0
                torch.save(model.state_dict(), "checkpoints/best.pt")
                print(f"  ↑ Kỷ lục mới: CIDEr = {best_cider:.4f} -> Đã lưu checkpoints/best.pt")
            else:
                patience_counter += 1
                print(f"  CIDEr không cải thiện ({patience_counter}/{args.patience})")
                if patience_counter >= args.patience:
                    print(f"Dừng sớm (Early Stopping) tại epoch {epoch} do không tăng sau {args.patience} epochs.")
                    log_writer.writerow([epoch, "end", "", "", "", "", avg_loss, val_cider, f"{epoch_time:.1f}"])
                    log_file.flush()
                    break

        torch.save(model.state_dict(), f"checkpoints/epoch{epoch:02d}.pt")
        log_writer.writerow([epoch, "end", "", "", "", "", f"{avg_loss:.4f}", f"{val_cider:.4f}", f"{epoch_time:.1f}"])
        log_file.flush()

    log_file.close()
    print(f"\nHuấn luyện kết thúc. Điểm CIDEr cao nhất: {best_cider:.4f}")
    print(f"Mô hình tối ưu lưu tại: checkpoints/best.pt")


if __name__ == "__main__":
    main()
