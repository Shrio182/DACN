"""
train.py — Vòng lặp train.
==========================
- Không có --data_dir  : chạy với DUMMY data (kiểm tra luồng).
- Có --data_dir        : train thật trên IU-Xray (cần chạy preprocess trước,
                         hoặc script tự chạy nếu chưa có processed file).

Tính năng:
  - Early stopping theo CIDEr trên tập val
  - Lưu checkpoint tốt nhất thành checkpoints/best.pt
  - In loss mỗi 10 step
  - Log CSV đơn giản (logs/train_log.csv)
  - GIỮ NGUYÊN 2 nhóm learning rate (encoder 5e-5, còn lại 1e-4) và 30 epochs

Ví dụ:
  python data/preprocess_iu_xray.py --data_dir ./iu_xray_data
  python train.py --data_dir ./iu_xray_data --epochs 30
"""
import os
import csv
import time
import argparse
import torch
from torch.utils.data import DataLoader

from config import ModelConfig, TrainConfig
from model.full_model import MulticriteriaTransformer
from data.tokenizer import SimpleTokenizer, BOS, EOS


# ---------------- DUMMY data (chưa có dataset) ----------------
class DummyDataIterator:
    def __init__(self, mcfg, tcfg, num_batches=5):
        self.mcfg, self.tcfg = mcfg, tcfg
        self.num_batches = num_batches

    def __iter__(self):
        B, T = self.tcfg.batch_size, self.mcfg.max_seq_len
        for _ in range(self.num_batches):
            if self.mcfg.use_paired_views:
                images = torch.randn(B, 2, self.mcfg.in_channels, self.mcfg.img_size, self.mcfg.img_size)
            else:
                images = torch.randn(B, self.mcfg.in_channels, self.mcfg.img_size, self.mcfg.img_size)
            report_ids = torch.randint(1, self.mcfg.vocab_size, (B, T))
            mlc_labels = torch.randint(0, 2, (B, self.mcfg.num_mlc_labels)).float()
            yield images, report_ids, mlc_labels


def build_optimizer(model, tcfg):
    encoder_params = list(model.encoder.parameters())
    rest_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    return torch.optim.Adam(
        [{"params": encoder_params, "lr": tcfg.lr_encoder},
         {"params": rest_params, "lr": tcfg.lr_rest}],
        weight_decay=tcfg.weight_decay,
    )


def maybe_preprocess(data_dir):
    """Nếu chưa có file processed thì tự chạy preprocess."""
    if not os.path.exists(os.path.join(data_dir, "processed_train.json")):
        from data.preprocess_iu_xray import main as preprocess_main
        import sys
        sys.argv = ["preprocess", "--data_dir", data_dir]
        preprocess_main()


# -------------- Đánh giá CIDEr trên tập val ------------------
def evaluate_cider_on_val(model, val_loader, tokenizer, device):
    """Sinh báo cáo trên tập val và tính CIDEr bằng pycocoevalcap."""
    model.eval()
    bos_idx = tokenizer.vocab[BOS]
    eos_idx = tokenizer.vocab[EOS]

    gts = {}  # ground truth
    res = {}  # generated
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

    # Tính CIDEr bằng pycocoevalcap
    try:
        from pycocoevalcap.cider.cider import Cider
        cider_scorer = Cider()
        score, _ = cider_scorer.compute_score(gts, res)
        return score
    except Exception as e:
        print(f"[WARNING] Không tính được CIDEr: {e}. Trả về 0.")
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None, help="Thư mục IU-Xray (annotation.json + images/)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None, help="Batch thật trên GPU (giảm nếu hết VRAM)")
    ap.add_argument("--accum_steps", type=int, default=1, help="Gradient accumulation: effective batch = batch_size * accum_steps")
    ap.add_argument("--amp", action="store_true", help="Bật mixed precision (giảm ~50%% VRAM, cần GPU NVIDIA)")
    ap.add_argument("--patience", type=int, default=5, help="Early stopping: dừng nếu CIDEr không cải thiện sau N epochs")
    args = ap.parse_args()

    mcfg, tcfg = ModelConfig(), TrainConfig()
    if args.epochs: tcfg.epochs = args.epochs
    if args.batch_size: tcfg.batch_size = args.batch_size

    term_weight_table = torch.ones(mcfg.vocab_size)
    tokenizer = None
    val_loader = None

    if args.data_dir:
        maybe_preprocess(args.data_dir)
        tokenizer = SimpleTokenizer.load(os.path.join(args.data_dir, "vocab.json"))
        mcfg.vocab_size = len(tokenizer)
        mcfg.pad_idx = 0
        mcfg.use_paired_views = True
        term_weight_table = torch.load(os.path.join(args.data_dir, "term_weights.pt"),
                                       weights_only=True)
        from data.iu_xray_dataset import IUXrayDataset
        ds = IUXrayDataset(
            os.path.join(args.data_dir, "processed_train.json"),
            image_root=args.data_dir,
            max_len=mcfg.max_seq_len, img_size=mcfg.img_size,
        )
        train_loader = DataLoader(ds, batch_size=tcfg.batch_size, shuffle=True, num_workers=0)
        print(f"Train thật: {len(ds)} mẫu, vocab={mcfg.vocab_size}")

        # Tạo val loader cho early stopping
        val_json = os.path.join(args.data_dir, "processed_val.json")
        if os.path.exists(val_json):
            val_ds = IUXrayDataset(val_json, image_root=args.data_dir,
                                   max_len=mcfg.max_seq_len, img_size=mcfg.img_size)
            val_loader = DataLoader(val_ds, batch_size=tcfg.batch_size, shuffle=False, num_workers=0)
            print(f"Val: {len(val_ds)} mẫu (dùng cho early stopping theo CIDEr)")
    else:
        train_loader = DummyDataIterator(mcfg, tcfg)
        print("Chạy với DUMMY data. Dùng --data_dir để train thật.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and torch.cuda.is_available()
    print(f"Thiết bị: {device} | AMP: {use_amp} | effective batch = {tcfg.batch_size * args.accum_steps}")

    model = MulticriteriaTransformer(mcfg, tcfg).to(device)
    optimizer = build_optimizer(model, tcfg)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    term_weight_table = term_weight_table.to(device)

    # ---- Chuẩn bị log CSV ----
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    log_path = "logs/train_log.csv"
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "step", "rg", "mlc", "itm", "w_itm", "total",
                         "val_cider", "time_sec"])
    log_file.flush()
    print(f"Log CSV: {os.path.abspath(log_path)}")

    # ---- Early stopping state ----
    best_cider = -1.0
    patience_counter = 0

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
                loss, loss_dict = model(images, report_ids, mlc_labels, term_weight_table, epoch)
                loss = loss / args.accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss_sum += loss_dict["total"]
            epoch_steps += 1

            # In loss mỗi 10 step
            if step % 10 == 0:
                print(f"[epoch {epoch:02d} step {step:04d}] " +
                      "  ".join(f"{k}={v:.4f}" for k, v in loss_dict.items()))

        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss_sum / max(epoch_steps, 1)

        # ---- Đánh giá CIDEr trên val ----
        val_cider = 0.0
        if val_loader is not None and tokenizer is not None:
            print(f"[epoch {epoch:02d}] Đang đánh giá CIDEr trên val...")
            val_cider = evaluate_cider_on_val(model, val_loader, tokenizer, device)
            print(f"[epoch {epoch:02d}] Val CIDEr = {val_cider:.4f} | "
                  f"Avg loss = {avg_loss:.4f} | Time = {epoch_time:.1f}s")

            # Early stopping + lưu best checkpoint
            if val_cider > best_cider:
                best_cider = val_cider
                patience_counter = 0
                torch.save(model.state_dict(), "checkpoints/best.pt")
                print(f"  ↑ Best CIDEr cập nhật: {best_cider:.4f} → Đã lưu checkpoints/best.pt")
            else:
                patience_counter += 1
                print(f"  CIDEr không cải thiện ({patience_counter}/{args.patience})")
                if patience_counter >= args.patience:
                    print(f"Early stopping tại epoch {epoch} (patience={args.patience}).")
                    # Ghi log epoch cuối
                    log_writer.writerow([epoch, "end", "", "", "", "", avg_loss,
                                         val_cider, f"{epoch_time:.1f}"])
                    log_file.flush()
                    break
        else:
            print(f"[epoch {epoch:02d}] Avg loss = {avg_loss:.4f} | Time = {epoch_time:.1f}s")

        # Lưu checkpoint mỗi epoch
        torch.save(model.state_dict(), f"checkpoints/epoch{epoch:02d}.pt")

        # Ghi log CSV
        log_writer.writerow([epoch, "end", "", "", "", "", f"{avg_loss:.4f}",
                             f"{val_cider:.4f}", f"{epoch_time:.1f}"])
        log_file.flush()

    log_file.close()
    print(f"\nXong. Best CIDEr = {best_cider:.4f}")
    print(f"Best checkpoint: checkpoints/best.pt")
    print(f"Log: {os.path.abspath(log_path)}")


if __name__ == "__main__":
    main()
