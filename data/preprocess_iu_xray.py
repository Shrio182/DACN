"""
preprocess_iu_xray.py — Tiền xử lý IU-Xray từ đầu.
==================================================
Input:  data_dir/annotation.json  (định dạng chuẩn dùng trong R2Gen/KERP:
        list các {id, image_path: [...], report: "..."} + thư mục images/).
        Nếu bạn chỉ có file CSV gốc của Open-i, xem hàm build_annotation_from_csv bên dưới.

Output: processed_train/val/test.json, vocab.json, term_weights.pt
        -> train.py đọc trực tiếp các file này.

Chạy:  python data/preprocess_iu_xray.py --data_dir ./iu_xray_data
"""
import os
import json
import argparse
import random
from collections import Counter
import torch

try:
    from .tokenizer import SimpleTokenizer, SPECIAL
    from .chexpert_labeler import rule_based_label, CHEXPERT_CLASSES
except ImportError:
    # Chạy trực tiếp: python data/preprocess_iu_xray.py
    from tokenizer import SimpleTokenizer, SPECIAL
    from chexpert_labeler import rule_based_label, CHEXPERT_CLASSES


def load_annotation(data_dir: str):
    """Đọc annotation.json chuẩn. Nếu chưa có, thử build từ indiana_reports.csv."""
    ann_path = os.path.join(data_dir, "annotation.json")
    if os.path.exists(ann_path):
        with open(ann_path) as f:
            return json.load(f)
    # Fallback: build từ CSV (cột uid, findings, impression) + projections (uid, filename)
    csv_reports = os.path.join(data_dir, "indiana_reports.csv")
    csv_proj = os.path.join(data_dir, "indiana_projections.csv")
    if os.path.exists(csv_reports) and os.path.exists(csv_proj):
        print("Tìm thấy CSV gốc, đang build annotation.json ... (xem code, có thể cần tinh chỉnh)")
        return build_annotation_from_csv(csv_reports, csv_proj)
    raise FileNotFoundError(
        f"Không tìm thấy annotation.json hay indiana_*.csv trong {data_dir}. "
        "Tải IU-Xray về (xem README) rồi đặt annotation.json + thư mục images/ vào đó."
    )


def build_annotation_from_csv(csv_reports, csv_proj):
    """Phù hợp bộ Kaggle raddar/chest-xrays-indiana-university.
    Lọc theo paper: chỉ giữ báo cáo có đủ findings+impression (không rỗng)
    VÀ có đủ 2 ảnh (frontal + lateral). Mục tiêu ~3195 mẫu (paper).

    CSV:
      - indiana_reports.csv   : uid, MeSH, Projection, findings, impression
      - indiana_projections.csv: uid, filename, projection
    """
    import csv

    def lower_row(row):
        return {k.strip().lower(): (v.strip() if v is not None else "") for k, v in row.items()}

    # Đọc reports — chỉ giữ uid có cả findings VÀ impression không rỗng
    reports = {}
    with open(csv_reports, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            r = lower_row(row)
            findings  = r.get("findings", "").strip()
            impression = r.get("impression", "").strip()
            # Paper: yêu cầu cả hai phần findings và impression đều có nội dung
            if findings and impression:
                text = findings + " " + impression
                reports[str(r["uid"]).strip()] = text

    # Đọc projections — phân loại frontal / lateral theo cột projection
    frontal_imgs = {}   # uid -> filename frontal
    lateral_imgs = {}   # uid -> filename lateral
    with open(csv_proj, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            r = lower_row(row)
            uid   = str(r["uid"]).strip()
            fname = r.get("filename", "").strip()
            proj  = r.get("projection", "").strip().lower()
            if not fname:
                continue
            # Phân loại: frontal = PA/AP, lateral = lateral/ll
            if proj in ("frontal", "pa", "ap"):
                frontal_imgs.setdefault(uid, fname)
            elif proj in ("lateral", "ll", "left lateral"):
                lateral_imgs.setdefault(uid, fname)
            else:
                # Fallback theo thứ tự file
                frontal_imgs.setdefault(uid, fname)

    # Giữ chỉ uid có đủ report + 2 ảnh (frontal + lateral)
    data = []
    skipped_no_report = 0
    skipped_no_lateral = 0
    for uid, report_text in reports.items():
        if uid not in frontal_imgs:
            skipped_no_report += 1
            continue
        if uid not in lateral_imgs:
            skipped_no_lateral += 1
            continue
        data.append({
            "id": uid,
            "image_path": [
                os.path.join("images", frontal_imgs[uid]),
                os.path.join("images", lateral_imgs[uid]),
            ],
            "report": report_text,
        })

    print(f"  Lọc: bỏ {skipped_no_report} uid thiếu ảnh, "
          f"{skipped_no_lateral} uid thiếu lateral → còn {len(data)} mẫu đủ điều kiện")
    return data


def compute_term_weights(tokenizer: SimpleTokenizer, train_reports):
    """Trọng số từ: từ hiếm (tên bệnh) -> trọng số cao. IDF-like, chuẩn hóa mean=1."""
    counter = Counter()
    for r in train_reports:
        counter.update(tokenizer.tokenize(r))
    N = sum(counter.values()) + 1
    weights = torch.ones(len(tokenizer))
    for w, c in counter.items():
        if w in tokenizer.vocab:
            idx = tokenizer.vocab[w]
            weights[idx] = 1.0 + torch.log(torch.tensor(N / (1.0 + c))).item()
    weights = weights / weights.mean()  # giữ trung bình ~1 để loss không lệch
    return weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--max_len", type=int, default=60)
    ap.add_argument("--min_freq", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    data = load_annotation(args.data_dir)

    # Lưu annotation.json để lần sau không cần parse CSV lại
    with open(os.path.join(args.data_dir, "annotation.json"), "w") as f:
        json.dump(data, f)

    # Kiểm tra ảnh có tồn tại không
    sample_paths = data[0]["image_path"]
    missing = [p for p in sample_paths if not os.path.exists(os.path.join(args.data_dir, p))]
    if missing:
        print(f"[CẢNH BÁO] Không tìm thấy ảnh {missing}. Hãy đảm bảo thư mục images/ nằm trong {args.data_dir}.")

    random.shuffle(data)
    n = len(data)
    # Paper: 80% train, 10% val, 10% test
    n_test = int(0.1 * n)
    n_val  = int(0.1 * n)
    splits = {
        "train": data[: n - n_val - n_test],
        "val":   data[n - n_val - n_test : n - n_test],
        "test":  data[n - n_test :],
    }
    print(f"Tổng {n} mẫu → train/val/test = "
          f"{len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])} "
          f"(paper: ~{int(n*0.8)}/{int(n*0.1)}/{int(n*0.1)})")

    # Build vocab chỉ từ train
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab([d["report"] for d in splits["train"]], min_freq=args.min_freq)
    tokenizer.save(os.path.join(args.data_dir, "vocab.json"))
    print(f"Vocab size: {len(tokenizer)}")

    term_weights = compute_term_weights(tokenizer, [d["report"] for d in splits["train"]])
    torch.save(term_weights, os.path.join(args.data_dir, "term_weights.pt"))

    for split, items in splits.items():
        processed = []
        for d in items:
            processed.append({
                "id": d.get("id", ""),
                "image_path": d["image_path"],
                "report": d["report"],
                "report_ids": tokenizer.encode(d["report"], args.max_len),
                "mlc_labels": rule_based_label(d["report"]),
            })
        out = os.path.join(args.data_dir, f"processed_{split}.json")
        with open(out, "w") as f:
            json.dump(processed, f)
        print(f"Đã lưu {out} ({len(processed)} mẫu)")

    print("Xong. Giờ chạy: python train.py --data_dir", args.data_dir)


if __name__ == "__main__":
    main()
