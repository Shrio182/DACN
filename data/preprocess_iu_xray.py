# preprocess_iu_xray.py
# Kịch bản tiền xử lý toàn diện bộ dữ liệu IU-Xray từ các file CSV gốc:
# Đầu vào:
#   - indiana_reports.csv (chứa uid, MeSH tags, findings, impression)
#   - indiana_projections.csv (chứa uid, filename, góc chụp projection)
#   - Thư mục images/ chứa các ảnh X-quang
# Đầu ra:
#   - annotation.json : Tổng hợp các mẫu hợp lệ có đủ cặp ảnh Frontal + Lateral và văn bản báo cáo
#   - mesh_labels.json: 65 nhãn bệnh MeSH root tags tiêu chuẩn
#   - vocab.json      : Từ điển từ vựng được xây dựng riêng từ tập huấn luyện (train set)
#   - term_weights.pt : Bảng trọng số TF-IDF phục vụ tính hàm tổn thất RG Loss
#   - processed_train.json, processed_val.json, processed_test.json (chia tỉ lệ 80% / 10% / 10%)

import os
import json
import argparse
import random
from collections import Counter
import torch

try:
    from .tokenizer import SimpleTokenizer, SPECIAL
    from .mesh_labeler import build_mesh_vocab, mesh_label, save_mesh_vocab
except ImportError:
    from tokenizer import SimpleTokenizer, SPECIAL
    from mesh_labeler import build_mesh_vocab, mesh_label, save_mesh_vocab


def build_annotation_from_csv(csv_reports, csv_proj):
    # Lọc dữ liệu theo tiêu chuẩn khắt khe của bài báo IEEE TMI 2022:
    # 1. Báo cáo phải có đầy đủ cả 2 phần Findings VÀ Impression (không được rỗng).
    # 2. Bệnh nhân phải có đủ 2 ảnh X-quang tương ứng 2 góc chụp: Frontal (PA/AP) và Lateral (Nghiêng).
    import csv

    def lower_row(row):
        return {k.strip().lower(): (v.strip() if v is not None else "") for k, v in row.items()}

    # Bước 1: Đọc và lọc báo cáo y tế
    reports = {}
    mesh_data = {}
    with open(csv_reports, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            r = lower_row(row)
            findings = r.get("findings", "").strip()
            impression = r.get("impression", "").strip()
            if findings and impression:
                text = findings + " " + impression
                uid = str(r["uid"]).strip()
                reports[uid] = text
                mesh_data[uid] = row.get("MeSH", "").strip()

    # Bước 2: Đọc file góc chiếu và phân loại Frontal / Lateral
    frontal_imgs = {}
    lateral_imgs = {}
    with open(csv_proj, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            r = lower_row(row)
            uid = str(r["uid"]).strip()
            fname = r.get("filename", "").strip()
            proj = r.get("projection", "").strip().lower()
            if not fname:
                continue
            if proj in ("frontal", "pa", "ap"):
                frontal_imgs.setdefault(uid, fname)
            elif proj in ("lateral", "ll", "left lateral"):
                lateral_imgs.setdefault(uid, fname)
            else:
                frontal_imgs.setdefault(uid, fname)

    # Bước 3: Ghép cặp mẫu đạt chuẩn
    data = []
    skipped_no_frontal = 0
    skipped_no_lateral = 0
    for uid, report_text in reports.items():
        if uid not in frontal_imgs:
            skipped_no_frontal += 1
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
            "mesh": mesh_data.get(uid, ""),
        })

    print(f"  Lọc: bỏ {skipped_no_frontal} uid thiếu frontal, "
          f"{skipped_no_lateral} uid thiếu lateral → còn {len(data)} mẫu đủ điều kiện")
    return data


def compute_term_weights(tokenizer: SimpleTokenizer, train_reports):
    # Tính trọng số nghịch đảo tần suất từ (tương tự TF-IDF) theo Công thức 5 của bài báo:
    # w_k = 1 + log(N / (1 + count_k))
    # Giúp phạt nặng hơn khi mô hình sinh sai các thuật ngữ bệnh lý hiếm gặp
    counter = Counter()
    for r in train_reports:
        counter.update(tokenizer.tokenize(r))
    N = sum(counter.values()) + 1
    weights = torch.ones(len(tokenizer))
    for w, c in counter.items():
        if w in tokenizer.vocab:
            idx = tokenizer.vocab[w]
            weights[idx] = 1.0 + torch.log(torch.tensor(N / (1.0 + c))).item()
    # Chuẩn hóa về giá trị trung bình bằng 1 để giữ thang độ lớn loss ổn định
    weights = weights / weights.mean()
    return weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Thư mục chứa dữ liệu IU-Xray")
    ap.add_argument("--max_len", type=int, default=60, help="Độ dài tối đa của báo cáo")
    ap.add_argument("--min_freq", type=int, default=1, help="Tần suất xuất hiện tối thiểu của từ trong vocab")
    ap.add_argument("--mesh_min_freq", type=int, default=10, help="Ngưỡng tần suất tối thiểu cho nhãn MeSH")
    ap.add_argument("--seed", type=int, default=42, help="Hạt giống ngẫu nhiên chia split")
    args = ap.parse_args()

    random.seed(args.seed)

    # 1. Trích xuất danh sách mẫu từ file CSV
    csv_reports = os.path.join(args.data_dir, "indiana_reports.csv")
    csv_proj = os.path.join(args.data_dir, "indiana_projections.csv")
    if not os.path.exists(csv_reports) or not os.path.exists(csv_proj):
        raise FileNotFoundError(
            f"Không tìm thấy indiana_reports.csv hoặc indiana_projections.csv trong {args.data_dir}."
        )
    print("Đang đọc và tiền xử lý dữ liệu từ các file CSV...")
    data = build_annotation_from_csv(csv_reports, csv_proj)

    with open(os.path.join(args.data_dir, "annotation.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # 2. Xây dựng danh mục nhãn bệnh lý MeSH
    print(f"\nXây dựng nhãn bệnh lý MeSH (min_freq={args.mesh_min_freq})...")
    mesh_classes, mesh_to_idx = build_mesh_vocab(csv_reports, min_freq=args.mesh_min_freq)
    mesh_labels_path = os.path.join(args.data_dir, "mesh_labels.json")
    save_mesh_vocab(mesh_classes, mesh_to_idx, mesh_labels_path)
    print(f"  Số nhãn bệnh lý MeSH trích xuất được: {len(mesh_classes)}")

    # 3. Phân chia tập dữ liệu: Train (80%), Val (10%), Test (10%)
    random.shuffle(data)
    n = len(data)
    n_test = int(0.1 * n)
    n_val = int(0.1 * n)
    splits = {
        "train": data[: n - n_val - n_test],
        "val": data[n - n_val - n_test : n - n_test],
        "test": data[n - n_test :],
    }
    print(f"\nTổng số mẫu hợp lệ: {n} → Train/Val/Test = "
          f"{len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")

    # 4. Xây dựng từ điển chỉ từ tập Train (tránh rò rỉ dữ liệu)
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab([d["report"] for d in splits["train"]], min_freq=args.min_freq)
    tokenizer.save(os.path.join(args.data_dir, "vocab.json"))
    print(f"Kích thước từ điển (Vocab size): {len(tokenizer)} từ")

    # 5. Tính toán bảng trọng số TF-IDF
    term_weights = compute_term_weights(tokenizer, [d["report"] for d in splits["train"]])
    torch.save(term_weights, os.path.join(args.data_dir, "term_weights.pt"))

    # 6. Lưu file processed cho từng tập
    for split, items in splits.items():
        processed = []
        for d in items:
            processed.append({
                "id": d.get("id", ""),
                "image_path": d["image_path"],
                "report": d["report"],
                "report_ids": tokenizer.encode(d["report"], args.max_len),
                "mlc_labels": mesh_label(d.get("mesh", ""), mesh_to_idx),
            })
        out = os.path.join(args.data_dir, f"processed_{split}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(processed, f, ensure_ascii=False)
        print(f"Đã lưu {out} ({len(processed)} mẫu)")

    print(f"\nTiền xử lý hoàn tất thành công!")
    print(f"Có thể bắt đầu huấn luyện: python train.py --data_dir {args.data_dir}")


if __name__ == "__main__":
    main()
