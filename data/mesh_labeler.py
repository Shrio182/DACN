# mesh_labeler.py
# Bộ trích xuất nhãn bệnh lý đa nhãn (Multi-Label Target) từ trường MeSH của bộ dữ liệu IU-Xray.
# IU-Xray chứa các thẻ chẩn đoán chuẩn y khoa Medical Subject Headings (MeSH).
# Quy trình:
# 1. Quét toàn bộ trường MeSH trong indiana_reports.csv, lấy từ khóa gốc (Root Tag đứng trước dấu /).
# 2. Lọc các thẻ có tần suất xuất hiện tối thiểu >= min_freq (mặc định: 10 lần) để tạo 65 lớp bệnh chuẩn.
# 3. Mã hóa chẩn đoán của từng bệnh nhân thành vector nhị phân multi-hot (0 và 1).

import os
import csv
import json
from collections import Counter


def build_mesh_vocab(csv_path: str, min_freq: int = 10):
    # Xây dựng danh sách các nhãn bệnh MeSH từ file indiana_reports.csv
    counter = Counter()
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mesh = row.get("MeSH", "").strip()
            if not mesh:
                continue
            tags = [t.strip() for t in mesh.split(";") if t.strip()]
            roots = set()
            for t in tags:
                root = t.split("/")[0].strip().lower()
                if root:
                    roots.add(root)
            counter.update(roots)

    # Giữ lại các nhãn có tần suất xuất hiện >= min_freq và sắp xếp thứ tự chữ cái cố định
    mesh_classes = sorted([tag for tag, count in counter.items() if count >= min_freq])
    mesh_to_idx = {tag: i for i, tag in enumerate(mesh_classes)}
    return mesh_classes, mesh_to_idx


def mesh_label(mesh_str: str, mesh_to_idx: dict):
    # Chuyển chuỗi MeSH của một ca chụp thành vector nhị phân float [0.0, 1.0, ...]
    labels = [0.0] * len(mesh_to_idx)
    if not mesh_str:
        return labels
    tags = [t.strip() for t in mesh_str.split(";") if t.strip()]
    for t in tags:
        root = t.split("/")[0].strip().lower()
        if root in mesh_to_idx:
            labels[mesh_to_idx[root]] = 1.0
    return labels


def save_mesh_vocab(mesh_classes: list, mesh_to_idx: dict, out_path: str):
    # Lưu danh sách 65 nhãn MeSH ra file JSON
    data = {
        "num_labels": len(mesh_classes),
        "mesh_classes": mesh_classes,
        "mesh_to_idx": mesh_to_idx,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_mesh_vocab(json_path: str):
    # Đọc lại danh sách nhãn MeSH đã lưu
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["mesh_classes"], data["mesh_to_idx"]
