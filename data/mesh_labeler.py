"""
mesh_labeler.py — Sinh multi-label targets từ MeSH tags thực tế của IU-Xray.
==============================================================================
Paper gốc sử dụng 14 structured labels cho MIMIC-CXR.
IU-Xray có trường MeSH trong indiana_reports.csv → dùng nó để xây dựng
multi-label targets THỰC TẾ thay vì áp dụng CheXpert 14 lớp.

Quy trình:
1. build_mesh_vocab(): Quét toàn bộ MeSH tags từ CSV, lấy root tag (trước /),
   giữ các tag xuất hiện >= min_freq lần → tạo vocabulary nhãn.
2. mesh_label(): Cho 1 report với MeSH string, trả về multi-hot vector.

Lưu ý: "normal" và "no indexing" cũng là MeSH tags hợp lệ — giữ lại.
"""
import os
import csv
import json


def build_mesh_vocab(csv_path: str, min_freq: int = 10):
    """Quét indiana_reports.csv, xây dựng vocabulary nhãn MeSH.

    Args:
        csv_path: Đường dẫn tới indiana_reports.csv
        min_freq: Ngưỡng tần số tối thiểu để giữ tag (default: 10)

    Returns:
        mesh_classes: list các MeSH root tags (đã sort)
        mesh_to_idx: dict {tag: index}
    """
    from collections import Counter
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

    # Giữ các tag >= min_freq, sort để thứ tự ổn định
    mesh_classes = sorted([tag for tag, count in counter.items() if count >= min_freq])
    mesh_to_idx = {tag: i for i, tag in enumerate(mesh_classes)}
    return mesh_classes, mesh_to_idx


def mesh_label(mesh_string: str, mesh_to_idx: dict) -> list:
    """Chuyển MeSH string thành multi-hot vector.

    Args:
        mesh_string: Chuỗi MeSH gốc (ví dụ: "Cardiomegaly/mild;Lung/hypoinflation")
        mesh_to_idx: Dict mapping tag → index

    Returns:
        list float [0.0, 1.0, ...] độ dài = len(mesh_to_idx)
    """
    num_labels = len(mesh_to_idx)
    labels = [0.0] * num_labels
    if not mesh_string:
        return labels
    tags = [t.strip() for t in mesh_string.split(";") if t.strip()]
    for t in tags:
        root = t.split("/")[0].strip().lower()
        if root in mesh_to_idx:
            labels[mesh_to_idx[root]] = 1.0
    return labels


def save_mesh_vocab(mesh_classes: list, save_path: str):
    """Lưu danh sách nhãn MeSH ra file JSON."""
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({"mesh_classes": mesh_classes, "num_labels": len(mesh_classes)}, f, indent=2)
    print(f"Đã lưu {len(mesh_classes)} MeSH labels → {save_path}")


def load_mesh_vocab(load_path: str):
    """Đọc vocabulary nhãn MeSH từ file JSON."""
    with open(load_path, encoding="utf-8") as f:
        data = json.load(f)
    mesh_classes = data["mesh_classes"]
    mesh_to_idx = {tag: i for i, tag in enumerate(mesh_classes)}
    return mesh_classes, mesh_to_idx

