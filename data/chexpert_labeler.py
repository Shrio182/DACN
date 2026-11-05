"""
chexpert_labeler.py — Sinh nhãn 14 bệnh (đầu vào cho MLC head) bằng rule-based.

PHƯƠNG ÁN TRIỂN KHAI THAY THẾ:
  Paper gốc sử dụng 14 structured labels từ MIMIC-CXR (Bảng I).
  IU-Xray gốc chỉ có trường MeSH tags (không có 14 nhãn cấu trúc).
  Ở đây sử dụng CheXpert rule-based labeler áp dụng lên IU-Xray như
  phương án thay thế — KHÔNG phải cấu hình nhãn được paper xác nhận.
  Để chính xác hơn, nên chạy CheXpert Labeler chính thức (Stanford)
  hoặc xây dựng multi-label targets từ trường MeSH của IU-Xray.

Cơ chế: quét keyword, xử lý phủ định đơn giản (no/without/clear/normal ... + bệnh = 0).
"""
import re

# 14 lớp theo CheXpert
CHEXPERT_CLASSES = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Effusion", "Pleural Thickening", "Fibrosis", "Fracture", "Support Devices",
]

KEYWORDS = {
    "Enlarged Cardiomediastinum": ["mediastinal widening", "widened mediastinum", "enlarged mediastinum"],
    "Cardiomegaly": ["cardiomegaly", "enlarged heart", "cardiac silhouette enlarged", "heart size enlarged"],
    "Lung Opacity": ["opacity", "opacification", "hazy opacity", "airspace disease"],
    "Lung Lesion": ["lung lesion", "nodule", "mass", "coin lesion", "pulmonary nodule"],
    "Edema": ["edema", "pulmonary edema", "vascular congestion", "cephalization"],
    "Consolidation": ["consolidation", "consolidated"],
    "Pneumonia": ["pneumonia", "pneumonic"],
    "Atelectasis": ["atelectasis", "atelectatic", "plate-like"],
    "Pneumothorax": ["pneumothorax", "pleural air"],
    "Effusion": ["effusion", "pleural fluid", "blunting of the costophrenic"],
    "Pleural Thickening": ["pleural thickening", "pleural scar"],
    "Fibrosis": ["fibrosis", "fibrotic", "scarring", "interstitial disease"],
    "Fracture": ["fracture", "rib fracture"],
    "Support Devices": ["catheter", "tube", "line", "pacemaker", "device", "stent", "drain", "prosthetic"],
}

NEGATION_TERMS = ["no", "without", "negative for", "clear of", "free of", "normal",
                  "absence of", "no evidence of", "without evidence of", "resolved"]


def rule_based_label(report: str):
    """Trả về list 14 giá trị 0/1 (float)."""
    text = " " + report.lower() + " "
    labels = [0.0] * len(CHEXPERT_CLASSES)
    for i, cls_name in enumerate(CHEXPERT_CLASSES):
        for kw in KEYWORDS[cls_name]:
            for m in re.finditer(re.escape(kw), text):
                # Kiểm tra 3 từ ngay trước keyword có từ phủ định không
                prefix = text[max(0, m.start() - 40): m.start()]
                if any(neg in prefix for neg in NEGATION_TERMS):
                    continue  # bị phủ định -> không ghi nhận
                labels[i] = 1.0
                break
            if labels[i] == 1.0:
                break
    return labels
