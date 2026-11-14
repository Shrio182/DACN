# chexpert_labeler.py
# Bộ gán nhãn 14 bệnh lý CheXpert dựa trên tập luật (Rule-Based Keyword Matching).
# Hỗ trợ nhận diện các từ khóa bệnh lý ngực thường gặp và xử lý ngữ cảnh phủ định
# (Ví dụ: "no evidence of pneumonia", "lungs are clear" -> không gán nhãn viêm phổi).

import re

# 14 nhóm bệnh lý tiêu chuẩn của CheXpert (Stanford)
CHEXPERT_CLASSES = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Effusion", "Pleural Thickening", "Fibrosis", "Fracture", "Support Devices",
]

# Từ điển các từ khóa tương ứng với từng nhóm bệnh lý
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

# Các cụm từ phủ định thường dùng trong báo cáo X-quang
NEGATION_TERMS = [
    "no", "without", "negative for", "clear of", "free of", "normal",
    "absence of", "no evidence of", "without evidence of", "resolved"
]


def rule_based_label(report: str):
    # Phân tích văn bản báo cáo và trả về vector 14 phần tử mang giá trị 0.0 hoặc 1.0
    text = " " + report.lower() + " "
    labels = [0.0] * len(CHEXPERT_CLASSES)
    
    for i, cls_name in enumerate(CHEXPERT_CLASSES):
        for kw in KEYWORDS[cls_name]:
            for m in re.finditer(re.escape(kw), text):
                # Kiểm tra 40 ký tự phía trước từ khóa xem có từ ngữ mang nghĩa phủ định không
                start = max(0, m.start() - 40)
                prefix = text[start:m.start()]
                is_neg = any(neg in prefix for neg in NEGATION_TERMS)
                if not is_neg:
                    labels[i] = 1.0
                    break
            if labels[i] == 1.0:
                break
    return labels
