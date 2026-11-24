# clinical_utils.py
# Mô-đun tiện ích hỗ trợ chẩn đoán lâm sàng cho hệ thống X-quang ngực
# Bao gồm: Dịch thuật ngữ y khoa chuyên sâu, tính tương đồng ROUGE-L, quản lý hồ sơ thẩm định và xuất phiếu kết quả.

import os
import re
import csv

import json
from datetime import datetime
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


# ── DANH MỤC NHÃN BỆNH LÝ CHEXPERT (14 LỚP) VÀ DỊCH NGHĨA Y KHOA ───────────────
CHEXPERT_TRANSLATIONS = {
    "Enlarged Cardiomediastinum": "Trung thất giãn rộng (Enlarged Cardiomediastinum)",
    "Cardiomegaly": "Bóng tim to (Cardiomegaly)",
    "Lung Opacity": "Mờ phế trường phổi (Lung Opacity)",
    "Lung Lesion": "Tổn thương dạng nốt/khối phổi (Lung Lesion)",
    "Edema": "Phù phổi cấp / Ứ huyết (Pulmonary Edema)",
    "Consolidation": "Đông đặc phế nang (Consolidation)",
    "Pneumonia": "Viêm phổi (Pneumonia)",
    "Atelectasis": "Xẹp phổi (Atelectasis)",
    "Pneumothorax": "Tràn khí màng phổi (Pneumothorax)",
    "Effusion": "Tràn dịch màng phổi (Pleural Effusion)",
    "Pleural Thickening": "Dày dính màng phổi (Pleural Thickening)",
    "Fibrosis": "Xơ phổi / Xơ hóa mô kẽ (Fibrosis)",
    "Fracture": "Gãy xương sườn / đòn (Fracture)",
    "Support Devices": "Thiết bị can thiệp y tế (Support Devices)",
}


def has_spinal_degeneration(text: str) -> bool:
    if not isinstance(text, str):
        return False
    has_degenerative_finding = re.search(
        r"\b(?:degenerative|spondylosis|osteophyte\w*)\b",
        text,
        flags=re.IGNORECASE,
    )
    has_spinal_location = re.search(
        r"\b(?:spine|thoracic|cervical|lumbar)\b",
        text,
        flags=re.IGNORECASE,
    )
    return bool(has_degenerative_finding and has_spinal_location)


SPINAL_DEGENERATION_VI = (
    "Ghi nhận thay đổi thoái hóa cột sống ngực mức độ nhẹ; "
    "không thấy gãy xương cấp tính."
)


# ── DỊCH NGHĨA CÁC NHÃN MESH THƯỜNG GẶP TRÊN IU-XRAY ──────────────────────────
MESH_TRANSLATIONS = {
    "normal": "Bình thường (Normal)",
    "no indexing": "Không phân nhóm bệnh (No Indexing)",
    "cardiomegaly": "Bóng tim to (Cardiomegaly)",
    "pleural effusion": "Tràn dịch màng phổi (Pleural Effusion)",
    "pneumothorax": "Tràn khí màng phổi (Pneumothorax)",
    "infiltrate": "Thâm nhiễm nhu mô phổi (Infiltrate)",
    "consolidation": "Đông đặc phế nang (Consolidation)",
    "opacity": "Đám mờ phế trường (Lung Opacity)",
    "atelectasis": "Xẹp phổi (Atelectasis)",
    "pulmonary atelectasis": "Xẹp phổi phế nang (Pulmonary Atelectasis)",
    "edema": "Phù phổi (Pulmonary Edema)",
    "pulmonary edema": "Phù phổi / Ứ huyết tĩnh mạch (Pulmonary Edema)",
    "emphysema": "Khí phế thũng (Emphysema)",
    "pulmonary emphysema": "Khí phế thũng (Pulmonary Emphysema)",
    "pneumonia": "Viêm phổi (Pneumonia)",
    "nodule": "Nốt mờ nhu mô phổi (Lung Nodule)",
    "granuloma": "U hạt phổi (Granuloma)",
    "granulomatous disease": "Bệnh u hạt phổi (Granulomatous Disease)",
    "calcinosis": "Vôi hóa nhu mô / hạch (Calcinosis)",
    "calcified granuloma": "U hạt vôi hóa (Calcified Granuloma)",
    "wire": "Chỉ thép khâu xương ức (Sternotomy Wire)",
    "sternotomy": "Vết mổ mở xương ức (Sternotomy)",
    "spine": "Cột sống ngực (Thoracic Spine)",
    "spondylosis": "Thoái hóa đốt sống ngực (Thoracic Spondylosis)",
    "scoliosis": "Vẹo cột sống ngực (Thoracic Scoliosis)",
    "kyphosis": "Gù cột sống ngực (Thoracic Kyphosis)",
    "lordosis": "Ưỡn cột sống (Lordosis)",
    "ribs": "Xương sườn (Ribs)",
    "bone": "Xương thành ngực (Bone)",
    "fractures, bone": "Gãy xương sườn / đòn (Bone Fracture)",
    "cicatrix": "Sẹo xơ nhu mô phổi (Cicatrix / Scar)",
    "diaphragm": "Vòm hoành (Diaphragm)",
    "costophrenic angle": "Góc sườn hoành (Costophrenic Angle)",
    "heart": "Bóng tim (Heart)",
    "mediastinum": "Trung thất (Mediastinum)",
    "aorta": "Quai động mạch chủ (Aorta)",
    "aorta, thoracic": "Động mạch chủ ngực (Thoracic Aorta)",
    "airspace disease": "Bệnh lý phế nang (Airspace Disease)",
    "pulmonary disease, chronic obstructive": "Bệnh phổi tắc nghẽn mạn tính (COPD)",
    "pulmonary congestion": "Sung huyết mạch máu phổi (Pulmonary Congestion)",
    "pulmonary fibroses": "Xơ phổi tiến triển (Pulmonary Fibrosis)",
    "catheters, indwelling": "Ống thông lưu tĩnh mạch (Indwelling Catheter)",
    "surgical instruments": "Dụng cụ phẫu thuật lồng ngực (Surgical Clips)",
    "thoracic vertebrae": "Đốt sống ngực (Thoracic Vertebrae)",
    "thoracotomy": "Mở lồng ngực (Thoracotomy)",
    "vascular calcification": "Vôi hóa thành mạch (Vascular Calcification)",
    "atherosclerosis": "Xơ vữa động mạch (Atherosclerosis)",
    "hypoinflation": "Phổi kém giãn nở (Hypoinflation)",
    "hernia, hiatal": "Thoát vị khe hoành (Hiatal Hernia)",
    "lymph nodes": "Hạch bạch huyết trung thất (Lymph Nodes)",
    "tuberculosis": "Lao phổi (Tuberculosis)",
}


def format_clinical_report(raw_text: str) -> str:
    """Normalize generated English radiology text into concise clinical sentences."""
    if not raw_text or not isinstance(raw_text, str):
        return ""

    text = raw_text.replace("xxxx", " ").replace("XXXX", " ")
    text = re.sub(r"<(?:bos|eos|pad|unk)>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\n+\s*", ". ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*,\s*\.", ",", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*", ", ", text)
    text = re.sub(r"\s*\.\s*", ". ", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\s+", " ", text).strip(" .")

    landmarks = (
        "there is", "there are", "bony structures", "no pneumothorax",
        "no pleural", "no focal", "no acute", "cardiomediastinal",
        "lungs are", "lungs", "heart", "mediastinum", "lung", "pleural",
        "osseous", "pneumothorax", "pneumonia", "cardiomegaly", "findings",
        "consolidation", "edema", "effusion", "acute", "impression",
    )
    connectors = {
        "no", "or", "nor", "without", "for", "of", "with", "and",
        "the", "a", "an", "is", "are",
    }
    compound_predecessors = {
        "heart": {"normal", "enlarged", "large"},
        "cardiomegaly": {"mild", "moderate", "borderline"},
        "consolidation": {"focal", "lobar", "airspace"},
        "edema": {"pulmonary"},
        "effusion": {"pleural"},
        "pneumothorax": {"small", "large", "left", "right"},
    }
    insertion_points = set()
    for landmark in sorted(landmarks, key=len, reverse=True):
        pattern = re.compile(rf"\b{re.escape(landmark)}\b", re.IGNORECASE)
        for match in pattern.finditer(text):
            start = match.start()
            if landmark in {"heart", "lung", "lungs", "mediastinum", "pleural"}:
                article = re.search(r"\b(the|a|an)\s+$", text[:start], re.IGNORECASE)
                if article:
                    start -= len(article.group(0))
            prefix = text[:start]
            if prefix.endswith(". ") or not prefix:
                continue
            previous = prefix.rstrip()
            if previous.endswith(","):
                continue
            words = re.findall(r"[a-z']+", previous.lower())
            if words and words[-1] in connectors:
                continue
            if words and words[-1] in compound_predecessors.get(landmark, set()):
                continue
            if landmark == "consolidation" and len(words) >= 2:
                if words[-2:] == ["no", "focal"]:
                    continue

            if start > 0 and not text[:start].endswith(". "):
                insertion_points.add(start)

    for point in sorted(insertion_points, reverse=True):
        text = text[:point].rstrip() + ". " + text[point:].lstrip()

    text = re.sub(r"\s*,\s*\.", ",", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r"\s*\.\s*", ". ", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\s+", " ", text).strip(" .")

    dangling = {
        "and", "or", "nor", "with", "of", "in", "to", "from", "the", "a", "an",
        "has", "have", "had", "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did", "without",
    }
    sentences = []
    for sentence in (part.strip(" .") for part in text.split(".")):
        sentence = re.sub(
            r"^(?:findings|impression)\s*[:=-]?\s*",
            "",
            sentence,
            flags=re.IGNORECASE,
        ).strip(" ,")
        words = sentence.split()
        was_truncated = False
        ended_with_auxiliary = False
        while words and words[-1].lower() in dangling:
            removed = words.pop().lower()
            was_truncated = True
            ended_with_auxiliary = ended_with_auxiliary or removed in {
                "has", "have", "had", "is", "are", "was", "were", "be", "been",
                "being", "do", "does", "did",
            }
        if was_truncated and "and" in [word.lower() for word in words]:
            last_conjunction = max(
                index for index, word in enumerate(words) if word.lower() == "and"
            )
            words = words[:last_conjunction]
        elif ended_with_auxiliary or len(words) < 3:
            continue
        if len(words) >= 2:
            normalized = " ".join(words)
            normalized = normalized[0].upper() + normalized[1:]
            sentences.append(normalized)

    if not sentences:
        return "Heart size and mediastinal contours are within normal limits. The lungs are clear."
    return ". ".join(sentences) + "."


def split_clinical_report(raw_text: str) -> dict:
    """Split normalized text into FINDINGS and IMPRESSION sections."""
    text = format_clinical_report(raw_text)
    sentences = [sentence.strip() for sentence in text.split(".") if sentence.strip()]

    findings = []
    impression = []
    for sentence in sentences:
        low = sentence.lower()
        if any(marker in low for marker in ("no acute", "conclusion", "impression")):
            impression.append(sentence)
        else:
            findings.append(sentence)

    if not impression:
        negative = [s for s in findings if s.lower().startswith("no ")]
        impression = negative[-1:] if negative else findings[-1:]
        if impression:
            findings = [s for s in findings if s not in impression]

    if not findings:
        findings = ["No focal abnormality is described."]
    if not impression:
        impression = ["No acute cardiopulmonary abnormality."]

    return {
        "findings": ". ".join(findings) + ".",
        "impression": ". ".join(impression) + ".",
        "raw": text,
    }

# ── CÁC MẪU CÂU Y KHOA CHUẨN X-QUANG ĐỂ DỊCH TRỰC TIẾP ─────────────────────────
PATTERNS = [
    # 1. Tim, trung thất & rốn phổi
    ("the heart is normal in size and contour", "Kích thước và đường viền bóng tim trong giới hạn bình thường."),
    ("heart size and mediastinal contours are within normal limits", "Kích thước tim và đường viền trung thất nằm trong giới hạn bình thường."),
    ("the cardiac silhouette and mediastinum size are within normal limits", "Kích thước bóng tim và trung thất nằm trong giới hạn bình thường."),
    ("the cardiac and mediastinal contours are within normal limits", "Đường viền tim và trung thất trong giới hạn bình thường."),
    ("cardiac , mediastinal and hilar contours are normal", "Đường viền tim, trung thất và rốn phổi bình thường."),
    ("cardiac, mediastinal and hilar contours are normal", "Đường viền tim, trung thất và rốn phổi bình thường."),
    ("the cardiomediastinal silhouette is normal", "Bóng tim và trung thất bình thường."),
    ("the cardiomediastinal silhouette is within normal limits for size and contour", "Bóng tim và trung thất có kích thước và đường viền trong giới hạn bình thường."),
    ("the cardiomediastinal silhouette is within normal limits", "Bóng tim và trung thất trong giới hạn bình thường."),
    ("the cardiomediastinal silhouette is unremarkable", "Bóng tim và trung thất không ghi nhận bất thường."),
    ("cardio mediastinal silhouette is unremarkable", "Bóng tim và trung thất không có dấu hiệu bệnh lý."),
    ("the heart is normal in size", "Kích thước tim bình thường."),
    ("the heart size is normal", "Kích thước tim bình thường."),
    ("heart size is normal", "Kích thước tim bình thường."),
    ("heart size within normal limits", "Kích thước tim trong giới hạn bình thường."),
    ("normal heart size", "Kích thước tim bình thường."),
    ("the mediastinum is unremarkable", "Vùng trung thất không có dấu hiệu bất thường."),
    ("the cardiac contours are normal", "Đường viền bóng tim bình thường."),
    ("the cardiac silhouette is normal", "Bóng tim bình thường."),
    ("cardiac silhouette is normal", "Bóng tim bình thường."),
    ("the hila and pulmonary vasculature are normal", "Rốn phổi và tuần hoàn mạch máu phổi bình thường."),
    ("pulmonary vasculature is normal", "Tuần hoàn mạch máu phổi bình thường."),

    # 2. Phế trường, màng phổi & tổn thương âm tính
    ("the lungs are normally inflated without evidence of focal airspace disease pleural effusion or pneumothorax", "Hai phổi nở tốt, không có tổn thương phế nang khu trú, không có tràn dịch hay tràn khí màng phổi."),
    ("the lungs are clear without evidence of infiltrate", "Hai phế trường phổi sáng, không có dấu hiệu thâm nhiễm phế nang."),
    ("the lungs are well-inflated and clear", "Hai phổi giãn nở tốt và sáng rõ."),
    ("the lungs are clear bilaterally", "Hai phế trường phổi sáng rõ cả hai bên."),
    ("well-expanded and clear lungs", "Hai phổi giãn nở tốt và sáng rõ."),
    ("the lungs are clear", "Hai phế trường phổi sáng rõ."),
    ("clear lungs", "Hai phế trường phổi sáng."),
    ("there is no pneumothorax or pleural effusion", "Không có hình ảnh tràn khí hay tràn dịch màng phổi."),
    ("there is no pneumothorax or effusion", "Không phát hiện tràn khí hay tràn dịch màng phổi."),
    ("no pneumothorax or pleural effusion identified", "Không phát hiện tràn khí hay tràn dịch màng phổi."),
    ("no displaced rib fractures , pneumothorax , or pleural effusion identified", "Không có gãy xương sườn di lệch, tràn khí hay tràn dịch màng phổi."),
    ("no pneumothorax or pleural effusion", "Không có tràn khí hay tràn dịch màng phổi."),
    ("no pleural effusion or pneumothorax is seen", "Không thấy hình ảnh tràn dịch hoặc tràn khí màng phổi."),
    ("no pleural effusion or pneumothorax", "Không có tràn dịch hay tràn khí màng phổi."),
    ("no pleural effusion", "Không có tràn dịch màng phổi."),
    ("no pleural effusions", "Không có tràn dịch màng phổi."),
    ("no pneumothorax", "Không có tràn khí màng phổi."),
    ("there is no focal airspace consolidation or suspicious pulmonary opacity", "Không có đông đặc phế nang khu trú hoặc đám mờ phổi nghi ngờ."),
    ("there is no focal airspace consolidation", "Không thấy hình ảnh đông đặc phế nang khu trú."),
    ("no focal airspace consolidation to suggest pneumonia", "Không có đông đặc phế nang khu trú gợi ý viêm phổi."),
    ("no focal airspace consolidation", "Không có đông đặc phế nang khu trú."),
    ("there is no focal consolidation", "Không có đông đặc phế nang khu trú."),
    ("no focal areas of consolidation", "Không ghi nhận các vùng đông đặc khu trú."),
    ("no focal consolidation", "Không có đông đặc khu trú."),
    ("no consolidation", "Không có đông đặc phế nang."),
    ("no suspicious pulmonary opacities", "Không thấy đám mờ nghi ngờ ở nhu mô phổi."),
    ("no suspicious pulmonary opacity", "Không thấy đám mờ phổi nghi ngờ."),
    ("there is no pulmonary edema", "Không có hình ảnh phù phổi cấp hay ứ huyết mạch máu phổi."),
    ("no pulmonary edema", "Không có phù phổi cấp."),
    ("there is no nodules or masses", "Không thấy tổn thương dạng nốt hay khối u phổi."),
    ("no displaced rib fractures", "Không thấy gãy xương sườn di lệch."),
    ("low lung volumes are present", "Thể tích hai phổi giảm nhẹ."),
    ("low lung volumes", "Thể tích hai phổi giảm nhẹ."),

    # 3. Tiến triển bệnh & Hậu phẫu lâm sàng
    ("the previously seen small left pneumothorax has resolved", "Hình ảnh tràn khí màng phổi nhỏ bên trái trước đó đã thoái lui hồi phục."),
    ("resolution of pneumothorax", "Tràn khí màng phổi đã thoái lui."),
    ("rib fractures are stable", "Gãy xương sườn trong trạng thái ổn định."),
    ("unchanged left sixth and seventh rib fractures", "Gãy xương sườn số 6 và 7 bên trái không thay đổi so với phim cũ."),
    ("surgical clips are now present", "Kẹp phẫu thuật kim loại hiện diện ở thành ngực/trung thất."),
    ("surgical clips are present", "Có kẹp phẫu thuật trung thất."),
    ("surgical clips", "Kẹp phẫu thuật trung thất còn nguyên vị trí."),
    ("portable ap chest radiograph", "Phim chụp X-quang ngực tư thế trước-sau (AP) tại giường."),

    # 4. Kết luận lâm sàng
    ("no other acute cardiopulmonary process", "Không có diễn tiến bệnh lý tim phổi cấp tính nào khác."),
    ("no acute cardiopulmonary disease", "Không phát hiện bệnh lý tim phổi cấp tính."),
    ("no acute cardiopulmonary abnormality", "Không ghi nhận bất thường tim phổi cấp tính."),
    ("no acute cardiopulmonary process", "Không thấy quá trình bệnh lý tim phổi cấp tính."),
    ("no acute cardiopulmonary findings", "Không phát hiện dấu hiệu tim phổi cấp tính."),
    ("no acute radiographic cardiopulmonary process", "Hình ảnh X-quang chưa ghi nhận tổn thương tim phổi cấp tính."),
    ("no acute pulmonary findings", "Không phát hiện tổn thương phổi cấp tính."),
    ("no acute bony abnormality", "Khung xương lồng ngực không có bất thường cấp."),
    ("no acute bone abnormality", "Không có tổn thương xương cấp tính."),
    ("no acute disease", "Không có dấu hiệu bệnh lý cấp tính."),
    ("no evidence of active disease", "Không có bằng chứng bệnh lý đang tiến triển."),
    ("no acute findings", "Không ghi nhận bất thường cấp tính."),
    ("normal chest", "Hình ảnh X-quang tim phổi bình thường."),

    # 5. Bệnh lý bất thường
    ("borderline cardiomegaly", "Bóng tim to ranh giới (chớm to)."),
    ("moderate cardiomegaly", "Bóng tim to mức độ vừa."),
    ("mild cardiomegaly", "Bóng tim to nhẹ."),
    ("cardiomegaly", "Bóng tim to."),
    ("enlarged pulmonary arteries", "Giãn các nhánh động mạch phổi."),
    ("basilar atelectasis", "Xẹp phổi dạng đĩa ở vùng đáy phổi."),
    ("pulmonary atelectasis", "Xẹp phổi phế nang."),
    ("thoracic spondylosis", "Thoái hóa đốt sống ngực."),
    ("degenerative changes of the spine", "Thoái hóa cột sống ngực."),
    ("degenerative changes are present in the spine", "Có biến đổi thoái hóa ở cột sống ngực."),
    ("midline sternotomy wires", "Chỉ thép khâu xương ức đường giữa còn nguyên vẹn."),
    ("midline sternotomy", "Chỉ thép khâu xương ức đường giữa."),
    ("sternotomy wires", "Chỉ thép khâu cố định xương ức sau mổ."),
    ("calcified granuloma within the right upper lung", "U hạt vôi hóa ổn định ở thùy trên phổi phải."),
    ("calcified granuloma", "U hạt vôi hóa nhu mô phổi."),
    ("stable calcified granuloma", "U hạt vôi hóa phổi ổn định."),
    ("bullous emphysema and interstitial fibrosis", "Khí phế thũng bóng khí kèm xơ hóa mô kẽ phổi."),
    ("bullous emphysema", "Khí phế thũng bóng khí."),
    ("interstitial fibrosis", "Xơ hóa mô kẽ phổi."),
    ("diffuse bilateral interstitial and alveolar opacities", "Đám mờ mô kẽ và phế nang lan tỏa hai bên."),
    ("pulmonary edema", "Phù phổi cấp / ứ huyết."),
    ("confluent lobar consolidation", "Đông đặc thùy phổi."),
    ("in comparison with the study of", "So sánh với lần chụp trước đó,"),
    ("there is no significant change", "không ghi nhận biến đổi đáng kể."),
]


def translate_radiology_report(en_text: str) -> str:
    """
    Dịch báo cáo X-quang tiếng Anh sang tiếng Việt chuẩn văn phong lâm sàng.
    Sử dụng cơ chế khớp vùng ngữ nghĩa từ dài đến ngắn và sắp xếp theo trật tự câu gốc.
    """
    if not en_text or not isinstance(en_text, str):
        return ""

    # Dọn dẹp tiền tố danh sách hoặc định dạng 'findings: ... impression: ...'
    clean = en_text.replace("xxxx", "").replace("XXXX", "").replace("<bos>", "").replace("<eos>", "").replace("<pad>", "")
    clean = clean.replace("['", "").replace("']", "").replace('["', "").replace('"]', "")
    clean = re.sub(r"\bfindings:\s*", "Kết quả ghi nhận: ", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bimpression:\s*", ". Kết luận: ", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s+", " ", clean).strip()

    clean_lower = clean.lower()
    sorted_pats = sorted(PATTERNS, key=lambda x: len(x[0]), reverse=True)
    spans = []
    occupied = [False] * len(clean_lower)

    for pat, vi in sorted_pats:
        for m in re.finditer(re.escape(pat), clean_lower):
            st_pos, en_pos = m.start(), m.end()
            if not any(occupied[st_pos:en_pos]):
                for i in range(st_pos, en_pos):
                    occupied[i] = True
                spans.append((st_pos, vi))

    spans.sort(key=lambda x: x[0])
    res = []
    seen = set()
    for _, vi in spans:
        if vi not in seen:
            seen.add(vi)
            res.append(vi)

    if not res:
        fallback = clean_lower
        if "normal" in fallback or "clear" in fallback:
            return "Kích thước tim và trung thất trong giới hạn bình thường. Hai phế trường phổi sáng, không có dấu hiệu bệnh lý tim phổi cấp tính."
        return "Hình ảnh X-quang ngực chưa ghi nhận tổn thương cấp tính nổi bật."

    return " ".join(res)


def compute_report_similarity(ai_text: str, gt_text: str) -> dict:
    """
    Tính chỉ số tương đồng giữa báo cáo AI sinh ra và báo cáo tham chiếu của bác sĩ.
    Bao gồm ROUGE-L F1 (%) và Tỷ lệ trùng từ khóa bệnh lý y khoa.
    """
    def tokenize(text):
        clean = re.sub(r"[^\w\s]", " ", text.lower())
        return [w for w in clean.split() if len(w) > 1 and w not in ("the", "is", "are", "and", "or", "in", "to", "of", "with", "there", "within")]

    ai_words = tokenize(ai_text)
    gt_words = tokenize(gt_text)

    if not ai_words or not gt_words:
        return {"rouge_l": 0.0, "keyword_overlap": 0.0, "matched_keywords": []}

    # Tính LCS (Longest Common Subsequence) cho ROUGE-L
    m, n = len(ai_words), len(gt_words)
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            tmp = dp[j]
            if ai_words[i - 1] == gt_words[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    lcs_len = dp[n]

    precision = lcs_len / m
    recall = lcs_len / n
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Đối sánh từ khóa y khoa chính
    clinical_keywords = [
        "normal", "cardiomegaly", "effusion", "pneumothorax", "edema", "consolidation",
        "atelectasis", "infiltrate", "opacity", "nodule", "granuloma", "emphysema",
        "fibrosis", "sternotomy", "wire", "spondylosis", "degenerative", "fracture",
        "heart", "lungs", "mediastinum", "airspace", "disease", "acute"
    ]
    ai_kws = {w for w in ai_words if any(k in w for k in clinical_keywords)}
    gt_kws = {w for w in gt_words if any(k in w for k in clinical_keywords)}
    matched = sorted(list(ai_kws & gt_kws))
    kw_overlap = len(matched) / max(len(gt_kws), 1)

    return {
        "rouge_l": round(f1 * 100, 2),
        "keyword_overlap": round(kw_overlap * 100, 2),
        "matched_keywords": matched,
    }


def get_display_label(label_raw: str) -> str:
    """Ánh xạ tên nhãn tiếng Anh sang nhãn song ngữ Anh - Việt."""
    low = label_raw.strip().lower()
    if label_raw in CHEXPERT_TRANSLATIONS:
        return CHEXPERT_TRANSLATIONS[label_raw]
    if low in MESH_TRANSLATIONS:
        return MESH_TRANSLATIONS[low]
    for k, v in MESH_TRANSLATIONS.items():
        if k in low:
            return v
    return label_raw.capitalize()


def save_doctor_review(csv_path: str, review_record: dict) -> None:
    """Lưu đánh giá và kết luận của bác sĩ vào file CSV với chuẩn utf-8-sig."""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    fieldnames = [
        "timestamp",
        "case_id",
        "model_used",
        "doctor_name",
        "ai_evaluation",
        "doctor_notes",
        "verified_diseases",
        "final_clinical_conclusion",
        "ai_report_vi",
        "ai_report_en",
    ]
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(review_record)


def load_doctor_reviews(csv_path: str) -> pd.DataFrame:
    """Nạp danh sách các ca bệnh đã được bác sĩ thẩm định từ file CSV."""
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    try:
        return pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception:
        try:
            return pd.read_csv(csv_path, encoding="utf-8")
        except Exception:
            return pd.DataFrame()


def export_clinical_certificate(case_id: str, model_name: str, doctor_name: str,
                                ai_eval: str, verified_diseases: list,
                                final_conclusion: str, notes: str,
                                ai_report_vi: str, ai_report_en: str) -> str:
    """Tạo văn bản phiếu kết quả chẩn đoán hình ảnh chuyên nghiệp."""
    now_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    diseases_str = ", ".join(verified_diseases) if verified_diseases else "Chưa ghi nhận bệnh lý nổi bật"
    cert = f"""================================================================================
                    PHIẾU KẾT QUẢ CHẨN ĐOÁN HÌNH ẢNH X-QUANG NGỰC
                        (Hệ thống AI Hỗ trợ Lâm sàng IEEE TMI 2022)
================================================================================
- Mã hồ sơ ca bệnh : {case_id}
- Thời gian thực hiện: {now_str}
- Mô hình AI hỗ trợ : {model_name}
- Bác sĩ thẩm định  : {doctor_name}
- Đánh giá của BS   : {ai_eval}
--------------------------------------------------------------------------------
1. DANH MỤC BỆNH LÝ XÁC NHẬN BỞI BÁC SĨ:
   {diseases_str}

2. CHÚ THÍCH LÂM SÀNG & VỊ TRÍ TỔN THƯƠNG:
   {notes if notes else "(Không có chú thích bổ sung)"}

3. KẾT LUẬN CHẨN ĐOÁN CUỐI CÙNG:
   {final_conclusion}

--------------------------------------------------------------------------------
4. THAM CHIẾU GỢI Ý TỪ MÔ HÌNH AI:
   [Bản dịch tiếng Việt do AI sinh ra]:
   {ai_report_vi}

   [Nguyên văn tiếng Anh gốc (Multicriteria Transformer)]:
   {ai_report_en}

================================================================================
                     CHỮ KÝ BÁC SĨ CHẨN ĐOÁN HÌNH ẢNH
                               (Đã duyệt)
                               {doctor_name}
================================================================================
"""
    return cert
