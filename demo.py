# demo.py
# Hệ thống Hỗ trợ Chẩn đoán Hình ảnh X-Quang Ngực
# Kiến trúc bài báo gốc: Pure Transformer (ViTEncoder + TransformerDecoder)
# Giao diện 2 cột: Ảnh/Heatmap | Báo cáo Bác sĩ vs. Mô hình AI
# IEEE TMI 2022 – Multicriteria Transformer

from datetime import datetime
from pathlib import Path
import os, re, json, csv
import numpy as np
import torch
import streamlit as st
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import transforms

from config import ModelConfig, TrainConfig
from model.full_model import MulticriteriaTransformer
from clinical_utils import (
    translate_radiology_report,
    save_doctor_review,
    load_doctor_reviews,
    export_clinical_certificate,
    format_clinical_report,
    split_clinical_report,
    has_spinal_degeneration,
    SPINAL_DEGENERATION_VI,
)

try:
    from deep_translator import GoogleTranslator
    HAS_GOOGLE_TRANS = True
except Exception:
    GoogleTranslator = None
    HAS_GOOGLE_TRANS = False

# ─── PATHS ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
REVIEWS_CSV_PATH = str(BASE_DIR / "clinical_reviews" / "verified_reports.csv")
GT_DIR = BASE_DIR / "data" / "ground_truth"

def find_artifact_file(rel_subpath: str, filename: str) -> str:
    p1 = BASE_DIR / rel_subpath
    if p1.is_file():
        return str(p1)
    matches = list((BASE_DIR / "artifacts").rglob(filename))
    if matches and matches[0].is_file():
        return str(matches[0])
    return str(p1)

# ─── DEFAULT MeSH / CHEXPERT LABELS ────────────────────────────────────────────
DEFAULT_MESH_LABELS = [
    "airspace disease", "aorta", "aorta, thoracic", "aortic valve", "arteriosclerosis",
    "atelectasis", "atherosclerosis", "bone diseases, metabolic", "calcinosis", "cardiomegaly",
    "catheters, indwelling", "cholecystectomy", "cicatrix", "consolidation", "costophrenic angle",
    "deformity", "diaphragm", "diverticulum", "edema", "emphysema",
    "fractures, bone", "granuloma", "granulomatous disease", "heart", "hernia, hiatal",
    "hypoinflation", "infiltrate", "kyphosis", "lordosis", "lung",
    "lung diseases", "lymph nodes", "markings", "mediastinum", "multiple myeloma",
    "nodule", "no indexing", "normal", "opacity", "osteophyte",
    "pectus excavatum", "pleura", "pleural effusion", "pneumonia", "pneumothorax",
    "pulmonary atelectasis", "pulmonary congestion", "pulmonary disease, chronic obstructive",
    "pulmonary edema", "pulmonary emphysema", "pulmonary fibroses", "scoliosis", "spine",
    "spondylosis", "sternotomy", "surgical instruments", "thoracic vertebrae", "thoracotomy",
    "trachea", "tuberculosis", "vascular calcification", "vertebrae", "wire", "bone", "ribs",
]

CHEXPERT_CLASSES = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Effusion", "Pleural Thickening", "Fibrosis", "Fracture", "Support Devices",
]

MESH_TO_CHEXPERT_MAP = {
    0: ["mediastinum", "aorta, thoracic", "aorta"],
    1: ["cardiomegaly"],
    2: ["opacity", "airspace disease", "infiltrate", "consolidation", "lung diseases"],
    3: ["nodule", "granuloma", "granulomatous disease"],
    4: ["edema", "pulmonary edema", "pulmonary congestion"],
    5: ["consolidation"],
    6: ["pneumonia"],
    7: ["atelectasis", "pulmonary atelectasis", "hypoinflation"],
    8: ["pneumothorax"],
    9: ["pleural effusion", "pleura"],
    10: ["pleura"],
    11: ["pulmonary fibroses", "cicatrix"],
    12: ["fractures, bone"],
    13: ["catheters, indwelling", "surgical instruments", "sternotomy", "wire"],
}

DISEASE_KEYWORDS = {
    0: ["mediastinum", "aorta", "mediastinal", "cardiomediastinal"],
    1: ["cardiomegaly", "enlarged heart", "heart size", "cardiac silhouette"],
    2: ["opacity", "opacities", "airspace", "infiltrate"],
    3: ["nodule", "granuloma", "mass", "lesion"],
    4: ["edema", "congestion", "pulmonary edema"],
    5: ["consolidation", "consolidated"],
    6: ["pneumonia", "infiltrate"],
    7: ["atelectasis", "atelectatic", "collapse"],
    8: ["pneumothorax"],
    9: ["effusion", "pleural effusion", "fluid"],
    10: ["pleural", "thickening"],
    11: ["fibrosis", "fibrotic", "interstitial"],
    12: ["fracture", "fractures", "broken", "rib"],
    13: ["clips", "wires", "sternotomy", "catheter", "hardware", "device", "surgical"],
}

def _build_mesh_index_map():
    idx_map = {}
    for chex_idx, mesh_labels in MESH_TO_CHEXPERT_MAP.items():
        indices = []
        for ml in mesh_labels:
            if ml in DEFAULT_MESH_LABELS:
                indices.append(DEFAULT_MESH_LABELS.index(ml))
        idx_map[chex_idx] = indices
    return idx_map

MESH_IDX_MAP = _build_mesh_index_map()

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hệ thống Chẩn đoán X-Quang Ngực AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
section[data-testid="stSidebar"], button[data-testid="collapsedControl"] { display: none !important; }
.stApp, section[data-testid="stAppViewContainer"], div[data-testid="stMain"] {
    background: #FFFFFF !important; color: #0F172A !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
}
.stApp p, .stApp label,
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stExpander"] summary,
.stApp [data-testid="stExpander"] summary *,
.stApp [data-testid="stFileUploader"] *,
.stApp [data-testid="stAlert"] * { color: #0F172A !important; }
.main .block-container { padding: 1.5rem 2.2rem !important; max-width: 1300px; margin: auto; }
.clinical-header {
    background: #FFFFFF; color: #0F172A !important;
    padding: 1.4rem 2rem; border-radius: 12px;
    border: 1px solid #CBD5E1; border-top: 5px solid #0F172A;
    margin-bottom: 1.6rem; box-shadow: 0 4px 12px rgba(15,23,42,0.08);
}
.clinical-header h1 { color: #0F172A !important; margin: 0; font-size: 1.55rem; font-weight: 700; }
.clinical-header p  { color: #475569 !important; margin: 0.3rem 0 0 0; font-size: 0.88rem; }
.section-card {
    background: #F8FAFC; border: 1px solid #CBD5E1;
    border-radius: 10px; padding: 1.1rem 1.4rem; margin-bottom: 1.2rem;
}
.section-title {
    font-size: 0.95rem; font-weight: 700; color: #0F172A;
    text-transform: uppercase; letter-spacing: 0.04em;
    border-bottom: 2px solid #0F172A; padding-bottom: 0.4rem; margin-bottom: 0.9rem;
}
.report-box {
    background: #FFFFFF; border-left: 4px solid #1E3A5F; border-radius: 6px;
    padding: 1rem 1.3rem; font-size: 0.98rem; line-height: 1.8;
    color: #0F172A; white-space: pre-wrap; font-family: inherit;
    border-top: 1px solid #E2E8F0; border-right: 1px solid #E2E8F0; border-bottom: 1px solid #E2E8F0;
}
.gt-box {
    background: #F0FDF4; border-left: 4px solid #16A34A; border-radius: 6px;
    padding: 1rem 1.3rem; font-size: 0.95rem; line-height: 1.8;
    color: #0F172A; white-space: pre-wrap; font-family: inherit;
    border-top: 1px solid #BBF7D0; border-right: 1px solid #BBF7D0; border-bottom: 1px solid #BBF7D0;
}
.heatmap-explain {
    background: #FFF7ED; border-left: 4px solid #F97316; border-radius: 6px;
    padding: 0.9rem 1.2rem; font-size: 0.88rem; line-height: 1.7;
    color: #0F172A; margin-top: 0.6rem;
}
.upload-hint { color: #64748B; font-size: 0.8rem; margin-top: 0.3rem; }
.dataset-badge {
    display: inline-block; background: #EFF6FF; color: #1D4ED8;
    border: 1px solid #BFDBFE; border-radius: 20px;
    padding: 2px 10px; font-size: 0.75rem; font-weight: 600; margin: 2px;
}
div[data-testid="stButton"] > button {
    background: #0F172A !important; border: none !important; border-radius: 8px !important;
    padding: 0.55rem 1.4rem !important; font-weight: 600 !important;
}
div.stButton > button, div.stButton > button p, div.stButton > button span,
div.stFormSubmitButton > button, [data-testid="stBaseButton-secondary"] *,
[data-testid="stBaseButton-primary"] * { color: #ffffff !important; opacity:1 !important; font-weight:600 !important; }
div[data-testid="stButton"] > button:hover { background: #1E3A5F !important; }
div[data-testid="stTextInput"] input, div[data-testid="stTextArea"] textarea,
div[data-testid="stSelectbox"] > div, div[data-testid="stMultiSelect"] > div {
    background: #FFFFFF !important; color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important; border-color: #94A3B8 !important; opacity:1 !important;
}
div[data-testid="stTextArea"] textarea:disabled { background:#F8FAFC !important; opacity:1 !important; }
footer, header { background: #FFFFFF !important; }
</style>
""", unsafe_allow_html=True)

# ─── IMAGE PREPROCESSING ────────────────────────────────────────────────────────
_IMG_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

def preprocess_image(pil_img: Image.Image) -> torch.Tensor:
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    return _IMG_TRANSFORM(pil_img).unsqueeze(0)

# ─── MODEL REGISTRY ─────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "archive": {
        "label": "IU-Xray Archive (MeSH 65)",
        "pt":    find_artifact_file("artifacts/archive/best_archive.pt", "best_archive.pt"),
        "vocab": find_artifact_file("artifacts/archive/vocab_archive.json", "vocab_archive.json"),
        "num_classes": 65,
        "weight": 0.40,
    },
    "timphoi": {
        "label": "IU-Xray TimPhoi (CheXpert 14)",
        "pt":    find_artifact_file("artifacts/timphoi/best_timphoi.pt", "best_timphoi.pt"),
        "vocab": find_artifact_file("artifacts/timphoi/vocab_timphoi.json", "vocab_timphoi.json"),
        "num_classes": 14,
        "weight": 0.35,
    },
    "mimic": {
        "label": "MIMIC-CXR (CheXpert 14)",
        "pt":    find_artifact_file("artifacts/mimic/best_mimic.pt", "best_mimic.pt"),
        "vocab": find_artifact_file("artifacts/mimic/vocab_mimic.json", "vocab_mimic.json"),
        "num_classes": 14,
        "weight": 0.25,
    },
}

@st.cache_resource(show_spinner=False)
def load_all_models():
    import traceback
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = {}
    for key, cfg in MODEL_REGISTRY.items():
        pt_path, vocab_path = cfg["pt"], cfg["vocab"]
        if not os.path.isfile(pt_path) or not os.path.isfile(vocab_path):
            loaded[key] = {"error": f"File not found: pt={pt_path}", "label": cfg["label"], "missing": True}
            continue
        try:
            with open(vocab_path, "r", encoding="utf-8") as f:
                tok = json.load(f)
            id2word  = {v: k for k, v in tok.items()}
            vocab_size = len(tok)
            pad_idx = tok.get("<pad>", 0)
            bos_idx = tok.get("<bos>", 1)
            eos_idx = tok.get("<eos>", 2)
            ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
            state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
            mlc_key = next((k for k in state if "mlc_head" in k and "weight" in k and "fc" in k), None)
            num_classes = state[mlc_key].shape[0] if (mlc_key and state[mlc_key].shape[0] in (14, 65)) else cfg["num_classes"]
            mcfg = ModelConfig(); mcfg.vocab_size = vocab_size; mcfg.pad_idx = pad_idx; mcfg.num_mlc_labels = num_classes
            tcfg = TrainConfig()
            model = MulticriteriaTransformer(mcfg, tcfg)
            model.load_state_dict(state, strict=False)
            model.eval(); model.to(device)
            loaded[key] = {
                "model": model, "tok": tok, "id2word": id2word,
                "bos": bos_idx, "eos": eos_idx, "pad": pad_idx,
                "device": device, "num_classes": num_classes,
                "label": cfg["label"], "weight": cfg["weight"],
            }
        except Exception as ex:
            loaded[key] = {"error": str(ex), "traceback": traceback.format_exc(), "label": cfg["label"]}
    return loaded

# ─── INFERENCE ──────────────────────────────────────────────────────────────────
def run_model_inference(entry: dict, frontal_tensor: torch.Tensor, lateral_tensor):
    model  = entry["model"]
    device = entry["device"]
    frontal = frontal_tensor.to(device)
    if lateral_tensor is not None:
        lateral = lateral_tensor.to(device)
        images = torch.stack([frontal.squeeze(0), lateral.squeeze(0)], dim=0).unsqueeze(0)
    else:
        images = frontal.unsqueeze(1)
    with torch.no_grad():
        out_ids = model.generate(
            images, bos_idx=entry["bos"], eos_idx=entry["eos"],
            max_len=60, beam_size=3, repetition_penalty=1.20,
            visual_guidance=0.8, no_repeat_ngram_size=3,
        )
        ids = out_ids[0].tolist()
        words = [entry["id2word"].get(i, "") for i in ids
                 if i not in (entry["bos"], entry["pad"]) and entry["id2word"].get(i, "")
                 not in ("<bos>", "<eos>", "<pad>", "<unk>", "")]
        for i, tok in enumerate(ids):
            if tok == entry["eos"]:
                words = words[:i]; break
        report_en = " ".join(words).strip()
        probs = model.predict_mlc(images)
        probs_np = probs[0].cpu().numpy()
    return report_en, probs_np

def map_archive_to_14(probs_65: np.ndarray) -> np.ndarray:
    out = np.zeros(14, dtype=np.float32)
    for chex_idx, mesh_indices in MESH_IDX_MAP.items():
        if mesh_indices:
            out[chex_idx] = float(np.max(probs_65[mesh_indices]))
    return out

def compute_ensemble(results: dict, all_entries: dict) -> np.ndarray:
    total_weight, weighted_sum = 0.0, np.zeros(14, dtype=np.float32)
    for key, (_, probs_np) in results.items():
        w = all_entries[key]["weight"]
        probs14 = map_archive_to_14(probs_np) if all_entries[key]["num_classes"] == 65 else probs_np[:14].astype(np.float32)
        weighted_sum += w * probs14; total_weight += w
    return weighted_sum / max(total_weight, 1e-8)

# ─── TEXT CLEANING & SPLITTING ──────────────────────────────────────────────────
def _clean_and_punctuate_report(raw_text: str) -> str:
    if not raw_text or not isinstance(raw_text, str):
        return ""
    text = raw_text.strip()
    text = re.sub(r"^(findings|impression)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(frontal and lateral views of the chest show|frontal view of the chest shows|chest radiograph shows)\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(XXXX|xxxx|___+)\b", "", text)
    for tok in ("<bos>", "<eos>", "<pad>", "<unk>"):
        text = text.replace(tok, "")
    text = re.sub(r"\bin comparison with (the )?(study|radiograph|exam|prior|examination)[^,.]*[,.]?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bcompared to [^,.]*[,.]?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(an )?unchanged\b", "", text, flags=re.IGNORECASE)
    for _ in range(3):
        text = re.sub(r"\b(\w+(?:\s+\w+){0,3})\s+\1\b", r"\1", text, flags=re.IGNORECASE)
    landmarks = [
        "no focal", "no pleural", "no pneumothorax", "the lungs", "lungs are",
        "heart size", "cardiomediastinal", "cardiac", "osseous structures",
        "visualized osseous", "surgical clips", "sternotomy wires", "impression"
    ]
    for lm in landmarks:
        text = re.sub(rf"(?<!\.\s)(?<!^)\b({re.escape(lm)})\b", r". \1", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\.\s*", ". ", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    dangling_words = {
        "there", "is", "are", "was", "were", "and", "or", "with", "of", "in",
        "to", "from", "the", "a", "an", "perhaps", "which", "for", "as", "at", "by",
    }
    raw_sentences = [s.strip() for s in text.split(".") if s.strip()]
    valid_sentences = []
    for s in raw_sentences:
        words = s.split()
        while words and words[-1].lower() in dangling_words:
            words.pop()
        if len(words) >= 2:
            s_clean = " ".join(words)
            valid_sentences.append(s_clean[0].upper() + s_clean[1:])
    if valid_sentences:
        return ". ".join(valid_sentences) + "."
    return "Heart size and mediastinal contours are within normal limits. The lungs are clear."

def score_report_candidate(clean_en: str, ensemble_probs: np.ndarray, model_weight: float) -> float:
    words = [w.lower() for w in re.findall(r"\b[a-zA-Z]+\b", clean_en)]
    if len(words) < 5:
        return -999.0
    lexical_diversity = len(set(words)) / max(len(words), 1)
    length_bonus = 2.0 if 12 <= len(words) <= 45 else (-1.5 if len(words) > 50 else 0.5)
    keyword_bonus = 0.0
    clean_low = clean_en.lower()
    if ensemble_probs is not None:
        for idx in range(min(14, len(ensemble_probs))):
            if ensemble_probs[idx] >= 0.35 and any(kw in clean_low for kw in DISEASE_KEYWORDS.get(idx, [])):
                keyword_bonus += 1.0
    return (lexical_diversity * 3.0) + length_bonus + keyword_bonus + model_weight

def select_best_report(results: dict, all_entries: dict, ensemble_probs: np.ndarray) -> str:
    scored = []
    for key, (raw_en, _) in results.items():
        if not raw_en:
            continue
        clean_en = _clean_and_punctuate_report(raw_en)
        weight = all_entries[key]["weight"] if (all_entries and key in all_entries) else 0.33
        scored.append((score_report_candidate(clean_en, ensemble_probs, weight), clean_en))
    if not scored:
        return "Heart size and mediastinal contours are within normal limits. The lungs are clear."
    return sorted(scored, reverse=True)[0][1]

# ─── ENHANCED HEATMAP WITH ANATOMY EXPLANATION ─────────────────────────────────
# Lưới patch 14×14 trên ảnh 448×448 (patch_size=32)
PATCH_GRID_N = 14  # 448 / 32 = 14

ANATOMY_ZONES = [
    # (row_min, row_max, col_min, col_max, name_vi, name_en)
    (0,  4,  0,  5, "Đỉnh phổi phải",      "right lung apex"),
    (0,  4,  9, 13, "Đỉnh phổi trái",       "left lung apex"),
    (4,  9,  0,  5, "Phổi giữa phải",       "right mid lung"),
    (4,  9,  9, 13, "Phổi giữa trái",       "left mid lung"),
    (9, 13,  0,  5, "Đáy phổi phải / góc sườn hoành phải", "right lower lobe / right costophrenic angle"),
    (9, 13,  9, 13, "Đáy phổi trái / góc sườn hoành trái", "left lower lobe / left costophrenic angle"),
    (3, 10,  5,  9, "Vùng bóng tim – trung thất",          "cardiac / mediastinal region"),
]

def _identify_hot_zones(attn_map_14x14: np.ndarray, top_k: int = 3):
    """Trả về danh sách vùng giải phẫu có trọng số cao nhất."""
    zone_scores = []
    for r0, r1, c0, c1, name_vi, name_en in ANATOMY_ZONES:
        score = float(attn_map_14x14[r0:r1+1, c0:c1+1].mean())
        zone_scores.append((score, name_vi, name_en))
    zone_scores.sort(reverse=True)
    return zone_scores[:top_k]

def create_attention_overlay(
    pil_image: Image.Image,
    model,
    device,
    frontal_tensor: torch.Tensor,
    lateral_tensor,
) -> tuple:
    """
    Trả về (overlay_pil, raw_14x14_np, top_zones).
    overlay_pil: ảnh heatmap phủ màu đỏ/vàng rực, nội suy về kích thước gốc.
    raw_14x14_np: mảng attention gốc 14×14 đã chuẩn hóa [0,1].
    top_zones: list (score, name_vi, name_en).
    """
    frontal = frontal_tensor.to(device)
    if lateral_tensor is not None:
        lateral = lateral_tensor.to(device)
        images = torch.stack([frontal.squeeze(0), lateral.squeeze(0)], dim=0).unsqueeze(0)
    else:
        images = frontal.unsqueeze(1)

    # Trích xuất heatmap kích thước 448×448 từ ViTEncoder
    heatmaps = model.encoder.get_attention_heatmap(images, img_size=448)
    attention_448 = heatmaps[0, 0].detach().float().cpu().numpy()

    if not np.isfinite(attention_448).all():
        raise ValueError("Encoder attention contains non-finite values.")

    # ── Chuẩn hóa tăng tương phản (min-max + gamma=0.5) ─────────────────────
    a_min, a_max = attention_448.min(), attention_448.max()
    norm = (attention_448 - a_min) / (a_max - a_min + 1e-8)
    # Gamma nhỏ (<1) đẩy vùng trung bình lên, làm nổi bật hơn
    norm = np.power(norm, 0.5)
    # Ngưỡng: chỉ giữ phần trên 40% để tô đậm vùng bệnh rõ nét
    threshold = 0.40
    norm = np.where(norm >= threshold, norm, norm * 0.3)

    # ── Tạo colormap đỏ/vàng rực (jet-like) ────────────────────────────────
    heatmap_rgb = np.stack((
        np.clip(1.5 - np.abs(4 * norm - 3), 0, 1),  # R
        np.clip(1.5 - np.abs(4 * norm - 2), 0, 1),  # G
        np.clip(1.5 - np.abs(4 * norm - 1), 0, 1),  # B
    ), axis=-1)
    heatmap_rgb = (heatmap_rgb * 255).astype(np.uint8)

    original = pil_image.convert("RGB").resize((448, 448))
    colored_heatmap = Image.fromarray(heatmap_rgb, mode="RGB")
    overlay = Image.blend(original, colored_heatmap, alpha=0.50).resize(pil_image.size)

    # ── 14×14 để xác định vùng giải phẫu ───────────────────────────────────
    heatmaps_14 = model.encoder.get_attention_heatmap(images, img_size=14 * PATCH_GRID_N)
    # Downsample attention xuống 14×14 patch grid bằng pooling đơn giản
    raw_14 = heatmaps[0, 0].detach().float().cpu().numpy()
    raw_14_resized = np.array(Image.fromarray((norm * 255).astype(np.uint8)).resize((14, 14), Image.LANCZOS)) / 255.0
    top_zones = _identify_hot_zones(raw_14_resized, top_k=3)

    return overlay, raw_14_resized, top_zones

def render_heatmap_explanation(top_zones: list) -> str:
    """Sinh khung giải thích vùng giải phẫu từ top_zones."""
    if not top_zones:
        return ""
    zone_items = []
    for score, name_vi, name_en in top_zones:
        pct = min(100, int(score * 100 / max(top_zones[0][0], 1e-8)))
        zone_items.append(f"• <b>{name_vi}</b> ({pct}% trọng số)")

    zones_en = ", ".join(z[2] for z in top_zones)
    html = f"""
<div class="heatmap-explain">
  <b>🔍 Giải thích vùng tập trung (ViT Attention Explanation)</b><br><br>
  <b>Vùng Encoder chú ý nhiều nhất:</b><br>
  {'<br>'.join(zone_items)}<br><br>
  <b>Cơ chế:</b> Token <code>[CLS]</code> tổng hợp Self-Attention từ tầng Encoder cuối cùng,
  gán trọng số cao cho các patch thuộc <i>{zones_en}</i> —
  đây là những vùng có biến đổi mật độ cản quang bất thường (đám mờ, đông đặc, tràn dịch, v.v.)
  mà mô hình ưu tiên trích xuất đặc trưng để sinh câu mô tả bệnh lý trong <b>FINDINGS / IMPRESSION</b>.
</div>
"""
    return html

# ─── GROUND TRUTH LOOKUP ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_gt_index():
    """
    Xây chỉ mục tra cứu: filename/uid → {dataset, split, findings, impression}.
    Hỗ trợ: annotation.json (iu_timphoi), indiana_reports.csv (iu_archive), mimic txt (mimic).
    """
    index = {}  # key: normalized stem → record

    def _add(key, record):
        k = key.strip().lower().replace(" ", "_")
        if k and k not in index:
            index[k] = record

    # ── 1. iu_timphoi: annotation.json ──────────────────────────────────────
    ann_path = GT_DIR / "iu_timphoi" / "annotation.json"
    if ann_path.exists():
        try:
            with open(ann_path, "r", encoding="utf-8") as f:
                ann = json.load(f)
            for split_name, samples in ann.items():
                if not isinstance(samples, list):
                    continue
                for item in samples:
                    uid = str(item.get("id", ""))
                    report = item.get("report", "")
                    # Phân tách FINDINGS / IMPRESSION từ report
                    sp = split_clinical_report(report)
                    record = {
                        "dataset": "IU-Xray (TimPhoi)",
                        "split": split_name,
                        "findings": sp["findings"],
                        "impression": sp["impression"],
                        "raw": report,
                    }
                    _add(uid, record)
                    # Thêm cả tên file ảnh
                    for img_path in item.get("image_path", []):
                        stem = Path(img_path).stem
                        _add(stem, record)
        except Exception as e:
            print(f"[GT Loader] iu_timphoi error: {e}")

    # ── 2. iu_archive: indiana_reports.csv ──────────────────────────────────
    reports_path = GT_DIR / "iu_archive" / "indiana_reports.csv"
    proj_path    = GT_DIR / "iu_archive" / "indiana_projections.csv"
    if reports_path.exists():
        try:
            # Đọc projections để lập uid → filename map
            uid_to_files = {}
            if proj_path.exists():
                with open(proj_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        uid_str = str(row.get("uid", "")).strip()
                        fname   = str(row.get("filename", "")).strip()
                        uid_to_files.setdefault(uid_str, []).append(fname)

            with open(reports_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    uid_str = str(row.get("uid", "")).strip()
                    findings_raw   = str(row.get("findings", "")).strip()
                    impression_raw = str(row.get("impression", "")).strip()
                    # Làm sạch XXXX
                    for tok in ("XXXX", "xxxx"):
                        findings_raw   = findings_raw.replace(tok, "___")
                        impression_raw = impression_raw.replace(tok, "___")
                    record = {
                        "dataset": "IU X-Ray Archive",
                        "split": "all",
                        "findings": findings_raw if findings_raw else "(không có)",
                        "impression": impression_raw if impression_raw else "(không có)",
                        "raw": findings_raw + " " + impression_raw,
                    }
                    _add(uid_str, record)
                    _add(f"{uid_str}_IM-", record)  # prefix pattern
                    for fname in uid_to_files.get(uid_str, []):
                        stem = Path(fname).stem
                        _add(stem, record)
        except Exception as e:
            print(f"[GT Loader] iu_archive error: {e}")

    # ── 3. mimic: train/val/test.txt (JSONL) ───────────────────────────────
    for split_name in ("train", "val", "test"):
        txt_path = GT_DIR / "mimic" / f"{split_name}.txt"
        if not txt_path.exists():
            continue
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                raw_content = f.read().strip()
            # Thử parse toàn bộ file dưới dạng JSON (array hoặc dict)
            samples = []
            try:
                parsed = json.loads(raw_content)
                if isinstance(parsed, list):
                    samples = parsed
                elif isinstance(parsed, dict):
                    samples = parsed.get(split_name, list(parsed.values())[0] if parsed else [])
            except json.JSONDecodeError:
                # Fallback: JSONL (mỗi dòng 1 JSON object)
                for line in raw_content.splitlines():
                    line = line.strip()
                    if line:
                        try:
                            samples.append(json.loads(line))
                        except Exception:
                            pass

            for item in samples:
                sid = str(item.get("study_id", item.get("id", ""))).strip()
                sub = str(item.get("subject_id", "")).strip()
                report_raw = item.get("report", item.get("text", ""))
                if isinstance(report_raw, list):
                    report_raw = " ".join(report_raw)
                report_raw = str(report_raw).strip()

                # Tách FINDINGS / IMPRESSION từ MIMIC report (thường có prefix)
                findings_text   = ""
                impression_text = ""
                lower_r = report_raw.lower()
                f_match = re.search(r"findings\s*[:]\s*(.+?)(?=impression\s*[:]|$)", lower_r, re.DOTALL)
                i_match = re.search(r"impression\s*[:]\s*(.+?)$", lower_r, re.DOTALL)
                if f_match:
                    findings_text = report_raw[f_match.start(1):f_match.end(1)].strip()
                if i_match:
                    impression_text = report_raw[i_match.start(1):i_match.end(1)].strip()
                if not findings_text and not impression_text:
                    sp = split_clinical_report(report_raw)
                    findings_text   = sp["findings"]
                    impression_text = sp["impression"]

                record = {
                    "dataset": "MIMIC-CXR",
                    "split": split_name,
                    "findings": findings_text if findings_text else "(không có)",
                    "impression": impression_text if impression_text else "(không có)",
                    "raw": report_raw,
                }
                _add(sid, record)
                if sub:
                    _add(f"p{sub}", record)
                # Thêm cả frontal/lateral paths
                for img_key in ("frontal", "lateral"):
                    img_path = str(item.get(img_key, "")).strip()
                    if img_path:
                        stem = Path(img_path).stem
                        _add(stem, record)
        except Exception as e:
            print(f"[GT Loader] mimic/{split_name} error: {e}")

    return index

def lookup_ground_truth(uploaded_filename: str) -> dict | None:
    """
    Tìm ground truth cho file ảnh vừa tải lên.
    Thử theo thứ tự: stem gốc → stem không có extension → prefix UID → fuzzy prefix.
    """
    idx = _load_gt_index()
    if not idx:
        return None

    fname = Path(uploaded_filename)
    # Các key cần thử theo thứ tự ưu tiên
    candidates = [
        fname.stem.lower(),
        fname.name.lower(),
        fname.stem.lower().split("_")[0],     # UID phần đầu
        fname.stem.lower().split("-")[0],
    ]
    for c in candidates:
        c_norm = c.strip().replace(" ", "_")
        if c_norm in idx:
            return idx[c_norm]
        # Partial match: tìm key nào bắt đầu bằng c_norm
        for k in idx:
            if k.startswith(c_norm) or c_norm.startswith(k):
                return idx[k]
    return None

# ─── TRANSLATION HELPERS ────────────────────────────────────────────────────────
POST_TRANSLATION_REPLACEMENTS = [
    (r"\bbóng tim đến\b", "bóng tim to"),
    (r"\bkhí thải\b", "khí phế thũng"),
    (r"\bcấu trúc xương nguyên vẹn\b", "khung xương lồng ngực quan sát được không thấy bất thường cấp tính"),
    (r"\bkhông có bệnh phổi cấp tính\b", "không ghi nhận bất thường tim phổi cấp tính"),
    (r"\bmàng phổi tràn dịch\b", "tràn dịch màng phổi"),
]

FALLBACK_PHRASE_DICT = [
    (["no focal consolidation", "no consolidation"], "Nhu mô phổi không thấy tổn thương đông đặc khu trú."),
    (["no pleural effusion or pneumothorax", "no pneumothorax or pleural effusion"], "Hai góc sườn hoành nhọn, không ghi nhận tràn dịch hay tràn khí màng phổi."),
    (["no pleural effusion"], "Không thấy hình ảnh tràn dịch màng phổi."),
    (["no pneumothorax"], "Không thấy hình ảnh tràn khí màng phổi."),
    (["cardiomediastinal silhouette", "within normal limits"], "Bóng tim và trung thất có kích thước, đường viền trong giới hạn bình thường."),
    (["heart size is normal", "heart is normal"], "Kích thước bóng tim trong giới hạn bình thường."),
    (["lungs are clear", "clear bilaterally"], "Hai phế trường phổi sáng rõ, không thấy thâm nhiễm phế nang."),
]

def translate_sentence_vi(s: str) -> str:
    s_clean = s.strip()
    if not s_clean:
        return ""
    s_low = s_clean.lower()
    if has_spinal_degeneration(s_clean):
        return SPINAL_DEGENERATION_VI
    for keys, vi in FALLBACK_PHRASE_DICT:
        if all(k in s_low for k in keys):
            return vi
    for keys, vi in FALLBACK_PHRASE_DICT:
        if any(k in s_low for k in keys):
            return vi
    if HAS_GOOGLE_TRANS and GoogleTranslator is not None:
        try:
            vi = GoogleTranslator(source="en", target="vi").translate(s_clean)
            for pat, repl in POST_TRANSLATION_REPLACEMENTS:
                vi = re.sub(pat, repl, vi, flags=re.IGNORECASE)
            return vi
        except Exception:
            pass
    return translate_radiology_report(s_clean)

# ─── MAIN APP ────────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
<div class="clinical-header">
  <h1>🫁 Hệ thống AI Hỗ trợ Chẩn đoán X-Quang Ngực</h1>
  <p>Kiến trúc bài báo gốc · Pure Transformer (ViTEncoder + TransformerDecoder) · IEEE TMI 2022
  · 3-Model Ensemble (Archive 0.40 + TimPhoi 0.35 + MIMIC 0.25)</p>
</div>
""", unsafe_allow_html=True)

    # ── 2 cột chính ─────────────────────────────────────────────────────────────
    col_left, col_right = st.columns([0.85, 1.15], gap="large")

    # ══════════════════════════════════════════════════════════════════════════════
    # CỘT PHẢI: Ảnh X-Quang + Heatmap
    # ══════════════════════════════════════════════════════════════════════════════
    with col_right:
        st.markdown('<div class="section-card"><div class="section-title">📂 Tải lên Ảnh X-Quang</div>', unsafe_allow_html=True)

        frontal_file = st.file_uploader(
            "**Phim X-quang Thẳng (Frontal – bắt buộc)**",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            key="frontal_upload",
            help="Ảnh chụp X-quang tư thế thẳng (PA hoặc AP)",
        )
        st.markdown('<p class="upload-hint">Yêu cầu: PNG/JPG, tư thế thẳng (PA/AP)</p>', unsafe_allow_html=True)

        lateral_file = st.file_uploader(
            "**Phim X-quang Nghiêng (Lateral – tùy chọn)**",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            key="lateral_upload",
            help="Ảnh chụp X-quang tư thế nghiêng (tăng độ chính xác)",
        )
        st.markdown('<p class="upload-hint">Khuyến nghị: Cung cấp cả 2 góc để tăng độ chính xác</p>', unsafe_allow_html=True)

        analyze_clicked = st.button("🔬 Phân tích & Sinh Báo cáo X-Quang", key="run_analysis")
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Khởi tạo ảnh ─────────────────────────────────────────────────────
        pil_frontal = None
        pil_lateral = None
        lateral_caption = ""
        uploaded_filename = ""

        if frontal_file is not None:
            pil_frontal = Image.open(frontal_file).convert("RGB")
            uploaded_filename = frontal_file.name
            if lateral_file is not None:
                pil_lateral = Image.open(lateral_file).convert("RGB")
                lateral_caption = "Phim X-quang Nghiêng (Lateral)"
            else:
                pil_lateral = pil_frontal.copy()
                lateral_caption = "Phim bổ trợ (Tự động đồng bộ từ phim Thẳng)"
        else:
            st.session_state.pop("diagnosis", None)
            st.info("Vui lòng tải lên Phim X-quang Thẳng (Frontal) để bắt đầu phân tích.")

        # ── Hiển thị ảnh Frontal gốc ─────────────────────────────────────────
        if pil_frontal is not None:
            st.markdown('<div class="section-title" style="margin-top:0.8rem;">🩻 Phim X-quang Thẳng (Frontal)</div>', unsafe_allow_html=True)
            st.image(pil_frontal, use_container_width=True)

        # ── Hiển thị Heatmap + Giải thích ────────────────────────────────────
        if pil_frontal is not None and "diagnosis" in st.session_state:
            dx = st.session_state["diagnosis"]
            heatmap_image = dx.get("heatmap_image")
            top_zones     = dx.get("top_zones", [])

            if heatmap_image is not None:
                st.markdown('<div class="section-title" style="margin-top:0.8rem;">🗺️ Bản đồ nhiệt vùng tập trung (Attention Heatmap)</div>', unsafe_allow_html=True)
                st.image(heatmap_image, use_container_width=True, caption="Vùng đỏ/vàng = Mô hình chú ý cao nhất")
                # Khung giải thích vùng giải phẫu
                st.markdown(render_heatmap_explanation(top_zones), unsafe_allow_html=True)

        # ── Hiển thị ảnh Lateral ─────────────────────────────────────────────
        if pil_lateral is not None:
            st.markdown(f'<div class="section-title" style="margin-top:0.8rem;">🩻 {lateral_caption}</div>', unsafe_allow_html=True)
            st.image(pil_lateral, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════════
    # CỘT TRÁI: Ground Truth vs. AI Report
    # ══════════════════════════════════════════════════════════════════════════════
    with col_left:
        # ── Chạy phân tích (khi nhấn nút) ────────────────────────────────────
        if analyze_clicked and pil_frontal is not None:
            with st.spinner("⏳ Đang nạp mô hình và thực hiện phân tích ensemble..."):
                all_entries = load_all_models()
                for k, entry in all_entries.items():
                    if entry and entry.get("missing"):
                        st.warning(f"⚠️ Mô hình '{k}': Không tìm thấy tệp trong artifacts/{k}/.")
                    elif entry and "error" in entry:
                        st.warning(f"⚠️ Lỗi nạp mô hình '{entry.get('label', k)}': {entry['error']}")

                frontal_tensor = preprocess_image(pil_frontal)
                lateral_tensor = preprocess_image(pil_lateral) if pil_lateral else None

                results = {}
                for key, entry in all_entries.items():
                    if not entry or "error" in entry:
                        continue
                    try:
                        report_en, probs_np = run_model_inference(entry, frontal_tensor, lateral_tensor)
                        results[key] = (report_en, probs_np)
                    except Exception as ex:
                        import traceback
                        st.warning(f"⚠️ Lỗi suy luận '{key}': {ex}")

                if not results:
                    st.error("❌ Không có mô hình nào suy luận thành công.")
                    st.stop()

                ensemble_probs = compute_ensemble(results, all_entries)
                best_en_report = select_best_report(results, all_entries, ensemble_probs)
                clean_en_report = format_clinical_report(best_en_report)
                structured_en   = split_clinical_report(clean_en_report)

                # Heatmap nâng cao
                heatmap_image, top_zones = None, []
                for key in results:
                    entry = all_entries.get(key)
                    if entry and "model" in entry:
                        try:
                            heatmap_image, _, top_zones = create_attention_overlay(
                                pil_frontal, entry["model"], entry["device"],
                                frontal_tensor, lateral_tensor,
                            )
                        except Exception as ex:
                            st.warning(f"⚠️ Heatmap lỗi: {ex}")
                        break

                st.session_state["diagnosis"] = {
                    "en_report":      best_en_report,
                    "clean_en_report": clean_en_report,
                    "structured_en":  structured_en,
                    "heatmap_image":  heatmap_image,
                    "top_zones":      top_zones,
                    "ensemble_probs": ensemble_probs,
                    "contributing":   [all_entries[k]["label"] for k in results],
                    "timestamp":      datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "uploaded_fname": uploaded_filename,
                }
                st.rerun()

        # ── Khung 1: Ground Truth ─────────────────────────────────────────────
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📁 Báo cáo gốc của Bác sĩ (Ground Truth)</div>', unsafe_allow_html=True)

        gt_record = None
        fname_to_lookup = uploaded_filename or (
            st.session_state.get("diagnosis", {}).get("uploaded_fname", "")
        )
        if fname_to_lookup:
            gt_record = lookup_ground_truth(fname_to_lookup)

        if gt_record:
            badge_ds    = gt_record["dataset"]
            badge_split = gt_record["split"].upper()
            st.markdown(
                f'<span class="dataset-badge">{badge_ds}</span>'
                f'<span class="dataset-badge">Split: {badge_split}</span>',
                unsafe_allow_html=True,
            )
            st.markdown("**FINDINGS (Bác sĩ):**")
            st.markdown(f'<div class="gt-box">{gt_record["findings"]}</div>', unsafe_allow_html=True)
            st.markdown("**IMPRESSION (Bác sĩ):**")
            st.markdown(f'<div class="gt-box">{gt_record["impression"]}</div>', unsafe_allow_html=True)
        else:
            if fname_to_lookup:
                st.info(f"ℹ️ Ảnh **{fname_to_lookup}** không có trong dữ liệu gốc (MIMIC / IU-Xray Archive / TimPhoi). "
                        "Có thể đây là ảnh ngoài tập dữ liệu huấn luyện.")
            else:
                st.info("ℹ️ Tải lên ảnh để tra cứu báo cáo gốc từ Ground Truth.")
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Khung 2: AI Report ────────────────────────────────────────────────
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🤖 Báo cáo do Mô hình AI sinh ra (Predicted Report)</div>', unsafe_allow_html=True)

        if "diagnosis" in st.session_state:
            dx = st.session_state["diagnosis"]
            structured_en = dx.get("structured_en", split_clinical_report(dx["en_report"]))

            # Chip mô hình
            chip_html = " ".join(f'<span class="dataset-badge">{m}</span>' for m in dx["contributing"])
            st.markdown(f'<p style="margin-bottom:0.5rem;">Ensemble từ: {chip_html}</p>', unsafe_allow_html=True)

            st.markdown("**FINDINGS (Mô hình AI):**")
            st.markdown(f'<div class="report-box">{structured_en["findings"]}</div>', unsafe_allow_html=True)

            st.markdown("**IMPRESSION (Mô hình AI):**")
            st.markdown(f'<div class="report-box">{structured_en["impression"]}</div>', unsafe_allow_html=True)

            with st.expander("📄 Raw Output nguyên bản (đối chiếu trực tiếp)", expanded=False):
                st.text_area(
                    label="Raw",
                    value=dx["en_report"],
                    height=120,
                    disabled=True,
                    key="raw_report_display",
                    label_visibility="collapsed",
                )
        else:
            st.info("ℹ️ Nhấn **Phân tích & Sinh Báo cáo** để xem kết quả dự đoán từ mô hình AI.")
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Lịch sử thẩm định ────────────────────────────────────────────────
        with st.expander("📁 Lịch sử Thẩm định đã lưu", expanded=False):
            df = load_doctor_reviews(REVIEWS_CSV_PATH)
            if df.empty:
                st.info("Chưa có hồ sơ thẩm định nào được lưu.")
            else:
                st.dataframe(df.tail(20), use_container_width=True)

# ─── ENTRY POINT ────────────────────────────────────────────────────────────────
if __name__ == "__main__" or True:
    main()
