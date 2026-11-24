# demo.py
# Hệ thống Hỗ trợ Chẩn đoán Hình ảnh X-Quang Ngực Lâm Sàng
# 3-Model Ensemble: Archive (IU-Xray) + TimPhoi + MIMIC-CXR → 14 CheXpert categories
# Phiên bản chuẩn hóa Bác sĩ Chẩn đoán Hình ảnh:
# - Contrastive Beam Search (k=3, visual_guidance=0.8, rep_penalty=1.20, no_repeat_ngram_size=3)
# - Làm sạch, ngắt câu giải phẫu và cắt đuôi câu đứt gãy (_clean_and_punctuate_report)
# - Chọn báo cáo tốt nhất theo đa dạng từ vựng, độ dài tự nhiên và đối sánh từ khóa MLC
# - Phiếu đọc phim chuẩn Bệnh viện Việt Nam (4 nhóm giải phẫu + Kết luận đồng bộ MLC)

from datetime import datetime
from pathlib import Path
import os, re, json, csv
import numpy as np
import torch
import streamlit as st
from PIL import Image
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

# ─── PATHS & ROBUST ARTIFACT RESOLVER ──────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
CURRENT_DIR = str(BASE_DIR)
REVIEWS_CSV_PATH = str(BASE_DIR / "clinical_reviews" / "verified_reports.csv")

def find_artifact_file(rel_subpath: str, filename: str) -> str:
    """Định vị tệp artifact chính xác, hỗ trợ tìm kiếm đệ quy nếu đường dẫn bị lệch."""
    p1 = BASE_DIR / rel_subpath
    if p1.is_file():
        return str(p1)
    # Tìm kiếm đệ quy bên trong thư mục artifacts/
    matches = list((BASE_DIR / "artifacts").rglob(filename))
    if matches and matches[0].is_file():
        return str(matches[0])
    return str(p1)

# ─── 14 CheXpert CLASSES ────────────────────────────────────────────────────────
CHEXPERT_CLASSES = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Effusion", "Pleural Thickening", "Fibrosis", "Fracture", "Support Devices",
]

VI_14_LABELS = [
    "Trung thất giãn rộng (Enlarged Cardiomediastinum)",
    "Bóng tim to (Cardiomegaly)",
    "Mờ phế trường phổi (Lung Opacity)",
    "Tổn thương dạng nốt/khối phổi (Lung Lesion)",
    "Phù phổi cấp / Ứ huyết (Pulmonary Edema)",
    "Đông đặc phế nang (Consolidation)",
    "Viêm phổi (Pneumonia)",
    "Xẹp phổi (Atelectasis)",
    "Tràn khí màng phổi (Pneumothorax)",
    "Tràn dịch màng phổi (Pleural Effusion)",
    "Dày dính màng phổi (Pleural Thickening)",
    "Xơ phổi / Xơ hóa mô kẽ (Fibrosis)",
    "Gãy xương sườn / đòn (Fracture)",
    "Thiết bị can thiệp y tế (Support Devices)",
]

# Từ khóa y khoa đối sánh cho từng nhóm bệnh CheXpert
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

# ─── DEFAULT MeSH LABELS (Archive model, 65 lớp) ────────────────────────────────
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

# ─── MeSH → CheXpert 14-class MAPPING ───────────────────────────────────────────
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

# ─── CSS: White/Black Clinical UI, hide sidebar ─────────────────────────────────
st.markdown("""
<style>
/* Hide sidebar completely */
section[data-testid="stSidebar"], button[data-testid="collapsedControl"] { display: none !important; }

/* White clinical background & typography */
.stApp,
section[data-testid="stAppViewContainer"],
div[data-testid="stAppViewBlockContainer"],
div[data-testid="stMain"] {
    background: #FFFFFF !important;
    color: #0F172A !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
}
.stApp p,
.stApp label,
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stExpander"] summary,
.stApp [data-testid="stExpander"] summary *,
.stApp [data-testid="stFileUploader"] *,
.stApp [data-testid="stAlert"] * {
    color: #0F172A !important;
}
.main .block-container { padding: 1.5rem 2.2rem !important; max-width: 1200px; margin: auto; }

/* Header card */
.clinical-header {
    background: #FFFFFF;
    color: #0F172A !important;
    padding: 1.6rem 2rem;
    border-radius: 12px;
    border: 1px solid #CBD5E1;
    border-top: 5px solid #0F172A;
    margin-bottom: 1.8rem;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
}
.clinical-header h1 { color: #0F172A !important; margin: 0; font-size: 1.65rem; font-weight: 700; }
.clinical-header p { color: #475569 !important; margin: 0.35rem 0 0 0; font-size: 0.9rem; }

/* Section cards */
.section-card {
    background: #F8FAFC;
    border: 1px solid #CBD5E1;
    border-radius: 10px;
    padding: 1.25rem 1.6rem;
    margin-bottom: 1.4rem;
}
.section-title {
    font-size: 1.02rem;
    font-weight: 700;
    color: #0F172A;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    border-bottom: 2px solid #0F172A;
    padding-bottom: 0.45rem;
    margin-bottom: 1rem;
}

/* Vietnamese report styling */
.vi-report {
    background: #FFFFFF;
    border-left: 4px solid #1E3A5F;
    border-radius: 6px;
    padding: 1.2rem 1.5rem;
    font-size: 1.02rem;
    line-height: 1.85;
    color: #0F172A;
    margin-bottom: 0.5rem;
    white-space: pre-wrap;
    font-family: inherit;
    border-top: 1px solid #E2E8F0;
    border-right: 1px solid #E2E8F0;
    border-bottom: 1px solid #E2E8F0;
}

/* Progress bar container */
.prob-bar-wrap { margin-bottom: 0.55rem; }
.prob-label { font-size: 0.83rem; color: #1E293B; font-weight: 600; margin-bottom: 3px; }
.prob-bar-bg { background: #E2E8F0; border-radius: 6px; height: 14px; width: 100%; }
.prob-bar-fg { height: 14px; border-radius: 6px; transition: width 0.4s ease; }
.prob-pct { font-size: 0.78rem; color: #475569; font-weight: 600; min-width: 44px; text-align: right; }

/* Upload hint */
.upload-hint { color: #64748B; font-size: 0.82rem; margin-top: 0.35rem; }

/* Buttons */
div[data-testid="stButton"] > button {
    background: #0F172A !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.4rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}
div.stButton > button,
div.stButton > button p,
div.stButton > button span,
div.stFormSubmitButton > button,
div.stFormSubmitButton > button p,
[data-testid="stBaseButton-secondary"] *,
[data-testid="stBaseButton-primary"] * {
    color: #ffffff !important;
    opacity: 1 !important;
    font-weight: 600 !important;
}
div[data-testid="stButton"] > button:hover {
    background: #1E3A5F !important;
}

/* Model chips */
.model-chip {
    display: inline-block;
    background: #EFF6FF;
    color: #1D4ED8;
    border: 1px solid #BFDBFE;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.76rem;
    font-weight: 600;
    margin: 2px;
}

/* Native Streamlit controls */
div[data-testid="stForm"],
div[data-testid="stExpander"],
div[data-testid="stFileUploader"],
div[data-testid="stAlert"],
div[data-testid="stFileUploader"] section,
div[data-testid="stFileUploader"] button,
div[data-testid="stExpander"] details,
div[data-testid="stExpander"] summary {
    background: #FFFFFF;
    border-color: #CBD5E1 !important;
}
div[data-testid="stForm"] label,
div[data-testid="stExpander"] label,
.stTextInput label,
.stTextArea label,
.stRadio label,
.stMultiSelect label {
    color: #0F172A !important;
}
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea,
textarea[data-testid="stTextArea"],
div[data-testid="stSelectbox"] > div,
div[data-testid="stMultiSelect"] > div {
    background: #FFFFFF !important;
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
    border-color: #94A3B8 !important;
    opacity: 1 !important;
}
div[data-testid="stTextArea"] textarea:disabled,
textarea[data-testid="stTextArea"]:disabled {
    background: #F8FAFC !important;
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
    opacity: 1 !important;
}
div[data-testid="stExpander"] summary,
div[data-testid="stFileUploader"] button,
div[data-testid="stAlert"] {
    color: #0F172A !important;
}
div[data-testid="stRadio"] label,
div[data-testid="stRadio"] p,
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] span {
    color: #0F172A;
}
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

# ─── LOAD ALL MODELS (cached) ──────────────────────────────────────────────────
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
        pt_path = cfg["pt"]
        vocab_path = cfg["vocab"]

        if not os.path.isfile(pt_path) or not os.path.isfile(vocab_path):
            print(f"[Model Loader Warning] {key}: File not found: pt={pt_path} (exists={os.path.isfile(pt_path)}), vocab={vocab_path} (exists={os.path.isfile(vocab_path)})")
            loaded[key] = {
                "error": f"Không tìm thấy file tệp: pt_exists={os.path.isfile(pt_path)}, vocab_exists={os.path.isfile(vocab_path)}",
                "label": cfg["label"],
                "missing": True,
            }
            continue

        try:
            print(f"[Model Loader] Loading checkpoint {key} from {pt_path}...")
            with open(vocab_path, "r", encoding="utf-8") as f:
                tok = json.load(f)
            id2word = {v: k for k, v in tok.items()}
            vocab_size = len(tok)
            pad_idx = tok.get("<pad>", 0)
            bos_idx = tok.get("<bos>", 1)
            eos_idx = tok.get("<eos>", 2)

            ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
            state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))

            mlc_key = None
            for k in state.keys():
                if "mlc_head" in k and "weight" in k and "fc" in k:
                    mlc_key = k
                    break
            if mlc_key and state[mlc_key].shape[0] in (14, 65):
                num_classes = state[mlc_key].shape[0]
            else:
                num_classes = cfg["num_classes"]

            mcfg = ModelConfig()
            mcfg.vocab_size = vocab_size
            mcfg.pad_idx = pad_idx
            mcfg.num_mlc_labels = num_classes
            tcfg = TrainConfig()

            model = MulticriteriaTransformer(mcfg, tcfg)
            missing, unexpected = model.load_state_dict(state, strict=False)
            model.eval()
            model.to(device)

            loaded[key] = {
                "model": model,
                "tok": tok,
                "id2word": id2word,
                "bos": bos_idx,
                "eos": eos_idx,
                "pad": pad_idx,
                "device": device,
                "num_classes": num_classes,
                "label": cfg["label"],
                "weight": cfg["weight"],
                "missing_keys": len(missing),
                "unexpected_keys": len(unexpected),
            }
            print(f"[Model Loader] {key} loaded successfully on {device} (classes={num_classes})")
        except Exception as ex:
            tb = traceback.format_exc()
            print(f"[Model Loader Error] {key}: {tb}")
            loaded[key] = {"error": str(ex), "traceback": tb, "label": cfg["label"]}

    return loaded

# ─── INFERENCE & BEAM SEARCH WITH N-GRAM BLOCKING ──────────────────────────────
def run_model_inference(entry: dict, frontal_tensor: torch.Tensor, lateral_tensor: torch.Tensor | None):
    model = entry["model"]
    device = entry["device"]
    bos = entry["bos"]
    eos = entry["eos"]
    id2word = entry["id2word"]
    pad = entry["pad"]

    frontal = frontal_tensor.to(device)
    if lateral_tensor is not None:
        lateral = lateral_tensor.to(device)
        images = torch.stack([frontal.squeeze(0), lateral.squeeze(0)], dim=0).unsqueeze(0)
    else:
        images = frontal.unsqueeze(1)

    with torch.no_grad():
        # Contrastive Beam Search (k=3, rep_penalty=1.20, visual_guidance=0.8, no_repeat_ngram_size=3)
        out_ids = model.generate(
            images,
            bos_idx=bos,
            eos_idx=eos,
            max_len=60,
            beam_size=3,
            repetition_penalty=1.20,
            visual_guidance=0.8,
            no_repeat_ngram_size=3,
        )

        ids = out_ids[0].tolist()
        words = []
        for idx in ids:
            if idx in (bos, eos, pad):
                if idx == eos:
                    break
                continue
            word = id2word.get(idx, "")
            if word and word not in ("<bos>", "<eos>", "<pad>", "<unk>"):
                words.append(word)
        report_en = " ".join(words).strip()

        # MLC probabilities
        probs = model.predict_mlc(images)
        probs_np = probs[0].cpu().numpy()

    return report_en, probs_np


def create_attention_overlay(
    pil_image: Image.Image,
    model,
    device,
    frontal_tensor: torch.Tensor,
    lateral_tensor: torch.Tensor,
) -> Image.Image:
    """Overlay the final encoder CLS attention on the frontal radiograph."""
    frontal = frontal_tensor.to(device)
    lateral = lateral_tensor.to(device)
    images = torch.stack(
        [frontal.squeeze(0), lateral.squeeze(0)],
        dim=0,
    ).unsqueeze(0)
    heatmaps = model.encoder.get_attention_heatmap(images, img_size=448)
    attention = heatmaps[0, 0].detach().float().cpu().numpy()
    if not np.isfinite(attention).all():
        raise ValueError("Encoder attention contains non-finite values.")
    heatmap = (
        (attention - attention.min())
        / (attention.max() - attention.min() + 1e-8)
    )

    heatmap_rgb = np.stack(
        (
            np.clip(1.5 - np.abs(4 * heatmap - 3), 0, 1),
            np.clip(1.5 - np.abs(4 * heatmap - 2), 0, 1),
            np.clip(1.5 - np.abs(4 * heatmap - 1), 0, 1),
        ),
        axis=-1,
    )
    heatmap_rgb = (heatmap_rgb * 255).astype(np.uint8)
    original = pil_image.convert("RGB").resize((448, 448))
    colored_heatmap = Image.fromarray(heatmap_rgb, mode="RGB")
    return Image.blend(original, colored_heatmap, alpha=0.45).resize(pil_image.size)

def map_archive_to_14(probs_65: np.ndarray) -> np.ndarray:
    out = np.zeros(14, dtype=np.float32)
    for chex_idx, mesh_indices in MESH_IDX_MAP.items():
        if mesh_indices:
            out[chex_idx] = float(np.max(probs_65[mesh_indices]))
    return out

def compute_ensemble(results: dict, all_entries: dict) -> np.ndarray:
    total_weight = 0.0
    weighted_sum = np.zeros(14, dtype=np.float32)

    for key, (report_en, probs_np) in results.items():
        w = all_entries[key]["weight"]
        num_cls = all_entries[key]["num_classes"]

        if num_cls == 65:
            probs14 = map_archive_to_14(probs_np)
        else:
            probs14 = probs_np[:14].astype(np.float32)

        weighted_sum += w * probs14
        total_weight += w

    if total_weight > 0:
        weighted_sum /= total_weight
    return weighted_sum

# ─── PHẦN 2: LÀM SẠCH, NGẮT CÂU & CẮT ĐUÔI CỤT TIẾNG ANH ────────────────────────
def _clean_and_punctuate_report(raw_text: str) -> str:
    """
    Làm sạch sâu chuỗi thô từ Decoder:
    - Loại bỏ tiền tố và tham chiếu phim cũ không có thật
    - Khử lặp cụm từ liền kề (1 đến 4 từ)
    - Tự động ngắt câu trước các mốc giải phẫu chuẩn
    - Cắt bỏ đuôi câu đứt gãy ở giới hạn 60 tokens
    """
    if not raw_text or not isinstance(raw_text, str):
        return ""
    text = raw_text.strip()

    # 1. Xóa tiền tố, góc chụp và token ẩn danh
    text = re.sub(r"^(findings|impression)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(frontal and lateral views of the chest show|frontal view of the chest shows|chest radiograph shows)\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(XXXX|xxxx|___+)\b", "", text)
    for tok in ("<bos>", "<eos>", "<pad>", "<unk>"):
        text = text.replace(tok, "")

    # Xóa so sánh với phim cũ không có thực
    text = re.sub(r"\bin comparison with (the )?(study|radiograph|exam|prior|examination)[^,.]*[,.]?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bcompared to [^,.]*[,.]?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bperhaps from (the )?(previous|prior) (exam|examination|study)[^,.]*[,.]?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bfrom (the )?(previous|prior) (exam|examination|study)[^,.]*[,.]?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(an )?unchanged\b", "", text, flags=re.IGNORECASE)

    # 2. Khử lặp cụm từ liền kề (1 đến 4 từ)
    for _ in range(3):
        text = re.sub(r"\b(\w+(?:\s+\w+){0,3})\s+\1\b", r"\1", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(\w+(?:\s+\w+){0,3})\s+and\s+\1\b", r"\1", text, flags=re.IGNORECASE)

    # Khử các cụm lặp cụ thể thường gặp
    text = re.sub(r"central airways pulmonary vasculature central airways", "central airways and pulmonary vasculature", text, flags=re.IGNORECASE)
    text = re.sub(r"indeterminant grade prominence and", "", text, flags=re.IGNORECASE)
    text = re.sub(r"and was consolidated of", "", text, flags=re.IGNORECASE)

    # 3. Chèn dấu chấm trước các mốc giải phẫu chuẩn nếu phía trước chưa có chấm
    landmarks = [
        "no focal", "no pleural", "no pneumothorax", "the lungs", "lungs are",
        "heart size", "cardiomediastinal", "cardiac", "osseous structures",
        "visualized osseous", "surgical clips", "sternotomy wires", "impression"
    ]
    for lm in landmarks:
        pattern = rf"(?<!\.\s)(?<!^)\b({re.escape(lm)})\b"
        text = re.sub(pattern, r". \1", text, flags=re.IGNORECASE)

    # Chuẩn hóa khoảng trắng và dấu chấm
    text = re.sub(r"\s*\.\s*", ". ", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\s{2,}", " ", text).strip()

    # 4. Cắt đuôi câu đứt gãy ở giới hạn 60 tokens
    dangling_words = {
        "there", "is", "are", "was", "were", "and", "or", "with", "of", "in",
        "to", "from", "the", "a", "an", "perhaps", "which", "for", "as", "at",
        "by", "patient", "age"
    }

    raw_sentences = [s.strip() for s in text.split(".") if s.strip()]
    valid_sentences = []
    for s in raw_sentences:
        words = s.split()
        if not words:
            continue
        while words and words[-1].lower() in dangling_words:
            words.pop()
        if len(words) >= 2:
            s_clean = " ".join(words)
            s_clean = s_clean[0].upper() + s_clean[1:]
            valid_sentences.append(s_clean)

    if valid_sentences:
        clean_text = ". ".join(valid_sentences) + "."
    else:
        clean_text = "Heart size and mediastinal contours are within normal limits. The lungs are clear."

    return clean_text

# ─── PHẦN 1: CHỌN BÁO CÁO TỐT NHẤT THEO ĐA DẠNG TỪ VỰNG & MLC MATCH ─────────────
def score_report_candidate(clean_en: str, ensemble_probs: np.ndarray, model_weight: float) -> float:
    words = [w.lower() for w in re.findall(r"\b[a-zA-Z]+\b", clean_en)]
    if len(words) < 5:
        return -999.0

    # Tỷ lệ đa dạng từ vựng
    lexical_diversity = len(set(words)) / max(len(words), 1)

    # Điểm thưởng độ dài tự nhiên (không chạm trần 60 tokens)
    if 12 <= len(words) <= 45:
        length_bonus = 2.0
    elif len(words) > 50:
        length_bonus = -1.5
    else:
        length_bonus = 0.5

    # Điểm thưởng khớp từ khóa bệnh lý MLC (xác suất >= 0.35)
    keyword_bonus = 0.0
    clean_low = clean_en.lower()
    if ensemble_probs is not None:
        for idx in range(min(14, len(ensemble_probs))):
            if ensemble_probs[idx] >= 0.35:
                kws = DISEASE_KEYWORDS.get(idx, [])
                if any(kw in clean_low for kw in kws):
                    keyword_bonus += 1.0

    score = (lexical_diversity * 3.0) + length_bonus + keyword_bonus + model_weight
    return score

def select_best_report(results: dict, all_entries: dict, ensemble_probs: np.ndarray) -> str:
    scored_candidates = []
    for key, (raw_en, _) in results.items():
        if not raw_en:
            continue
        clean_en = _clean_and_punctuate_report(raw_en)
        weight = all_entries[key]["weight"] if (all_entries and key in all_entries) else 0.33
        sc = score_report_candidate(clean_en, ensemble_probs, weight)
        scored_candidates.append((sc, clean_en))

    if not scored_candidates:
        return "Heart size and mediastinal contours are within normal limits. The lungs are clear."

    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    return scored_candidates[0][1]

# ─── PHẦN 3: DỊCH NGUYÊN BẢN & ĐỊNH DẠNG PHIẾU ĐỌC PHIM BỆNH VIỆN VIỆT NAM ───────
POST_TRANSLATION_REPLACEMENTS = [
    (r"\bbóng tim đến\b", "bóng tim to"),
    (r"\bkhí thải\b", "khí phế thũng"),
    (r"\bcấu trúc xương nguyên vẹn\b", "khung xương lồng ngực quan sát được không thấy bất thường cấp tính"),
    (r"\bcấu trúc xương có vẻ còn nguyên vẹn\b", "khung xương lồng ngực quan sát được không thấy bất thường cấp tính"),
    (r"\bcấu trúc xương xuất hiện nguyên vẹn\b", "khung xương lồng ngực quan sát được không thấy bất thường cấp tính"),
    (r"\bkhông có bệnh phổi cấp tính\b", "không ghi nhận bất thường tim phổi cấp tính"),
    (r"\bmàng phổi tràn dịch\b", "tràn dịch màng phổi"),
    (r"\bkhí màng phổi\b", "tràn khí màng phổi"),
    (r"\bkẹp phẫu thuật\b", "kẹp kim loại phẫu thuật cũ vùng trung thất còn nguyên vị trí"),
    (r"\bđông đặc đầu mối\b", "đông đặc khu trú"),
]

FALLBACK_PHRASE_DICT = [
    (["surgical clip", "clips"], "Ghi nhận hình ảnh kẹp kim loại phẫu thuật cũ vùng trung thất ổn định, đúng vị trí."),
    (["sternotomy wire", "wires"], "Chỉ thép khâu cố định xương ức sau mổ còn nguyên vẹn, đúng vị trí."),
    (["no focal consolidation", "no consolidation", "no focal airspace"], "Nhu mô phổi không thấy tổn thương đông đặc khu trú."),
    (["no pleural effusion or pneumothorax", "no pneumothorax or pleural effusion"], "Hai góc sườn hoành nhọn, không ghi nhận tràn dịch hay tràn khí màng phổi."),
    (["no pleural effusion", "no pleural effusions"], "Không thấy hình ảnh tràn dịch màng phổi."),
    (["no pneumothorax"], "Không thấy hình ảnh tràn khí màng phổi."),
    (["cardiomediastinal silhouette", "mediastinum size", "within normal limits"], "Bóng tim và trung thất có kích thước, đường viền trong giới hạn bình thường."),
    (["heart size is normal", "heart is normal", "cardiac silhouette is normal"], "Kích thước bóng tim trong giới hạn bình thường."),
    (["osseous structures appear intact", "osseous structures are intact", "bones are intact"], "Khung xương lồng ngực quan sát được không thấy bất thường cấp tính."),
    (["lungs are clear", "clear bilaterally", "well-expanded"], "Hai phế trường phổi sáng rõ, giãn nở tốt, không thấy thâm nhiễm phế nang."),
]

def translate_sentence_vi(s: str) -> str:
    s_clean = s.strip()
    if not s_clean:
        return ""
    s_low = s_clean.lower()

    if has_spinal_degeneration(s_clean):
        return SPINAL_DEGENERATION_VI

    # Khớp trước với bộ từ điển y khoa chuẩn
    for keys, vi in FALLBACK_PHRASE_DICT:
        if all(k in s_low for k in keys):
            return vi
    for keys, vi in FALLBACK_PHRASE_DICT:
        if any(k in s_low for k in keys):
            return vi

    # Dùng Google Translator nếu khả dụng
    if HAS_GOOGLE_TRANS and GoogleTranslator is not None:
        try:
            vi = GoogleTranslator(source="en", target="vi").translate(s_clean)
            for pat, repl in POST_TRANSLATION_REPLACEMENTS:
                vi = re.sub(pat, repl, vi, flags=re.IGNORECASE)
            return vi
        except Exception:
            pass

    # Fallback cuối cùng
    return translate_radiology_report(s_clean)

def format_vietnamese_clinical_report(clean_en: str, ensemble_probs=None,
                                       chexpert_labels=None, vi_labels=None) -> str:
    """
    Phân loại các nhận định vào 4 nhóm giải phẫu chuẩn Bệnh viện Việt Nam
    và đồng bộ chéo với nhánh phân loại bệnh lý MLC Ensemble.
    """
    sentences = [s.strip() for s in clean_en.split(".") if s.strip() and len(s.strip().split()) >= 2]

    translated = []
    for s in sentences:
        vi_sent = translate_sentence_vi(s)
        if vi_sent:
            translated.append(vi_sent)

    lung_findings = []
    pleural_findings = []
    heart_findings = []
    bone_findings = []

    for s in translated:
        low = s.lower()
        if any(k in low for k in ["trung thất", "bóng tim", "kẹp", "chỉ thép", "ống thông", "quai động mạch", "tim", "vasculature", "cardiomediastinal"]):
            heart_findings.append(s)
        elif any(k in low for k in ["sườn hoành", "tràn dịch", "tràn khí", "màng phổi", "khoang màng phổi", "pleural"]):
            pleural_findings.append(s)
        elif any(k in low for k in ["xương", "đốt sống", "sườn", "khung xương", "osseous"]):
            bone_findings.append(s)
        else:
            lung_findings.append(s)

    # Đồng bộ chéo với MLC (xác suất >= 40%)
    active_diseases = []
    if ensemble_probs is not None and vi_labels is not None:
        for prob, vi_lbl in zip(ensemble_probs, vi_labels):
            if prob >= 0.40:
                active_diseases.append((vi_lbl, prob))

    if ensemble_probs is not None:
        # 1. Cardiomegaly >= 40%
        if len(ensemble_probs) > 1 and ensemble_probs[1] >= 0.40:
            heart_findings = [f for f in heart_findings if "bình thường" not in f.lower()]
            heart_findings.append("Bóng tim có dấu hiệu to / mở rộng so với tỷ lệ lồng ngực.")
        # 2. Consolidation >= 40%
        if len(ensemble_probs) > 5 and ensemble_probs[5] >= 0.40:
            lung_findings = [f for f in lung_findings if "không thấy tổn thương" not in f.lower()]
            lung_findings.append("Ghi nhận hình ảnh đám mờ đông đặc nhu mô phổi phế nang.")
        # 3. Effusion >= 40%
        if len(ensemble_probs) > 9 and ensemble_probs[9] >= 0.40:
            pleural_findings = [f for f in pleural_findings if "không" not in f.lower()]
            pleural_findings.append("Ghi nhận hình ảnh mờ / tù góc sườn hoành, nghi ngờ tràn dịch màng phổi.")
        # 4. Pneumothorax >= 40%
        if len(ensemble_probs) > 8 and ensemble_probs[8] >= 0.40:
            pleural_findings = [f for f in pleural_findings if "không" not in f.lower()]
            pleural_findings.append("Nghi ngờ hình ảnh viền sáng tràn khí màng phổi khu trú.")

    # Khử lặp các nhận định trùng lặp trong từng nhóm
    def _dedup_findings(lst):
        seen = set()
        out = []
        for item in lst:
            norm = item.strip()
            if norm and norm not in seen:
                seen.add(norm)
                out.append(norm)
        return out

    lung_findings = _dedup_findings(lung_findings)
    pleural_findings = _dedup_findings(pleural_findings)
    heart_findings = _dedup_findings(heart_findings)
    bone_findings = _dedup_findings(bone_findings)

    # Điền giá trị bình thường chuẩn nếu nhóm còn trống
    if not lung_findings:
        lung_findings.append("Hai phế trường phổi sáng, không thấy tổn thương đông đặc hay đám mờ khu trú.")
    if not pleural_findings:
        pleural_findings.append("Hai góc sườn hoành nhọn, không có hình ảnh tràn dịch hoặc tràn khí màng phổi.")
    if not heart_findings:
        heart_findings.append("Bóng tim và trung thất có kích thước, đường viền trong giới hạn bình thường.")
    if not bone_findings:
        bone_findings.append("Khung xương lồng ngực quan sát được không thấy gãy xương hay tổn thương bất thường cấp tính.")

    # Kết luận lâm sàng
    conclusions = []
    if active_diseases:
        conclusions.append("Ghi nhận các dấu hiệu bất thường cần lưu ý:")
        for lbl, pr in active_diseases:
            conclusions.append(f"- {lbl} (Xác suất AI: {pr*100:.1f}%)")
        conclusions.append("Đề nghị kết hợp thăm khám lâm sàng hoặc chụp cắt lớp vi tính (CT) lồng ngực nếu cần thiết.")
    else:
        conclusions.append("Hiện chưa ghi nhận bất thường tim phổi cấp tính trên phim X-quang ngực.")

    lines = [
        "1. Nhu mô phổi:",
        "   " + " ".join(lung_findings),
        "",
        "2. Màng phổi:",
        "   " + " ".join(pleural_findings),
        "",
        "3. Tim – Trung thất:",
        "   " + " ".join(heart_findings),
        "",
        "4. Khung xương thành ngực:",
        "   " + " ".join(bone_findings),
        "",
        "KẾT LUẬN:",
        "   " + "\n   ".join(conclusions)
    ]
    return "\n".join(lines)

# ─── PROGRESS BAR 4-MÀU HTML ───────────────────────────────────────────────────
def _bar_color(prob: float) -> str:
    if prob < 0.25:
        return "#16A34A"   # Green (<25%)
    elif prob < 0.50:
        return "#EAB308"   # Yellow (25-49%)
    elif prob < 0.75:
        return "#F97316"   # Orange (50-74%)
    else:
        return "#DC2626"   # Red (>=75%)

def render_disease_bars(ensemble_probs: np.ndarray) -> None:
    cols = st.columns(2)
    for i, (en_name, vi_name) in enumerate(zip(CHEXPERT_CLASSES, VI_14_LABELS)):
        prob = float(ensemble_probs[i])
        pct = prob * 100
        color = _bar_color(prob)
        col = cols[i % 2]
        with col:
            bar_html = f"""
<div class="prob-bar-wrap">
  <div class="prob-label">{vi_name}</div>
  <div style="display:flex; align-items:center; gap:8px;">
    <div class="prob-bar-bg" style="flex:1;">
      <div class="prob-bar-fg" style="width:{min(pct,100):.1f}%; background:{color};"></div>
    </div>
    <span class="prob-pct">{pct:.1f}%</span>
  </div>
</div>"""
            st.markdown(bar_html, unsafe_allow_html=True)

# ─── MAIN APP ───────────────────────────────────────────────────────────────────
def main():
    # ── Header ──────────────────────────────────────────────────────────────────
    st.markdown("""
<div class="clinical-header">
  <h1>🫁 Hệ thống AI Hỗ trợ Chẩn đoán X-Quang Ngực</h1>
  <p>Phiên bản Lâm sàng · 3-Model Ensemble (Archive 0.40 + TimPhoi 0.35 + MIMIC 0.25) · Multicriteria Transformer</p>
</div>
""", unsafe_allow_html=True)

    # ── Upload Section ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-card"><div class="section-title">📂 Tải lên ảnh X-Quang</div>', unsafe_allow_html=True)
    col_f, col_l = st.columns(2)
    with col_f:
        frontal_file = st.file_uploader(
            "**Ảnh FRONTAL** *(bắt buộc)*",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            key="frontal_upload",
            help="Ảnh chụp X-quang tư thế thẳng (PA hoặc AP)",
        )
        st.markdown('<p class="upload-hint">Yêu cầu: PNG/JPG, tư thế thẳng (PA/AP)</p>', unsafe_allow_html=True)
    with col_l:
        lateral_file = st.file_uploader(
            "**Ảnh LATERAL** *(tùy chọn)*",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            key="lateral_upload",
            help="Ảnh chụp X-quang tư thế nghiêng (tăng độ chính xác Ensemble)",
        )
        st.markdown('<p class="upload-hint">Khuyến nghị: Cung cấp cả 2 góc để tăng độ chính xác</p>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Image Loading & Preview Logic ──────────────────────────────────────────
    pil_frontal = None
    pil_lateral = None
    lateral_caption = ""

    if frontal_file is not None and lateral_file is not None:
        pil_frontal = Image.open(frontal_file).convert("RGB")
        pil_lateral = Image.open(lateral_file).convert("RGB")
        lateral_caption = "Phim X-quang Nghiêng (Lateral)"
    elif frontal_file is not None and lateral_file is None:
        pil_frontal = Image.open(frontal_file).convert("RGB")
        pil_lateral = pil_frontal.copy()
        lateral_caption = "Phim bổ trợ (Tự động đồng bộ từ phim Thẳng)"
    else:
        st.session_state.pop("diagnosis", None)
        st.info("Vui lòng tải lên Ảnh X-quang Thẳng (Frontal) để bắt đầu phân tích.")

    # Hiển thị ảnh nếu đã có ảnh Frontal hợp lệ
    if pil_frontal is not None and pil_lateral is not None:
        heatmap_image = st.session_state.get("diagnosis", {}).get("heatmap_image")
        if heatmap_image is not None:
            col_img1, col_heatmap, col_img2 = st.columns(3)
            with col_img1:
                st.image(pil_frontal, caption="Phim X-quang Thẳng (Frontal)", width="stretch")
            with col_heatmap:
                st.image(
                    heatmap_image,
                    caption="Bản đồ chú ý phủ trên phim Frontal",
                    width="stretch",
                )
            with col_img2:
                st.image(pil_lateral, caption=lateral_caption, width="stretch")
        else:
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.image(pil_frontal, caption="Phim X-quang Thẳng (Frontal)", width="stretch")
            with col_img2:
                st.image(pil_lateral, caption=lateral_caption, width="stretch")

        # ── Run Analysis Button ──────────────────────────────────────────────────
        analyze_clicked = st.button(
            "🔬 Phân tích & Sinh Báo cáo X-Quang",
            key="run_analysis",
        )

        if analyze_clicked:
            with st.spinner("⏳ Đang nạp 3 mô hình và thực hiện phân tích ensemble..."):
                all_entries = load_all_models()

                # Cảnh báo chi tiết nếu có mô hình bị thiếu file hoặc lỗi nạp
                for k, entry in all_entries.items():
                    if entry is None or entry.get("missing"):
                        st.warning(f"⚠️ Mô hình '{k}': Không tìm thấy tệp trọng số hoặc từ điển trong `artifacts/{k}/`.")
                    elif "error" in entry:
                        st.warning(f"⚠️ Lỗi khi nạp mô hình '{entry.get('label', k)}': {entry['error']}")
                        with st.expander(f"Chi tiết ngoại lệ {k}", expanded=False):
                            st.code(entry.get("traceback", entry["error"]))

                # Preprocess images
                frontal_tensor = preprocess_image(pil_frontal)
                lateral_tensor = preprocess_image(pil_lateral)

                # Run inference on each available model
                results = {}
                model_status = {}
                for key, entry in all_entries.items():
                    if entry is None or "error" in entry:
                        model_status[key] = "load_failed"
                        continue
                    try:
                        report_en, probs_np = run_model_inference(entry, frontal_tensor, lateral_tensor)
                        results[key] = (report_en, probs_np)
                        model_status[key] = "ok"
                    except Exception as ex:
                        import traceback
                        tb = traceback.format_exc()
                        print(f"[Inference Error] {key}: {tb}")
                        model_status[key] = f"error: {str(ex)}"
                        st.warning(f"⚠️ Lỗi khi chạy suy luận trên mô hình '{entry.get('label', key)}': {ex}")
                        with st.expander(f"Chi tiết lỗi suy luận {key}", expanded=False):
                            st.code(tb)

                if not results:
                    st.error("❌ Không có mô hình nào suy luận thành công. Vui lòng kiểm tra các cảnh báo chi tiết ở trên.")
                    return

                # Compute 3-Model Ensemble
                ensemble_probs = compute_ensemble(results, all_entries)

                # Chọn báo cáo tốt nhất theo đa dạng từ vựng & khớp từ khóa MLC
                best_en_report = select_best_report(results, all_entries, ensemble_probs)

                # Định dạng Phiếu đọc phim chuẩn Bệnh viện Việt Nam
                vi_report = format_vietnamese_clinical_report(
                    best_en_report, ensemble_probs, CHEXPERT_CLASSES, VI_14_LABELS
                )
                clean_en_report = format_clinical_report(best_en_report)
                structured_en = split_clinical_report(clean_en_report)

                heatmap_image = None
                for key in results:
                    entry = all_entries.get(key)
                    if entry and "model" in entry:
                        heatmap_image = create_attention_overlay(
                            pil_frontal,
                            entry["model"],
                            entry["device"],
                            frontal_tensor,
                            lateral_tensor,
                        )
                        break

                contributing = [all_entries[k]["label"] for k in results]

                # Bệnh lý mặc định được chọn (xác suất >= 35%)
                default_diseases = [
                    VI_14_LABELS[i]
                    for i in range(14)
                    if float(ensemble_probs[i]) >= 0.35
                ]

                # Lưu vào session state
                st.session_state["diagnosis"] = {
                    "vi_report": vi_report,
                    "en_report": best_en_report,
                    "clean_en_report": clean_en_report,
                    "structured_en": structured_en,
                    "heatmap_image": heatmap_image,
                    "ensemble_probs": ensemble_probs,
                    "contributing": contributing,
                    "model_status": model_status,
                    "default_diseases": default_diseases,
                    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                }
                st.rerun()

    # ── Display Diagnosis Results ────────────────────────────────────────────────
    if "diagnosis" not in st.session_state:
        return

    dx = st.session_state["diagnosis"]

    # ── Section 1: Vietnamese Structured Report ──────────────────────────────────
    st.markdown('<div class="section-card"><div class="section-title">📋 Phiếu Đọc Phim X-Quang Lâm Sàng (AI Gợi Ý)</div>', unsafe_allow_html=True)

    chip_html = " ".join(f'<span class="model-chip">{m}</span>' for m in dx["contributing"])
    st.markdown(f'<p style="margin-bottom:0.6rem;">Ensemble từ: {chip_html}</p>', unsafe_allow_html=True)

    st.markdown(f'<div class="vi-report">{dx["vi_report"]}</div>', unsafe_allow_html=True)

    structured_en = dx.get("structured_en", split_clinical_report(dx["en_report"]))
    st.markdown(
        '<div class="section-card"><div class="section-title">FINDINGS</div>'
        f'<div class="vi-report">{structured_en["findings"]}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-card"><div class="section-title">IMPRESSION</div>'
        f'<div class="vi-report">{structured_en["impression"]}</div></div>',
        unsafe_allow_html=True,
    )

    with st.expander("📄 Raw Output nguyên bản", expanded=False):
        st.text_area(
            label="Raw model output",
            value=dx["en_report"],
            height=130,
            disabled=True,
            key="raw_report_display",
            label_visibility="collapsed",
        )

    with st.expander("📄 Xem báo cáo tiếng Anh đã ngắt câu chuẩn hóa (Clean English Source)", expanded=False):
        st.text_area(
            label="English Report",
            value=(
                dx["clean_en_report"]
                if "clean_en_report" in dx
                else format_clinical_report(dx["en_report"])
            ),
            height=130,
            disabled=True,
            key="en_report_display",
            label_visibility="collapsed",
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section 2: Disease Probability Bars ─────────────────────────────────────
    st.markdown('<div class="section-card"><div class="section-title">📊 Xác suất Bệnh lý (14 Nhóm CheXpert)</div>', unsafe_allow_html=True)

    st.markdown("""
<div style="display:flex; gap:16px; margin-bottom:1.1rem; flex-wrap:wrap; font-size:0.85rem;">
  <span><span style="background:#16A34A; border-radius:3px; padding:1px 8px; color:white; font-size:0.75rem;">■</span> &lt;25% (Rất thấp)</span>
  <span><span style="background:#EAB308; border-radius:3px; padding:1px 8px; color:white; font-size:0.75rem;">■</span> 25–49% (Trung bình)</span>
  <span><span style="background:#F97316; border-radius:3px; padding:1px 8px; color:white; font-size:0.75rem;">■</span> 50–74% (Cao)</span>
  <span><span style="background:#DC2626; border-radius:3px; padding:1px 8px; color:white; font-size:0.75rem;">■</span> ≥75% (Rất cao)</span>
</div>
""", unsafe_allow_html=True)

    render_disease_bars(dx["ensemble_probs"])
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section 3: Doctor Review Form ───────────────────────────────────────────
    st.markdown('<div class="section-card"><div class="section-title">👨‍⚕️ Thẩm định của Bác sĩ</div>', unsafe_allow_html=True)

    with st.form("doctor_review_form", clear_on_submit=False):
        col_dr1, col_dr2 = st.columns(2)
        with col_dr1:
            doctor_name = st.text_input(
                "Họ tên Bác sĩ Chẩn đoán Hình ảnh",
                value="BS. Chẩn đoán Hình ảnh",
                key="doctor_name",
            )
        with col_dr2:
            ai_evaluation = st.radio(
                "Đánh giá độ chính xác của AI",
                options=["✅ Chính xác", "✏️ Cần chỉnh sửa", "❌ Sai lệch"],
                index=0,
                horizontal=True,
                key="ai_eval",
            )

        verified_diseases = st.multiselect(
            "Xác nhận danh mục bệnh lý cuối cùng (mặc định: xác suất AI ≥ 35%)",
            options=VI_14_LABELS,
            default=dx["default_diseases"],
            key="verified_diseases",
        )

        doctor_notes = st.text_area(
            "Chú thích lâm sàng & vị trí tổn thương (nếu có)",
            placeholder="Nhập nhận định bổ sung về vị trí, tính chất hoặc khuyến nghị cận lâm sàng...",
            height=90,
            key="doctor_notes",
        )

        final_conclusion = st.text_area(
            "Kết luận chẩn đoán cuối cùng của Bác sĩ",
            value=dx["vi_report"],
            height=260,
            key="final_conclusion",
        )

        submitted = st.form_submit_button("💾 Lưu thẩm định & Xuất phiếu kết quả")

    if submitted:
        case_id = f"CASE_{dx['timestamp']}"
        ensemble_label = " + ".join(dx["contributing"])

        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "case_id": case_id,
            "model_used": ensemble_label,
            "doctor_name": doctor_name,
            "ai_evaluation": ai_evaluation,
            "doctor_notes": doctor_notes,
            "verified_diseases": "; ".join(verified_diseases) if isinstance(verified_diseases, list) else str(verified_diseases),
            "final_clinical_conclusion": final_conclusion,
            "ai_report_vi": dx["vi_report"],
            "ai_report_en": dx["en_report"],
        }

        try:
            save_doctor_review(REVIEWS_CSV_PATH, record)
            st.success("✅ Đã lưu hồ sơ thẩm định vào hệ thống thành công.")
        except Exception as ex:
            st.warning(f"⚠️ Lưu CSV thất bại: {ex}")

        diseases_list = verified_diseases if isinstance(verified_diseases, list) else [str(verified_diseases)]
        cert_text = export_clinical_certificate(
            case_id=case_id,
            model_name=ensemble_label,
            doctor_name=doctor_name,
            ai_eval=ai_evaluation,
            verified_diseases=diseases_list,
            final_conclusion=final_conclusion,
            notes=doctor_notes,
            ai_report_vi=dx["vi_report"],
            ai_report_en=dx["en_report"],
        )

        st.download_button(
            label="📥 Tải xuống Phiếu Kết quả (.txt)",
            data=cert_text.encode("utf-8-sig"),
            file_name=f"phieu_chandoan_{case_id}.txt",
            mime="text/plain",
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ── History (collapsed) ──────────────────────────────────────────────────────
    with st.expander("📁 Lịch sử Thẩm định đã lưu", expanded=False):
        df = load_doctor_reviews(REVIEWS_CSV_PATH)
        if df.empty:
            st.info("Chưa có hồ sơ thẩm định nào được lưu.")
        else:
            st.dataframe(df.tail(20), width="stretch")

# ─── ENTRY POINT ────────────────────────────────────────────────────────────────
if __name__ == "__main__" or True:
    main()
