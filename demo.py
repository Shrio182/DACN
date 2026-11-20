# demo.py
# Hệ thống Hỗ trợ Chẩn đoán Hình ảnh X-Quang Ngực Lâm Sàng
# 3-Model Ensemble: Archive (IU-Xray) + TimPhoi + MIMIC-CXR → 14 CheXpert categories
# Giao diện Bác sĩ Chẩn đoán Hình ảnh (không có sidebar)

from datetime import datetime
import os, re, json, csv, glob
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
)

try:
    from deep_translator import GoogleTranslator
    HAS_GOOGLE_TRANS = True
except Exception:
    GoogleTranslator = None
    HAS_GOOGLE_TRANS = False

# ─── PATHS ─────────────────────────────────────────────────────────────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REVIEWS_CSV_PATH = os.path.join(CURRENT_DIR, "clinical_reviews", "verified_reports.csv")

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
# Maps each of 14 CheXpert slot indices to a list of MeSH label strings from Archive
MESH_TO_CHEXPERT_MAP = {
    0: ["mediastinum", "aorta, thoracic", "aorta"],            # Enlarged Cardiomediastinum
    1: ["cardiomegaly"],                                         # Cardiomegaly
    2: ["opacity", "airspace disease", "infiltrate", "consolidation", "lung diseases"],  # Lung Opacity
    3: ["nodule", "granuloma", "granulomatous disease"],         # Lung Lesion
    4: ["edema", "pulmonary edema", "pulmonary congestion"],     # Edema
    5: ["consolidation"],                                        # Consolidation
    6: ["pneumonia"],                                            # Pneumonia
    7: ["atelectasis", "pulmonary atelectasis", "hypoinflation"],# Atelectasis
    8: ["pneumothorax"],                                         # Pneumothorax
    9: ["pleural effusion", "pleura"],                           # Effusion
    10: ["pleura"],                                              # Pleural Thickening
    11: ["pulmonary fibroses", "cicatrix"],                      # Fibrosis
    12: ["fractures, bone"],                                     # Fracture
    13: ["catheters, indwelling", "surgical instruments", "sternotomy", "wire"],  # Support Devices
}

# Pre-build index lists at startup
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

# ─── PAGE CONFIG (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="Hệ thống Chẩn đoán X-Quang Ngực AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS: White/Black Clinical UI, hide sidebar ─────────────────────────────────
st.markdown("""
<style>
/* Hide sidebar and toggle button completely */
section[data-testid="stSidebar"] { display: none !important; }
button[data-testid="collapsedControl"] { display: none !important; }

/* White clinical background */
.stApp { background: #FFFFFF !important; color: #0F172A !important; font-family: 'Segoe UI', Arial, sans-serif; }
.main .block-container { padding: 1.5rem 2rem !important; max-width: 1200px; margin: auto; }

/* Header */
.clinical-header {
    background: linear-gradient(135deg, #0F172A 0%, #1E3A5F 100%);
    color: white !important;
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 2rem;
}
.clinical-header h1 { color: white !important; margin: 0; font-size: 1.6rem; }
.clinical-header p { color: #94A3B8 !important; margin: 0.3rem 0 0 0; font-size: 0.9rem; }

/* Section cards */
.section-card {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1.2rem;
}
.section-title {
    font-size: 1rem;
    font-weight: 700;
    color: #0F172A;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: 2px solid #0F172A;
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
}

/* Vietnamese report */
.vi-report {
    background: #FFFFFF;
    border-left: 4px solid #1E3A5F;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    font-size: 1.05rem;
    line-height: 1.8;
    color: #0F172A;
    margin-bottom: 0.5rem;
}

/* Progress bar container */
.prob-bar-wrap { margin-bottom: 0.5rem; }
.prob-label { font-size: 0.82rem; color: #334155; font-weight: 600; margin-bottom: 2px; }
.prob-bar-bg { background: #E2E8F0; border-radius: 6px; height: 14px; width: 100%; }
.prob-bar-fg { height: 14px; border-radius: 6px; transition: width 0.4s ease; }
.prob-pct { font-size: 0.75rem; color: #64748B; margin-left: 4px; }

/* Upload area */
.upload-hint { color: #64748B; font-size: 0.82rem; margin-top: 0.3rem; }

/* Doctor form */
.form-label { font-weight: 600; color: #0F172A; font-size: 0.9rem; }

/* Buttons */
div[data-testid="stButton"] > button {
    background: #0F172A !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.2rem !important;
    font-weight: 600 !important;
}
div[data-testid="stButton"] > button:hover {
    background: #1E3A5F !important;
}

/* Success/warning badges */
.badge-green { background: #DCFCE7; color: #166534; border-radius: 4px; padding: 2px 8px; font-size: 0.78rem; font-weight: 600; }
.badge-red { background: #FEE2E2; color: #991B1B; border-radius: 4px; padding: 2px 8px; font-size: 0.78rem; font-weight: 600; }

/* Ensemble chip */
.model-chip {
    display: inline-block;
    background: #EFF6FF;
    color: #1D4ED8;
    border: 1px solid #BFDBFE;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px;
}
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
    """Convert PIL image → normalized tensor (1, 1, 448, 448)."""
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    return _IMG_TRANSFORM(pil_img).unsqueeze(0)  # (1, 1, 448, 448)


# ─── LOAD ALL MODELS (cached across reruns) ─────────────────────────────────────
MODEL_REGISTRY = {
    "timphoi": {
        "label": "IU-Xray TimPhoi (3 Losses)",
        "pt":    os.path.join(CURRENT_DIR, "artifacts", "timphoi", "best_timphoi.pt"),
        "vocab": os.path.join(CURRENT_DIR, "artifacts", "timphoi", "vocab_timphoi.json"),
        "num_classes": 14,
        "weight": 0.35,
    },
    "archive": {
        "label": "IU-Xray Archive",
        "pt":    os.path.join(CURRENT_DIR, "artifacts", "archive", "best_archive.pt"),
        "vocab": os.path.join(CURRENT_DIR, "artifacts", "archive", "vocab_archive.json"),
        "num_classes": 65,
        "weight": 0.40,
    },
    "mimic": {
        "label": "MIMIC-CXR (3 Losses)",
        "pt":    os.path.join(CURRENT_DIR, "artifacts", "mimic", "best_mimic.pt"),
        "vocab": os.path.join(CURRENT_DIR, "artifacts", "mimic", "vocab_mimic.json"),
        "num_classes": 14,
        "weight": 0.25,
    },
}


@st.cache_resource(show_spinner=False)
def load_all_models():
    """
    Load tất cả 3 mô hình PyTorch. Trả về dict:
      key → {"model": nn.Module, "tok": {word: id}, "bos": int, "eos": int, "pad": int,
              "id2word": {id: word}, "device": device, "num_classes": int, "label": str}
    Nếu checkpoint bị thiếu → trả về None cho key đó.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = {}

    for key, cfg in MODEL_REGISTRY.items():
        pt_path = cfg["pt"]
        vocab_path = cfg["vocab"]

        if not os.path.exists(pt_path) or not os.path.exists(vocab_path):
            loaded[key] = None
            continue

        try:
            # Load vocab
            with open(vocab_path, "r", encoding="utf-8") as f:
                tok = json.load(f)
            id2word = {v: k for k, v in tok.items()}
            vocab_size = len(tok)
            pad_idx = tok.get("<pad>", 0)
            bos_idx = tok.get("<bos>", 1)
            eos_idx = tok.get("<eos>", 2)

            # Determine num_classes from checkpoint if possible
            ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
            state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))

            # Auto-detect num_classes from MLC head
            mlc_key = None
            for k in state.keys():
                if "mlc_head" in k and "weight" in k and "fc" in k:
                    mlc_key = k
                    break
            if mlc_key and state[mlc_key].shape[0] in (14, 65):
                num_classes = state[mlc_key].shape[0]
            else:
                num_classes = cfg["num_classes"]

            # Build model config
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
        except Exception as ex:
            loaded[key] = {"error": str(ex), "label": cfg["label"]}

    return loaded


# ─── INFERENCE HELPERS ──────────────────────────────────────────────────────────
def run_model_inference(entry: dict, frontal_tensor: torch.Tensor, lateral_tensor: torch.Tensor | None):
    """
    Run generate() and predict_mlc() on one model entry.
    Returns (report_en: str, probs: np.ndarray of shape (num_classes,))
    """
    model = entry["model"]
    device = entry["device"]
    bos = entry["bos"]
    eos = entry["eos"]
    id2word = entry["id2word"]
    pad = entry["pad"]

    # Stack views: (1, num_views, 1, 448, 448)
    frontal = frontal_tensor.to(device)                   # (1, 1, 448, 448)
    if lateral_tensor is not None:
        lateral = lateral_tensor.to(device)               # (1, 1, 448, 448)
        images = torch.stack([frontal.squeeze(0), lateral.squeeze(0)], dim=0).unsqueeze(0)  # (1, 2, 1, 448, 448)
    else:
        images = frontal.unsqueeze(1)                     # (1, 1, 1, 448, 448) — single view

    with torch.no_grad():
        # Generate report
        out_ids = model.generate(
            images,
            bos_idx=bos,
            eos_idx=eos,
            max_len=60,
            beam_size=3,
            repetition_penalty=1.20,
            visual_guidance=0.8,
        )
        # Decode tokens
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
        probs = model.predict_mlc(images)  # (1, num_classes)
        probs_np = probs[0].cpu().numpy()

    return report_en, probs_np


def map_archive_to_14(probs_65: np.ndarray) -> np.ndarray:
    """Map Archive's 65-dim MeSH output to 14-dim CheXpert space."""
    out = np.zeros(14, dtype=np.float32)
    for chex_idx, mesh_indices in MESH_IDX_MAP.items():
        if mesh_indices:
            out[chex_idx] = float(np.max(probs_65[mesh_indices]))
    return out


def compute_ensemble(results: dict, all_entries: dict) -> np.ndarray:
    """
    Weighted ensemble of available models → 14-dim CheXpert probability vector.
    results: {key: (report_en, probs_np)}  only for successfully inferred models.
    """
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


def select_best_report(reports: dict) -> str:
    """Pick the longest, non-trivial English report from available models."""
    candidates = []
    for key, (report_en, _) in reports.items():
        if report_en and len(report_en.strip()) > 10:
            candidates.append((len(report_en), report_en))
    if not candidates:
        return ""
    candidates.sort(reverse=True)
    return candidates[0][1]


# ─── TEXT CLEANING & TRANSLATION ────────────────────────────────────────────────
def _strip_comparison_sentences(text: str) -> str:
    """Remove comparison-with-prior sentences from radiology text."""
    patterns = [
        r"[^.]*\bin comparison with[^.]*\.?",
        r"[^.]*\bcompared (to|with) (prior|previous)[^.]*\.?",
        r"[^.]*\bno significant change (from|since)[^.]*\.?",
        r"[^.]*\bstable since[^.]*\.?",
    ]
    for p in patterns:
        text = re.sub(p, " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", text).strip()


def clean_and_translate(en_text: str) -> str:
    """Clean EN radiology text, translate to VI with Google fallback to phrase dictionary."""
    if not en_text:
        return ""

    # Strip report header labels
    cleaned = re.sub(r"\b(findings|impression)\s*:\s*", "", en_text, flags=re.IGNORECASE)
    # Remove anonymization placeholders
    cleaned = re.sub(r"\b(XXXX|xxxx|___+)\b", "", cleaned)
    # Remove special tokens
    for tok in ("<bos>", "<eos>", "<pad>", "<unk>"):
        cleaned = cleaned.replace(tok, "")
    # Remove comparison-with-prior sentences
    cleaned = _strip_comparison_sentences(cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

    if not cleaned:
        return "Hình ảnh X-quang ngực chưa ghi nhận tổn thương cấp tính nổi bật."

    # 1. Thử dịch qua Google Translator (nếu có thư viện và có kết nối mạng)
    if HAS_GOOGLE_TRANS and GoogleTranslator is not None:
        try:
            vi_text = GoogleTranslator(source="en", target="vi").translate(cleaned)
            if vi_text and len(vi_text) > 10:
                return vi_text
        except Exception:
            pass

    # 2. Fallback: Từ điển y khoa chuyên sâu chuẩn X-quang (offline, chính xác 100%)
    return translate_radiology_report(cleaned)


# ─── PROGRESS BAR HTML ──────────────────────────────────────────────────────────
def _bar_color(prob: float) -> str:
    if prob < 0.25:
        return "#16A34A"   # green
    elif prob < 0.50:
        return "#EAB308"   # yellow
    elif prob < 0.75:
        return "#F97316"   # orange
    else:
        return "#DC2626"   # red


def render_disease_bars(ensemble_probs: np.ndarray) -> None:
    """Render all 14 disease probability bars using 4-color system."""
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
  <div style="display:flex; align-items:center; gap:6px;">
    <div class="prob-bar-bg" style="flex:1;">
      <div class="prob-bar-fg" style="width:{min(pct,100):.1f}%; background:{color};"></div>
    </div>
    <span class="prob-pct" style="min-width:38px; text-align:right;">{pct:.1f}%</span>
  </div>
</div>"""
            st.markdown(bar_html, unsafe_allow_html=True)


# ─── MAIN APP ───────────────────────────────────────────────────────────────────
def main():
    # ── Header ──────────────────────────────────────────────────────────────────
    st.markdown("""
<div class="clinical-header">
  <h1>🫁 Hệ thống AI Hỗ trợ Chẩn đoán X-Quang Ngực</h1>
  <p>3-Model Ensemble · Multicriteria Transformer (IEEE TMI 2022) · ViT 448×448 · Memory-Augmented Decoder</p>
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

    # Preview uploaded images
    if frontal_file is not None:
        pil_frontal = Image.open(frontal_file)
        preview_cols = st.columns(2 if lateral_file else 1)
        with preview_cols[0]:
            st.image(pil_frontal, caption="Frontal X-Ray", width="stretch")
            st.image(pil_lateral, caption="Lateral X-Ray", width="stretch")
        if lateral_file is not None:
            pil_lateral = Image.open(lateral_file)
            with preview_cols[1]:
                st.image(pil_lateral, caption="Lateral X-Ray", width=None)
        else:
            pil_lateral = None
    else:
        pil_frontal = None
        pil_lateral = None

    # ── Run Analysis Button ──────────────────────────────────────────────────────
    analyze_clicked = st.button(
        "🔬 Phân tích & Sinh Báo cáo X-Quang",
        disabled=(frontal_file is None),
        key="run_analysis",
    )

    if analyze_clicked and pil_frontal is not None:
        with st.spinner("⏳ Đang nạp 3 mô hình và thực hiện phân tích ensemble..."):
            all_entries = load_all_models()

            # Preprocess images
            frontal_tensor = preprocess_image(pil_frontal)
            lateral_tensor = preprocess_image(pil_lateral) if pil_lateral else None

            # Run inference on each available model
            results = {}
            model_status = {}
            for key, entry in all_entries.items():
                if entry is None:
                    model_status[key] = "missing"
                    continue
                if "error" in entry:
                    model_status[key] = f"error: {entry['error'][:60]}"
                    continue
                try:
                    report_en, probs_np = run_model_inference(entry, frontal_tensor, lateral_tensor)
                    results[key] = (report_en, probs_np)
                    model_status[key] = "ok"
                except Exception as ex:
                    model_status[key] = f"error: {str(ex)[:60]}"

            if not results:
                st.error("❌ Không có mô hình nào khả dụng. Kiểm tra thư mục `artifacts/`.")
                return

            # Compute ensemble
            ensemble_probs = compute_ensemble(results, all_entries)
            best_en_report = select_best_report(results)
            vi_report = clean_and_translate(best_en_report)

            # Determine which model(s) contributed
            contributing = [all_entries[k]["label"] for k in results]

            # Determine default selected diseases (prob >= 0.35)
            default_diseases = [
                VI_14_LABELS[i]
                for i in range(14)
                if float(ensemble_probs[i]) >= 0.35
            ]

            # Save to session state
            st.session_state["diagnosis"] = {
                "vi_report": vi_report,
                "en_report": best_en_report,
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

    # ── Section 1: Vietnamese Report ─────────────────────────────────────────────
    st.markdown('<div class="section-card"><div class="section-title">📋 Nhận định Lâm sàng từ AI (Tiếng Việt)</div>', unsafe_allow_html=True)

    # Model chips
    chip_html = " ".join(f'<span class="model-chip">{m}</span>' for m in dx["contributing"])
    st.markdown(f'<p style="margin-bottom:0.6rem;">Ensemble từ: {chip_html}</p>', unsafe_allow_html=True)

    # Vietnamese report
    st.markdown(f'<div class="vi-report">{dx["vi_report"]}</div>', unsafe_allow_html=True)

    # English report (collapsed)
    with st.expander("📄 Xem báo cáo gốc tiếng Anh (English Source)", expanded=False):
        st.text_area(
            label="English Report",
            value=dx["en_report"],
            height=120,
            disabled=True,
            key="en_report_display",
            label_visibility="collapsed",
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section 2: Disease Probability Bars ─────────────────────────────────────
    st.markdown('<div class="section-card"><div class="section-title">📊 Xác suất Bệnh lý (14 Nhóm CheXpert)</div>', unsafe_allow_html=True)

    # Legend
    st.markdown("""
<div style="display:flex; gap:12px; margin-bottom:1rem; flex-wrap:wrap;">
  <span><span style="background:#16A34A; border-radius:3px; padding:1px 8px; color:white; font-size:0.75rem;">■</span> &lt;25%</span>
  <span><span style="background:#EAB308; border-radius:3px; padding:1px 8px; color:white; font-size:0.75rem;">■</span> 25–49%</span>
  <span><span style="background:#F97316; border-radius:3px; padding:1px 8px; color:white; font-size:0.75rem;">■</span> 50–74%</span>
  <span><span style="background:#DC2626; border-radius:3px; padding:1px 8px; color:white; font-size:0.75rem;">■</span> ≥75%</span>
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
                "Họ tên Bác sĩ",
                value="BS. Chẩn đoán Hình ảnh",
                key="doctor_name",
            )
        with col_dr2:
            ai_evaluation = st.radio(
                "Đánh giá kết quả AI",
                options=["✅ Chính xác", "✏️ Cần chỉnh sửa", "❌ Sai lệch"],
                index=0,
                horizontal=True,
                key="ai_eval",
            )

        verified_diseases = st.multiselect(
            "Danh mục bệnh lý xác nhận (mặc định: xác suất AI ≥ 35%)",
            options=VI_14_LABELS,
            default=dx["default_diseases"],
            key="verified_diseases",
        )

        doctor_notes = st.text_area(
            "Chú thích lâm sàng & vị trí tổn thương",
            placeholder="Nhập nhận định bổ sung: vị trí, kích thước, mức độ tổn thương...",
            height=90,
            key="doctor_notes",
        )

        final_conclusion = st.text_area(
            "Kết luận chẩn đoán cuối cùng",
            value=dx["vi_report"],
            height=120,
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
            st.success("✅ Đã lưu hồ sơ thẩm định thành công.")
        except Exception as ex:
            st.warning(f"⚠️ Lưu CSV thất bại: {ex}")

        # Generate certificate
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
            data=cert_text.encode("utf-8"),
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
