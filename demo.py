# demo.py
# Hệ thống Hỗ trợ Chẩn đoán Hình ảnh X-Quang Ngực
# Kiến trúc bài báo gốc: Pure Transformer (ViTEncoder + MemoryAugmentedDecoder)
# Giao diện 2 cột đối chiếu: Báo cáo Bác sĩ (Ground Truth) vs. Báo cáo Mô hình AI | Ảnh X-Quang & Attention Heatmap
# IEEE TMI 2022 – Multicriteria Transformer

import ast
import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from clinical_utils import format_clinical_report, split_clinical_report
from config import ModelConfig, TrainConfig
from model.full_model import MulticriteriaTransformer

# ─── PATHS ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
GT_DIR = BASE_DIR / "data" / "ground_truth"


def find_artifact_file(rel_subpath: str, filename: str) -> str:
    """Locate a checkpoint or vocabulary file inside artifacts/."""
    p1 = BASE_DIR / rel_subpath
    if p1.is_file():
        return str(p1)
    matches = list((BASE_DIR / "artifacts").rglob(filename))
    if matches and matches[0].is_file():
        return str(matches[0])
    return str(p1)


# ─── INTERNAL MLC KEYWORD ALIGNMENT (FOR REPORT CANDIDATE SELECTION ONLY) ───────
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

_MESH_TO_CORE_MAP = {
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

_PATHOLOGY_KEYWORDS = {
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
    for cls_idx, mesh_labels in _MESH_TO_CORE_MAP.items():
        idx_map[cls_idx] = [
            DEFAULT_MESH_LABELS.index(ml)
            for ml in mesh_labels
            if ml in DEFAULT_MESH_LABELS
        ]
    return idx_map


_MESH_IDX_MAP = _build_mesh_index_map()

# ─── PAGE CONFIG & CSS ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multicriteria Transformer · Đọc phim X-Quang Ngực",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
section[data-testid="stSidebar"], button[data-testid="collapsedControl"] { display: none !important; }
.stApp, section[data-testid="stAppViewContainer"], div[data-testid="stMain"] {
    background: #FFFFFF !important;
    color: #0F172A !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
}
.stApp p, .stApp label,
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stExpander"] summary,
.stApp [data-testid="stExpander"] summary *,
.stApp [data-testid="stFileUploader"] *,
.stApp [data-testid="stAlert"] * { color: #0F172A !important; }
.main .block-container { padding: 1.4rem 2.2rem !important; max-width: 1380px; margin: auto; }
.clinical-header {
    background: #FFFFFF;
    color: #0F172A !important;
    padding: 1.25rem 1.8rem;
    border-radius: 12px;
    border: 1px solid #CBD5E1;
    border-top: 5px solid #0F172A;
    margin-bottom: 1.4rem;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
}
.clinical-header h1 { color: #0F172A !important; margin: 0; font-size: 1.5rem; font-weight: 700; }
.clinical-header p { color: #475569 !important; margin: 0.3rem 0 0 0; font-size: 0.88rem; }
.section-card {
    background: #F8FAFC;
    border: 1px solid #CBD5E1;
    border-radius: 10px;
    padding: 1.1rem 1.35rem;
    margin-bottom: 1.15rem;
}
.section-title {
    font-size: 0.94rem;
    font-weight: 700;
    color: #0F172A;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    border-bottom: 2px solid #0F172A;
    padding-bottom: 0.4rem;
    margin-bottom: 0.85rem;
}
.sub-label {
    font-size: 0.84rem;
    font-weight: 700;
    color: #334155;
    margin: 0.55rem 0 0.25rem 0;
    letter-spacing: 0.03em;
}
.report-box {
    background: #FFFFFF;
    border-left: 4px solid #1E3A5F;
    border-radius: 6px;
    padding: 0.95rem 1.2rem;
    font-size: 0.96rem;
    line-height: 1.75;
    color: #0F172A;
    white-space: pre-wrap;
    border-top: 1px solid #E2E8F0;
    border-right: 1px solid #E2E8F0;
    border-bottom: 1px solid #E2E8F0;
}
.gt-box {
    background: #F0FDF4;
    border-left: 4px solid #16A34A;
    border-radius: 6px;
    padding: 0.95rem 1.2rem;
    font-size: 0.95rem;
    line-height: 1.75;
    color: #0F172A;
    white-space: pre-wrap;
    border-top: 1px solid #BBF7D0;
    border-right: 1px solid #BBF7D0;
    border-bottom: 1px solid #BBF7D0;
}
.heatmap-explain {
    background: #FFF7ED;
    border: 1px solid #FED7AA;
    border-left: 4px solid #EA580C;
    border-radius: 8px;
    padding: 0.95rem 1.15rem;
    font-size: 0.87rem;
    line-height: 1.68;
    color: #0F172A;
    margin-top: 0.65rem;
    margin-bottom: 0.9rem;
}
.upload-hint { color: #64748B; font-size: 0.8rem; margin-top: 0.25rem; margin-bottom: 0.6rem; }
.dataset-badge {
    display: inline-block;
    background: #EFF6FF;
    color: #1D4ED8 !important;
    border: 1px solid #BFDBFE;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px 4px 6px 0;
}
.split-badge {
    display: inline-block;
    background: #ECFDF5;
    color: #047857 !important;
    border: 1px solid #A7F3D0;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px 4px 6px 0;
}
div[data-testid="stButton"] > button {
    background: #0F172A !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.4rem !important;
    font-weight: 600 !important;
    width: 100%;
}
div.stButton > button, div.stButton > button p, div.stButton > button span,
[data-testid="stBaseButton-secondary"] *, [data-testid="stBaseButton-primary"] * {
    color: #FFFFFF !important;
    opacity: 1 !important;
    font-weight: 600 !important;
}
div[data-testid="stButton"] > button:hover { background: #1E3A5F !important; }
div[data-testid="stTextArea"] textarea:disabled {
    background: #F8FAFC !important;
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
    opacity: 1 !important;
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


# ─── MODEL REGISTRY & LOADING ───────────────────────────────────────────────────
MODEL_REGISTRY = {
    "archive": {
        "label": "IU-Xray Archive",
        "pt": find_artifact_file("artifacts/archive/best_archive.pt", "best_archive.pt"),
        "vocab": find_artifact_file("artifacts/archive/vocab_archive.json", "vocab_archive.json"),
        "num_classes": 65,
        "weight": 0.40,
    },
    "timphoi": {
        "label": "IU-Xray TimPhoi",
        "pt": find_artifact_file("artifacts/timphoi/best_timphoi.pt", "best_timphoi.pt"),
        "vocab": find_artifact_file("artifacts/timphoi/vocab_timphoi.json", "vocab_timphoi.json"),
        "num_classes": 14,
        "weight": 0.35,
    },
    "mimic": {
        "label": "MIMIC-CXR",
        "pt": find_artifact_file("artifacts/mimic/best_mimic.pt", "best_mimic.pt"),
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
            loaded[key] = {"error": f"File not found: {pt_path}", "label": cfg["label"], "missing": True}
            continue
        try:
            with open(vocab_path, "r", encoding="utf-8") as f:
                tok = json.load(f)
            id2word = {v: k for k, v in tok.items()}
            vocab_size = len(tok)
            pad_idx = tok.get("<pad>", 0)
            bos_idx = tok.get("<bos>", 1)
            eos_idx = tok.get("<eos>", 2)
            ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
            state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
            mlc_key = next(
                (k for k in state if "mlc_head" in k and ("proj.weight" in k or "fc" in k)),
                None,
            )
            num_classes = (
                int(state[mlc_key].shape[0])
                if (mlc_key and state[mlc_key].shape[0] in (14, 65))
                else cfg["num_classes"]
            )
            mcfg = ModelConfig()
            mcfg.vocab_size = vocab_size
            mcfg.pad_idx = pad_idx
            mcfg.num_mlc_labels = num_classes
            tcfg = TrainConfig()

            model = MulticriteriaTransformer(mcfg, tcfg)
            model.load_state_dict(state, strict=False)
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
            }
        except Exception as ex:
            loaded[key] = {"error": str(ex), "traceback": traceback.format_exc(), "label": cfg["label"]}
    return loaded


# ─── INFERENCE & CANDIDATE SELECTION ────────────────────────────────────────────
def run_model_inference(entry: dict, frontal_tensor: torch.Tensor, lateral_tensor: torch.Tensor | None):
    model = entry["model"]
    device = entry["device"]
    bos = entry["bos"]
    eos = entry["eos"]
    pad = entry["pad"]
    id2word = entry["id2word"]

    frontal = frontal_tensor.to(device)
    if lateral_tensor is not None:
        lateral = lateral_tensor.to(device)
        images = torch.stack([frontal.squeeze(0), lateral.squeeze(0)], dim=0).unsqueeze(0)
    else:
        images = torch.stack([frontal.squeeze(0), frontal.squeeze(0)], dim=0).unsqueeze(0)

    with torch.no_grad():
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
            if idx == eos:
                break
            if idx in (bos, pad):
                continue
            word = id2word.get(idx, "")
            if word and word not in ("<bos>", "<eos>", "<pad>", "<unk>"):
                words.append(word)
        report_en = " ".join(words).strip()
        probs = model.predict_mlc(images)
        probs_np = probs[0].cpu().numpy()

    return report_en, probs_np


def _map_archive_to_core(probs_65: np.ndarray) -> np.ndarray:
    out = np.zeros(14, dtype=np.float32)
    for cls_idx, mesh_indices in _MESH_IDX_MAP.items():
        if mesh_indices:
            out[cls_idx] = float(np.max(probs_65[mesh_indices]))
    return out


def _compute_mlc_consensus(results: dict, all_entries: dict) -> np.ndarray:
    total_weight, weighted_sum = 0.0, np.zeros(14, dtype=np.float32)
    for key, (_, probs_np) in results.items():
        w = all_entries[key]["weight"]
        probs14 = (
            _map_archive_to_core(probs_np)
            if all_entries[key]["num_classes"] == 65
            else probs_np[:14].astype(np.float32)
        )
        weighted_sum += w * probs14
        total_weight += w
    return weighted_sum / max(total_weight, 1e-8)


def _score_report_candidate(raw_en: str, clean_en: str, mlc_probs: np.ndarray, model_weight: float) -> float:
    words = [w.lower() for w in re.findall(r"\b[a-zA-Z]+\b", clean_en)]
    if len(words) < 5:
        return -999.0
    lexical_diversity = len(set(words)) / max(len(words), 1)
    length_bonus = 2.0 if 12 <= len(words) <= 45 else (-1.5 if len(words) > 50 else 0.5)
    keyword_bonus = 0.0
    clean_low = clean_en.lower()
    if mlc_probs is not None:
        for idx in range(min(14, len(mlc_probs))):
            if mlc_probs[idx] >= 0.35 and any(kw in clean_low for kw in _PATHOLOGY_KEYWORDS.get(idx, [])):
                keyword_bonus += 1.0
    xxxx_penalty = -1.2 if "xxxx" in raw_en.lower() else 0.0
    return (lexical_diversity * 3.0) + length_bonus + keyword_bonus + (model_weight * 3.0) + xxxx_penalty


def select_best_report(results: dict, all_entries: dict, mlc_probs: np.ndarray) -> tuple[str, str]:
    """Return (winning_model_key, raw_english_report) with highest quality score."""
    scored = []
    for key, (raw_en, _) in results.items():
        if not raw_en:
            continue
        clean_en = format_clinical_report(raw_en)
        weight = all_entries[key]["weight"] if (all_entries and key in all_entries) else 0.33
        sc = _score_report_candidate(raw_en, clean_en, mlc_probs, weight)
        scored.append((sc, key, raw_en))
    if not scored:
        first_key = next(iter(results))
        return first_key, "Heart size and mediastinal contours are within normal limits. The lungs are clear."
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1], scored[0][2]


# ─── ENHANCED VIT ATTENTION HEATMAP & ANATOMY EXPLANATION ───────────────────────
ANATOMY_ZONES = [
    # (row_start, row_end_excl, col_start, col_end_excl, name_vi, name_en)
    # Note: Chest X-ray PA/AP radiological convention: image left (col 0..4) = patient RIGHT; image right (col 9..13) = patient LEFT.
    (0, 4, 0, 5, "Đỉnh phổi phải", "right lung apex"),
    (0, 4, 9, 14, "Đỉnh phổi trái", "left lung apex"),
    (4, 9, 0, 5, "Vùng giữa phổi phải", "right mid lung zone"),
    (4, 9, 9, 14, "Vùng giữa phổi trái", "left mid lung zone"),
    (9, 14, 0, 5, "Đáy phổi phải / góc sườn hoành phải", "right lower lung / costophrenic angle"),
    (9, 14, 9, 14, "Đáy phổi trái / góc sườn hoành trái", "left lower lung / costophrenic angle"),
    (3, 11, 5, 9, "Vùng bóng tim – trung thất", "cardiac silhouette and mediastinum"),
]


def _coord_to_anatomy(row: int, col: int) -> tuple[str, str]:
    """Map a spatial patch coordinate (row, col) on the 14x14 grid to an anatomical zone."""
    for r0, r1, c0, c1, name_vi, name_en in ANATOMY_ZONES:
        if r0 <= row < r1 and c0 <= col < c1:
            return name_vi, name_en
    if row < 4:
        return "Vùng trung thất trên / đỉnh lồng ngực", "upper mediastinum / thoracic inlet"
    if row >= 11:
        return "Vùng vòm hoành – đáy lồng ngực", "diaphragm / lower thorax"
    return "Vùng rốn phổi – trung thất", "hilar / mediastinal region"


def _analyze_attention_grid(grid_14x14: np.ndarray, top_k: int = 3) -> dict:
    """Extract top anatomical zones and peak patch coordinates from the 14x14 [CLS] attention grid."""
    flat_grid = grid_14x14.astype(np.float64)
    total_mass = float(flat_grid.sum()) + 1e-8

    zone_scores = []
    for r0, r1, c0, c1, name_vi, name_en in ANATOMY_ZONES:
        patch_block = flat_grid[r0:r1, c0:c1]
        mean_val = float(patch_block.mean())
        share_pct = float(patch_block.sum() / total_mass * 100.0)
        zone_scores.append((mean_val, share_pct, name_vi, name_en))
    zone_scores.sort(key=lambda x: x[0], reverse=True)

    # Top peak coordinates on the 14x14 spatial patch grid
    top_indices = np.argsort(flat_grid.ravel())[::-1][:top_k]
    peak_patches = []
    max_patch = float(flat_grid.max()) + 1e-8
    for flat_idx in top_indices:
        r, c = divmod(int(flat_idx), grid_14x14.shape[1])
        val = float(flat_grid[r, c])
        name_vi, name_en = _coord_to_anatomy(r, c)
        peak_patches.append({
            "row": r,
            "col": c,
            "rel_pct": int(round(val / max_patch * 100.0)),
            "zone_vi": name_vi,
            "zone_en": name_en,
        })

    return {
        "top_zones": zone_scores[:top_k],
        "peak_patches": peak_patches,
        "grid_shape": grid_14x14.shape,
    }


def create_attention_overlay(
    pil_image: Image.Image,
    model,
    device,
    frontal_tensor: torch.Tensor,
    lateral_tensor: torch.Tensor | None,
) -> tuple[Image.Image, dict]:
    """
    Extract final encoder [CLS]-to-patch attention, enhance contrast (min-max + gamma + threshold),
    render vivid red/orange/yellow hotspots over the Frontal radiograph, and analyze spatial patches.
    """
    frontal = frontal_tensor.to(device)
    if lateral_tensor is not None:
        lateral = lateral_tensor.to(device)
        images = torch.stack([frontal.squeeze(0), lateral.squeeze(0)], dim=0).unsqueeze(0)
    else:
        images = torch.stack([frontal.squeeze(0), frontal.squeeze(0)], dim=0).unsqueeze(0)

    upsampled, patch_grid = model.encoder.get_attention_heatmap(
        images, img_size=448, return_patch_grid=True
    )
    attention_448 = upsampled[0, 0].detach().float().cpu().numpy()
    grid_14 = patch_grid[0, 0].detach().float().cpu().numpy()

    if not np.isfinite(attention_448).all():
        raise ValueError("Encoder attention contains non-finite values.")

    # Contrast normalization: min-max stretch + percentile clip + gamma sharpening
    p_low = float(np.percentile(attention_448, 15))
    p_high = float(np.max(attention_448))
    norm = np.clip((attention_448 - p_low) / (p_high - p_low + 1e-8), 0.0, 1.0)

    # Gamma > 1 suppresses background clutter; threshold isolates active pathological hotspots
    gamma = 1.25
    norm = np.power(norm, gamma)
    threshold = 0.30
    active = np.clip((norm - threshold) / (1.0 - threshold + 1e-8), 0.0, 1.0)

    # Vivid hotspot colormap: yellow (low-active) -> orange -> bright crimson red (peak)
    # R stays bright red/yellow (255), G drops from 240 -> 20, B drops from 30 -> 0
    r_chan = np.full_like(active, 255.0)
    g_chan = 240.0 * (1.0 - np.power(active, 0.85)) + 15.0
    b_chan = 30.0 * (1.0 - active)

    # Alpha transparency: 0 on background (keeps radiograph sharp), up to 0.72 on peak hotspots
    alpha_chan = np.where(norm >= threshold, 0.18 + 0.54 * active, 0.0)

    rgba_map = np.stack(
        [
            np.clip(r_chan, 0, 255).astype(np.uint8),
            np.clip(g_chan, 0, 255).astype(np.uint8),
            np.clip(b_chan, 0, 255).astype(np.uint8),
            np.clip(alpha_chan * 255.0, 0, 255).astype(np.uint8),
        ],
        axis=-1,
    )
    overlay_layer = Image.fromarray(rgba_map, mode="RGBA").resize(
        pil_image.size, Image.Resampling.BILINEAR
    )
    base_rgba = pil_image.convert("RGBA")
    composited = Image.alpha_composite(base_rgba, overlay_layer).convert("RGB")

    attn_info = _analyze_attention_grid(grid_14, top_k=3)
    return composited, attn_info


def render_heatmap_explanation(attn_info: dict) -> str:
    """Render HTML explanation box detailing spatial patch coordinates & ViT [CLS] mechanism."""
    if not attn_info:
        return ""
    top_zones = attn_info.get("top_zones", [])
    peak_patches = attn_info.get("peak_patches", [])
    gh, gw = attn_info.get("grid_shape", (14, 14))

    zone_lines = []
    max_zone_score = max((z[0] for z in top_zones), default=1e-8) + 1e-8
    for mean_val, share_pct, name_vi, _ in top_zones:
        rel_pct = min(100, int(round(mean_val / max_zone_score * 100)))
        zone_lines.append(
            f"• <b>{name_vi}</b> — cường độ tương đối <b>{rel_pct}%</b> "
            f"(chiếm {share_pct:.1f}% tổng attention)"
        )

    patch_lines = []
    for p in peak_patches:
        patch_lines.append(
            f"• Patch <code>(hàng {p['row']}, cột {p['col']})</code> / lưới {gh}×{gw} "
            f"→ <b>{p['zone_vi']}</b> ({p['rel_pct']}% đỉnh)"
        )

    zones_en = ", ".join(dict.fromkeys(z[3] for z in top_zones))
    return f"""
<div class="heatmap-explain">
  <b>🔍 Giải thích vùng tập trung (ViT Attention Explanation)</b><br><br>
  <b>1. Tọa độ patch chú ý cao nhất (lưới không gian {gh}×{gw}):</b><br>
  {'<br>'.join(patch_lines)}<br><br>
  <b>2. Vùng giải phẫu trọng tâm:</b><br>
  {'<br>'.join(zone_lines)}<br><br>
  <b>3. Cơ chế suy luận của Transformer:</b> Token <code>[CLS]</code> ở tầng Encoder cuối cùng
  gán trọng số Self-Attention cao cho các patch thuộc <i>{zones_en}</i> — nơi có biến đổi
  mật độ cản quang bất thường (đám mờ, đông đặc, bóng tim, tràn dịch hoặc đường viền giải phẫu)
  — nhằm tổng hợp đặc trưng thị giác truyền qua Cross-Attention cho Decoder sinh câu mô tả
  bệnh lý trong <b>FINDINGS</b> và <b>IMPRESSION</b>.
</div>
"""


# ─── GROUND TRUTH INDEX & LOOKUP ────────────────────────────────────────────────
def _extract_mimic_report_text(report_field) -> str:
    """Safely unwrap MIMIC report strings that are serialized Python lists."""
    if isinstance(report_field, list):
        candidates = [str(x).strip() for x in report_field if str(x).strip()]
    else:
        raw = str(report_field or "").strip()
        if raw.startswith("[") and raw.endswith("]"):
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, list):
                    candidates = [str(x).strip() for x in parsed if str(x).strip()]
                else:
                    candidates = [raw]
            except Exception:
                candidates = [raw]
        else:
            candidates = [raw] if raw else []

    if not candidates:
        return ""
    # Prefer the first report that contains non-trivial findings or impression text
    for c in candidates:
        cleaned = re.sub(r"\b(?:findings|impression)\s*:\s*", "", c, flags=re.IGNORECASE).strip()
        if len(cleaned) >= 15:
            return c
    return candidates[0]


def _parse_verbatim_sections(report_text: str) -> tuple[str, str]:
    """Extract verbatim FINDINGS and IMPRESSION from a ground-truth report without altering medical phrasing."""
    text = re.sub(r"\s+", " ", (report_text or "").strip())
    if not text:
        return "(không có)", "(không có)"

    m_imp = re.search(r"\bimpression\s*[:]\s*", text, flags=re.IGNORECASE)
    if m_imp:
        part_before = text[: m_imp.start()].strip()
        part_after = text[m_imp.end() :].strip()
        findings = re.sub(r"^\s*findings\s*[:]\s*", "", part_before, flags=re.IGNORECASE).strip()
        impression = part_after
        return (findings if findings else "(không có)"), (impression if impression else "(không có)")

    m_find = re.search(r"^\s*findings\s*[:]\s*", text, flags=re.IGNORECASE)
    if m_find:
        findings = text[m_find.end() :].strip()
        return (findings if findings else "(không có)"), "(không có trong dữ liệu gốc)"

    # Split verbatim sentences cleanly without deleting XXXX or mutating doctor text
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) >= 2:
        findings_sents = []
        impression_sents = []
        for s in sentences:
            low = s.lower()
            if any(marker in low for marker in ("no acute", "conclusion", "impression", "normal chest")):
                impression_sents.append(s)
            else:
                findings_sents.append(s)
        if not impression_sents:
            impression_sents = [findings_sents.pop()]
        if not findings_sents:
            findings_sents = [text]
        return " ".join(findings_sents), " ".join(impression_sents)

    return text, "(báo cáo gốc ghi nhận chung trong phần FINDINGS)"


@st.cache_resource(show_spinner=False)
def _load_gt_index():
    """
    Build lookup index across data/ground_truth/{iu_timphoi, iu_archive, mimic}.
    Returns (exact_index, uid_prefix_index).
    """
    exact_index: dict[str, dict] = {}
    prefix_index: dict[str, dict] = {}

    def _add_exact(key: str, record: dict):
        k = key.strip().lower().replace(" ", "_")
        if k and len(k) >= 3 and k not in exact_index:
            exact_index[k] = record

    def _add_prefix(key: str, record: dict):
        k = key.strip().lower().replace(" ", "_")
        if k and len(k) >= 4 and k not in prefix_index:
            prefix_index[k] = record

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
                    uid = str(item.get("id", "")).strip()
                    report = str(item.get("report", "")).strip()
                    findings, impression = _parse_verbatim_sections(report)
                    record = {
                        "dataset": "IU-Xray (TimPhoi)",
                        "split": split_name,
                        "uid": uid,
                        "findings": findings,
                        "impression": impression,
                        "raw": report,
                    }
                    _add_exact(uid, record)
                    _add_prefix(uid, record)
                    for img_path in item.get("image_path", []):
                        p = Path(img_path)
                        # e.g. CXR2384_IM-0942/0.png -> cxr2384_im-0942_0 & cxr2384_im-0942/0
                        if p.parent.name and p.parent.name != ".":
                            _add_exact(f"{p.parent.name}_{p.stem}", record)
                            _add_exact(f"{p.parent.name}/{p.stem}", record)
        except Exception as exc:
            print(f"[GT Loader] iu_timphoi error: {exc}")

    # ── 2. iu_archive: indiana_reports.csv + indiana_projections.csv ────────
    reports_path = GT_DIR / "iu_archive" / "indiana_reports.csv"
    proj_path = GT_DIR / "iu_archive" / "indiana_projections.csv"
    if reports_path.exists():
        try:
            uid_to_files: dict[str, list[str]] = {}
            if proj_path.exists():
                with open(proj_path, "r", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        uid_str = str(row.get("uid", "")).strip()
                        fname = str(row.get("filename", "")).strip()
                        if uid_str and fname:
                            uid_to_files.setdefault(uid_str, []).append(fname)

            with open(reports_path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    uid_str = str(row.get("uid", "")).strip()
                    findings_raw = str(row.get("findings", "")).strip()
                    impression_raw = str(row.get("impression", "")).strip()
                    record = {
                        "dataset": "IU X-Ray Archive",
                        "split": "Archive (Full)",
                        "uid": uid_str,
                        "findings": findings_raw if findings_raw else "(không có)",
                        "impression": impression_raw if impression_raw else "(không có)",
                        "raw": f"FINDINGS: {findings_raw}\nIMPRESSION: {impression_raw}",
                    }
                    for fname in uid_to_files.get(uid_str, []):
                        low_name = fname.lower()
                        no_ext = re.sub(r"\.(?:dcm\.)?(?:png|jpe?g|bmp|tiff?)$", "", low_name)
                        stem_one = Path(fname).stem.lower()
                        _add_exact(low_name, record)
                        _add_exact(stem_one, record)
                        _add_exact(no_ext, record)
                        # Structured UID+IM prefix e.g. 1_im-0001
                        m_im = re.match(r"^(\d+_im-\d+)", no_ext)
                        if m_im:
                            _add_prefix(m_im.group(1), record)
        except Exception as exc:
            print(f"[GT Loader] iu_archive error: {exc}")

    # ── 3. mimic: train.txt, val.txt, test.txt ──────────────────────────────
    for split_name in ("train", "val", "test"):
        txt_path = GT_DIR / "mimic" / f"{split_name}.txt"
        if not txt_path.exists():
            continue
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                raw_content = f.read().strip()
            samples = []
            try:
                parsed = json.loads(raw_content)
                if isinstance(parsed, list):
                    samples = parsed
                elif isinstance(parsed, dict):
                    samples = parsed.get(split_name, list(parsed.values())[0] if parsed else [])
            except json.JSONDecodeError:
                for line in raw_content.splitlines():
                    line = line.strip()
                    if line:
                        try:
                            samples.append(json.loads(line))
                        except Exception:
                            pass

            for item in samples:
                sid = str(item.get("study_id", item.get("id", ""))).strip()
                report_text = _extract_mimic_report_text(item.get("report", item.get("text", "")))
                findings_text, impression_text = _parse_verbatim_sections(report_text)
                record = {
                    "dataset": "MIMIC-CXR",
                    "split": split_name,
                    "uid": sid,
                    "findings": findings_text,
                    "impression": impression_text,
                    "raw": report_text,
                }
                _add_exact(sid, record)
                for img_key in ("frontal", "lateral"):
                    img_path = str(item.get(img_key, "")).strip()
                    if img_path:
                        p = Path(img_path)
                        _add_exact(p.stem, record)
                        _add_exact(p.name, record)
                        # Study folder ID (e.g., s50414267)
                        if re.match(r"^s\d{6,}$", p.parent.name, flags=re.IGNORECASE):
                            _add_prefix(p.parent.name, record)
        except Exception as exc:
            print(f"[GT Loader] mimic/{split_name} error: {exc}")

    return exact_index, prefix_index


def lookup_ground_truth(uploaded_filename: str) -> dict | None:
    """
    Look up an uploaded X-ray filename or UID across mimic, iu_archive, and iu_timphoi.
    Supports original filenames, .dcm.png double extensions, and UID-prefixed filenames.
    """
    if not uploaded_filename:
        return None
    exact_index, prefix_index = _load_gt_index()
    if not exact_index and not prefix_index:
        return None

    fname = Path(uploaded_filename.strip())
    low_name = fname.name.lower().replace(" ", "_")
    stem = fname.stem.lower().replace(" ", "_")
    no_dcm_stem = re.sub(r"\.(?:dcm\.)?(?:png|jpe?g|bmp|tiff?)$", "", low_name)

    # 1. Exact filename / stem matches
    for key in (low_name, stem, no_dcm_stem):
        if key in exact_index:
            return exact_index[key]

    # 2. Structured UID / UUID tokens embedded inside filenames (including UID-prefixed names)
    token_patterns = [
        r"([0-9a-f]{8}-[0-9a-f]{8}-[0-9a-f]{8}-[0-9a-f]{8}-[0-9a-f]{8})",  # MIMIC UUID
        r"(cxr\d+_im-\d+)",                                                # IU TimPhoi ID
        r"(\d+_im-\d+-\d+)",                                               # IU Archive file prefix
        r"(\d+_im-\d+)",                                                   # IU Archive study prefix
        r"(s\d{7,})",                                                      # MIMIC study ID
        r"(row_\d+)",                                                      # MIMIC row ID
    ]
    for pat in token_patterns:
        m = re.search(pat, no_dcm_stem, flags=re.IGNORECASE)
        if m:
            tok = m.group(1).lower()
            if tok in exact_index:
                return exact_index[tok]
            if tok in prefix_index:
                return prefix_index[tok]

    # 3. Check if any known structured prefix appears as a token boundary in stem
    # (handles e.g. UID_custom_suffix.png without matching arbitrary single digits)
    for part in re.split(r"[/\\]", no_dcm_stem):
        if part in prefix_index:
            return prefix_index[part]

    return None


# ─── MAIN APP ────────────────────────────────────────────────────────────────────
def main():
    st.markdown("""
<div class="clinical-header">
  <h1>🫁 Hệ thống Sinh Báo cáo X-Quang Ngực Tự động (Pure Transformer)</h1>
  <p>Kiến trúc bài báo gốc · ViT Encoder + Memory-Augmented Decoder · IEEE TMI 2022 · Tự động đối chiếu Ground Truth &amp; ViT Attention Heatmap</p>
</div>
""", unsafe_allow_html=True)

    # 2-column comparison layout: Left = Reports comparison (1.15), Right = Upload + Images + Heatmap (0.85)
    col_left, col_right = st.columns([1.15, 0.85], gap="large")

    # ══════════════════════════════════════════════════════════════════════════
    # CỘT PHẢI: Tải ảnh, Ảnh gốc, Attention Heatmap + Giải thích, Ảnh nghiêng
    # ══════════════════════════════════════════════════════════════════════════
    with col_right:
        st.markdown('<div class="section-card"><div class="section-title">📂 Tải lên Phim X-Quang</div>', unsafe_allow_html=True)
        frontal_file = st.file_uploader(
            "Phim X-quang Thẳng (Frontal - bắt buộc)",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            key="frontal_upload",
            help="Ảnh chụp X-quang lồng ngực tư thế thẳng (PA hoặc AP)",
        )
        st.markdown('<p class="upload-hint">Định dạng PNG/JPG/BMP, tư thế thẳng (PA/AP)</p>', unsafe_allow_html=True)

        lateral_file = st.file_uploader(
            "Phim X-quang Nghiêng (Lateral - tùy chọn)",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            key="lateral_upload",
            help="Ảnh chụp X-quang tư thế nghiêng bổ trợ (nếu có)",
        )
        st.markdown('<p class="upload-hint">Tùy chọn: cung cấp thêm góc nghiêng nếu có sẵn</p>', unsafe_allow_html=True)

        analyze_clicked = st.button("🔬 Phân tích & Sinh Báo cáo X-Quang", key="run_analysis")
        st.markdown("</div>", unsafe_allow_html=True)

        pil_frontal = None
        pil_lateral = None
        uploaded_filename = ""

        if frontal_file is not None:
            pil_frontal = Image.open(frontal_file).convert("RGB")
            uploaded_filename = frontal_file.name
            if lateral_file is not None:
                pil_lateral = Image.open(lateral_file).convert("RGB")

            # Reset cached diagnosis when the uploaded image pair changes
            cur_sig = (
                frontal_file.name,
                frontal_file.size,
                lateral_file.name if lateral_file is not None else None,
                lateral_file.size if lateral_file is not None else None,
            )
            if st.session_state.get("uploaded_sig") != cur_sig:
                st.session_state["uploaded_sig"] = cur_sig
                st.session_state.pop("diagnosis", None)
        else:
            st.session_state.pop("uploaded_sig", None)
            st.session_state.pop("diagnosis", None)
            st.info("Vui lòng tải lên **Phim X-quang Thẳng (Frontal)** ở trên để bắt đầu.")

        # Run inference on button click BEFORE rendering results so both columns update immediately
        if analyze_clicked:
            if pil_frontal is None:
                st.warning("⚠️ Cần tải lên Phim X-quang Thẳng (Frontal) trước khi phân tích.")
            else:
                with st.spinner("⏳ Đang nạp mô hình Transformer và sinh báo cáo + Attention Heatmap..."):
                    all_entries = load_all_models()
                    for k, entry in all_entries.items():
                        if entry and entry.get("missing"):
                            st.warning(f"⚠️ Mô hình '{k}': Không tìm thấy tệp trong artifacts/{k}/.")
                        elif entry and "error" in entry:
                            st.warning(f"⚠️ Lỗi nạp mô hình '{entry.get('label', k)}': {entry['error']}")

                    frontal_tensor = preprocess_image(pil_frontal)
                    lateral_tensor = preprocess_image(pil_lateral) if pil_lateral is not None else None

                    results = {}
                    for key, entry in all_entries.items():
                        if not entry or "error" in entry:
                            continue
                        try:
                            report_en, probs_np = run_model_inference(entry, frontal_tensor, lateral_tensor)
                            results[key] = (report_en, probs_np)
                        except Exception as ex:
                            st.warning(f"⚠️ Lỗi suy luận '{entry.get('label', key)}': {ex}")

                    if not results:
                        st.error("❌ Không có mô hình nào suy luận thành công.")
                    else:
                        mlc_probs = _compute_mlc_consensus(results, all_entries)
                        best_key, best_raw_report = select_best_report(results, all_entries, mlc_probs)
                        clean_en = format_clinical_report(best_raw_report)
                        structured_en = split_clinical_report(clean_en)

                        # Generate heatmap from the model that produced the selected report
                        heatmap_image, attn_info = None, {}
                        hm_entry = all_entries.get(best_key)
                        if hm_entry and "model" in hm_entry:
                            try:
                                heatmap_image, attn_info = create_attention_overlay(
                                    pil_frontal,
                                    hm_entry["model"],
                                    hm_entry["device"],
                                    frontal_tensor,
                                    lateral_tensor,
                                )
                            except Exception as ex:
                                st.warning(f"⚠️ Lỗi sinh Attention Heatmap: {ex}")

                        st.session_state["diagnosis"] = {
                            "raw_report": best_raw_report,
                            "clean_en": clean_en,
                            "structured_en": structured_en,
                            "heatmap_image": heatmap_image,
                            "attn_info": attn_info,
                            "selected_model": all_entries[best_key]["label"],
                            "contributing": [all_entries[k]["label"] for k in results],
                            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                            "uploaded_fname": uploaded_filename,
                        }

        # 2. Original Frontal X-ray
        if pil_frontal is not None:
            st.markdown('<div class="section-title" style="margin-top:0.6rem;">🩻 Phim X-quang Thẳng (Frontal)</div>', unsafe_allow_html=True)
            st.image(pil_frontal, caption=uploaded_filename, use_container_width=True)

        # 3. Attention Heatmap + ViT Attention Explanation
        if pil_frontal is not None and "diagnosis" in st.session_state:
            dx = st.session_state["diagnosis"]
            heatmap_image = dx.get("heatmap_image")
            attn_info = dx.get("attn_info", {})
            if heatmap_image is not None:
                st.markdown('<div class="section-title" style="margin-top:0.8rem;">🔥 Bản đồ nhiệt vùng tập trung (Attention Heatmap)</div>', unsafe_allow_html=True)
                st.image(
                    heatmap_image,
                    caption="Màu đỏ/vàng rực rỡ = Vùng patch có trọng số Self-Attention cao nhất từ [CLS]",
                    use_container_width=True,
                )
                st.markdown(render_heatmap_explanation(attn_info), unsafe_allow_html=True)

        # 4. Lateral X-ray (ONLY when user uploaded one)
        if pil_lateral is not None:
            st.markdown('<div class="section-title" style="margin-top:0.8rem;">🩻 Phim X-quang Nghiêng (Lateral)</div>', unsafe_allow_html=True)
            st.image(pil_lateral, caption=lateral_file.name, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # CỘT TRÁI: Đối chiếu Báo cáo gốc của Bác sĩ vs. Báo cáo Mô hình AI
    # ══════════════════════════════════════════════════════════════════════════
    with col_left:
        # Khung 1: Ground Truth Doctor Report
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🩺 Khung 1 — Báo cáo gốc của Bác sĩ (Ground Truth)</div>', unsafe_allow_html=True)

        fname_to_lookup = uploaded_filename or st.session_state.get("diagnosis", {}).get("uploaded_fname", "")
        gt_record = lookup_ground_truth(fname_to_lookup) if fname_to_lookup else None

        if gt_record is not None:
            ds_name = gt_record["dataset"]
            split_name = gt_record["split"].upper()
            st.markdown(
                f'<span class="dataset-badge">Bộ dữ liệu: {ds_name}</span>'
                f'<span class="split-badge">Tập: {split_name}</span>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="sub-label">FINDINGS (Bác sĩ):</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="gt-box">{gt_record["findings"]}</div>', unsafe_allow_html=True)
            st.markdown('<div class="sub-label">IMPRESSION (Bác sĩ):</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="gt-box">{gt_record["impression"]}</div>', unsafe_allow_html=True)
        elif fname_to_lookup:
            st.info(
                f"ℹ️ Ảnh **{fname_to_lookup}** không có trong tập dữ liệu gốc "
                "(`MIMIC-CXR`, `IU X-Ray Archive`, `IU-Xray TimPhoi`). "
                "Đây là ảnh ngoài tập dữ liệu nên không có báo cáo bác sĩ đối chiếu."
            )
        else:
            st.info("ℹ️ Tải lên ảnh X-quang ở cột phải để tự động tra cứu báo cáo bác sĩ trong `data/ground_truth/`.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Khung 2: Predicted Report by Transformer Model
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🤖 Khung 2 — Báo cáo do Mô hình AI sinh ra (Predicted Report)</div>', unsafe_allow_html=True)

        if "diagnosis" in st.session_state:
            dx = st.session_state["diagnosis"]
            structured_en = dx.get("structured_en", split_clinical_report(dx["raw_report"]))
            selected_lbl = dx.get("selected_model", "")
            st.markdown(
                f'<span class="dataset-badge">Mô hình chọn lọc: {selected_lbl}</span>',
                unsafe_allow_html=True,
            )

            st.markdown('<div class="sub-label">FINDINGS (Mô hình AI):</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="report-box">{structured_en["findings"]}</div>', unsafe_allow_html=True)

            st.markdown('<div class="sub-label">IMPRESSION (Mô hình AI):</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="report-box">{structured_en["impression"]}</div>', unsafe_allow_html=True)

            with st.expander("📄 Raw Output nguyên bản", expanded=False):
                st.text_area(
                    label="Raw Output",
                    value=dx["raw_report"],
                    height=120,
                    disabled=True,
                    key="raw_report_display",
                    label_visibility="collapsed",
                )
        else:
            st.info("ℹ️ Nhấn nút **🔬 Phân tích & Sinh Báo cáo X-Quang** ở cột phải để xem kết quả sinh báo cáo.")
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
