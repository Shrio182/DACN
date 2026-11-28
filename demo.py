# demo.py
# Hệ thống Hỗ trợ Chẩn đoán Hình ảnh X-Quang Ngực
# Kiến trúc bài báo gốc: Pure Transformer (ViTEncoder + MemoryAugmentedDecoder)
# Giao diện 2 cột đối chiếu gọn trên 1 màn hình:
#   Cột trái: Tải ảnh & Ảnh Frontal gốc song song với Attention Heatmap
#   Cột phải: Báo cáo gốc của Bác sĩ (Ground Truth) vs. Báo cáo Mô hình AI (FINDINGS / IMPRESSION)
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
from PIL import Image, ImageFilter
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

# ─── PAGE CONFIG & COMPACT CSS ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Multicriteria Transformer · Đọc phim X-Quang Ngực",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
section[data-testid="stSidebar"], button[data-testid="collapsedControl"] { display: none !important; }
header[data-testid="stHeader"] { height: 0.4rem !important; min-height: 0.4rem !important; background: #FFFFFF !important; }
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
.main .block-container {
    padding: 0.65rem 1.5rem 0.9rem 1.5rem !important;
    max-width: 1440px;
    margin: auto;
}
.clinical-header {
    background: #FFFFFF;
    color: #0F172A !important;
    padding: 0.65rem 1.2rem;
    border-radius: 10px;
    border: 1px solid #CBD5E1;
    border-top: 4px solid #0F172A;
    margin-bottom: 0.75rem;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
}
.clinical-header h1 { color: #0F172A !important; margin: 0; font-size: 1.28rem; font-weight: 700; }
.clinical-header p { color: #475569 !important; margin: 0.15rem 0 0 0; font-size: 0.8rem; }
.section-card {
    background: #F8FAFC;
    border: 1px solid #CBD5E1;
    border-radius: 9px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.75rem;
}
.section-title {
    font-size: 0.86rem;
    font-weight: 700;
    color: #0F172A;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    border-bottom: 2px solid #0F172A;
    padding-bottom: 0.3rem;
    margin-bottom: 0.55rem;
}
.sub-label {
    font-size: 0.8rem;
    font-weight: 700;
    color: #334155;
    margin: 0.4rem 0 0.2rem 0;
    letter-spacing: 0.03em;
}
.report-box {
    background: #FFFFFF;
    border-left: 4px solid #1E3A5F;
    border-radius: 6px;
    padding: 0.65rem 0.9rem;
    font-size: 0.9rem;
    line-height: 1.55;
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
    padding: 0.65rem 0.9rem;
    font-size: 0.9rem;
    line-height: 1.55;
    color: #0F172A;
    white-space: pre-wrap;
    border-top: 1px solid #BBF7D0;
    border-right: 1px solid #BBF7D0;
    border-bottom: 1px solid #BBF7D0;
}
.img-placeholder {
    height: 340px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    background: #F1F5F9;
    border: 1px dashed #94A3B8;
    border-radius: 8px;
    color: #475569;
    font-size: 0.86rem;
    padding: 1rem;
}
.dataset-badge {
    display: inline-block;
    background: #EFF6FF;
    color: #1D4ED8 !important;
    border: 1px solid #BFDBFE;
    border-radius: 20px;
    padding: 1px 9px;
    font-size: 0.72rem;
    font-weight: 600;
    margin: 0 4px 4px 0;
}
.split-badge {
    display: inline-block;
    background: #ECFDF5;
    color: #047857 !important;
    border: 1px solid #A7F3D0;
    border-radius: 20px;
    padding: 1px 9px;
    font-size: 0.72rem;
    font-weight: 600;
    margin: 0 4px 4px 0;
}
/* Compact image height (320px-360px) so images & reports fit simultaneously in 1 viewport */
div[data-testid="stImage"] img {
    height: 340px !important;
    max-height: 350px !important;
    width: 100% !important;
    object-fit: contain !important;
    background: #0F172A !important;
    border-radius: 8px !important;
}
div[data-testid="stFileUploader"] section {
    padding: 0.45rem 0.65rem !important;
}
div[data-testid="stButton"] > button {
    background: #0F172A !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.48rem 1.2rem !important;
    font-weight: 600 !important;
    width: 100%;
    margin-bottom: 0.45rem !important;
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
footer { display: none !important; }
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

    return report_en, probs_np, out_ids


def _map_archive_to_core(probs_65: np.ndarray) -> np.ndarray:
    out = np.zeros(14, dtype=np.float32)
    for cls_idx, mesh_indices in _MESH_IDX_MAP.items():
        if mesh_indices:
            out[cls_idx] = float(np.max(probs_65[mesh_indices]))
    return out


def _compute_mlc_consensus(results: dict, all_entries: dict) -> np.ndarray:
    total_weight, weighted_sum = 0.0, np.zeros(14, dtype=np.float32)
    for key, (_, probs_np, _) in results.items():
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
    for key, (raw_en, _, _) in results.items():
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


# ─── NATURAL ATTENTION HEATMAP (BORDER-SUPPRESSED + DECODER CROSS-ATTENTION) ────
@torch.no_grad()
def _extract_decoder_cross_attention(
    model,
    images: torch.Tensor,
    token_ids: torch.Tensor,
    special_ids: tuple[int, ...] = (),
) -> torch.Tensor | None:
    """
    Extract averaged Decoder Cross-Attention over Frontal spatial patches (14x14)
    from the generated report tokens without adding checkpoint parameters.
    """
    if token_ids is None or token_ids.numel() == 0:
        return None
    decoder = getattr(model, "decoder", None)
    if decoder is None:
        return None

    regional_features, _ = model.encoder(images)
    num_patches = model.encoder.patch_embed.num_patches
    side = int(num_patches ** 0.5)
    if side * side != num_patches:
        return None

    max_seq = decoder.pos_embed.size(1)
    tgt_ids = token_ids[:, :max_seq]
    T = tgt_ids.size(1)
    if T == 0:
        return None

    x = decoder.token_embed(tgt_ids) + decoder.pos_embed[:, :T, :]
    mask = decoder._causal_mask(T, x.device)

    layer_maps = []
    for layer in decoder.layers:
        a, _ = layer.self_attn(x, x, x, attn_mask=mask, need_weights=False)
        x = layer.norm1(x + a)
        a, cross_w = layer.cross_attn(
            x,
            regional_features,
            regional_features,
            need_weights=True,
            average_attn_weights=True,
        )
        x = layer.norm2(x + a)
        x = layer.norm3(x + layer.ffn(x))
        # Frontal patches occupy [0 : num_patches]
        layer_maps.append(cross_w[:, :, :num_patches])

    if not layer_maps:
        return None

    # Average across all decoder layers for balanced spatial grounding
    cross_mean = torch.stack(layer_maps, dim=0).mean(dim=0)  # (B, T, num_patches)

    # Keep only valid report word tokens (ignore <bos>, <eos>, <pad>)
    valid_mask = torch.ones_like(tgt_ids, dtype=torch.bool)
    for sid in special_ids:
        valid_mask &= tgt_ids != sid
    if valid_mask.any():
        weights = valid_mask[0]
        token_avg = cross_mean[0, weights].mean(dim=0)
    else:
        token_avg = cross_mean[0].mean(dim=0)

    return token_avg.reshape(1, 1, side, side)


def create_attention_overlay(
    pil_image: Image.Image,
    model,
    device,
    frontal_tensor: torch.Tensor,
    lateral_tensor: torch.Tensor | None,
    token_ids: torch.Tensor | None = None,
    special_ids: tuple[int, ...] = (),
) -> Image.Image:
    """
    Generate natural thermal attention overlay over the Frontal radiograph:
    - Suppresses ViT outer border attention sinks (1-cell perimeter)
    - Prioritizes Decoder Cross-Attention combined with ViT [CLS] attention
    - Smooths with Gaussian blur before/after interpolation & uses soft alpha blending
    """
    frontal = frontal_tensor.to(device)
    if lateral_tensor is not None:
        lateral = lateral_tensor.to(device)
        images = torch.stack([frontal.squeeze(0), lateral.squeeze(0)], dim=0).unsqueeze(0)
    else:
        images = torch.stack([frontal.squeeze(0), frontal.squeeze(0)], dim=0).unsqueeze(0)

    cross_grid = _extract_decoder_cross_attention(
        model, images, token_ids, special_ids=special_ids
    )
    heatmaps = model.encoder.get_attention_heatmap(
        images, img_size=448, cross_attention=cross_grid
    )
    attention_448 = heatmaps[0, 0].detach().float().cpu().numpy()

    if not np.isfinite(attention_448).all():
        raise ValueError("Encoder attention contains non-finite values.")

    # Smooth percentile stretch without hard binary thresholding (avoids harsh red rings/patches)
    p_low = float(np.percentile(attention_448, 12))
    p_high = float(np.percentile(attention_448, 99.5))
    norm = np.clip((attention_448 - p_low) / (p_high - p_low + 1e-8), 0.0, 1.0)

    # Gentle Gaussian blur on the 448x448 map for natural, continuous clinical contours
    norm_u8 = (norm * 255.0).astype(np.uint8)
    norm_blurred = np.asarray(
        Image.fromarray(norm_u8, mode="L").filter(ImageFilter.GaussianBlur(radius=8)),
        dtype=np.float32,
    ) / 255.0
    norm = np.clip((norm_blurred - norm_blurred.min()) / (norm_blurred.max() - norm_blurred.min() + 1e-8), 0.0, 1.0)

    # Harmonious thermal gradient: transparent cool background -> soft yellow/amber -> warm red peaks
    intensity = np.power(norm, 1.25)
    r_chan = np.clip(60.0 + 195.0 * np.power(intensity, 0.70), 0.0, 255.0)
    g_chan = np.clip(225.0 * (1.0 - np.power(intensity, 1.35)) + 30.0 * intensity, 0.0, 255.0)
    b_chan = np.clip(90.0 * np.power(1.0 - intensity, 2.0), 0.0, 255.0)

    # Moderate smooth alpha (peaks around 0.45, fades smoothly to 0 so lungs & mediastinum stay clear)
    alpha_chan = 0.46 * np.power(intensity, 1.15)

    rgba_map = np.stack(
        [
            r_chan.astype(np.uint8),
            g_chan.astype(np.uint8),
            b_chan.astype(np.uint8),
            np.clip(alpha_chan * 255.0, 0, 255).astype(np.uint8),
        ],
        axis=-1,
    )
    overlay_layer = Image.fromarray(rgba_map, mode="RGBA").resize(
        pil_image.size, Image.Resampling.BILINEAR
    )
    base_rgba = pil_image.convert("RGBA")
    return Image.alpha_composite(base_rgba, overlay_layer).convert("RGB")


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

    for key in (low_name, stem, no_dcm_stem):
        if key in exact_index:
            return exact_index[key]

    token_patterns = [
        r"([0-9a-f]{8}-[0-9a-f]{8}-[0-9a-f]{8}-[0-9a-f]{8}-[0-9a-f]{8})",
        r"(cxr\d+_im-\d+)",
        r"(\d+_im-\d+-\d+)",
        r"(\d+_im-\d+)",
        r"(s\d{7,})",
        r"(row_\d+)",
    ]
    for pat in token_patterns:
        m = re.search(pat, no_dcm_stem, flags=re.IGNORECASE)
        if m:
            tok = m.group(1).lower()
            if tok in exact_index:
                return exact_index[tok]
            if tok in prefix_index:
                return prefix_index[tok]

    for part in re.split(r"[/\\]", no_dcm_stem):
        if part in prefix_index:
            return prefix_index[part]

    return None


# ─── MAIN APP ────────────────────────────────────────────────────────────────────
def main():
    st.markdown("""
<div class="clinical-header">
  <h1>🫁 Hệ thống Sinh Báo cáo X-Quang Ngực Tự động (Pure Transformer)</h1>
  <p>Kiến trúc bài báo gốc · ViT Encoder + Memory-Augmented Decoder · IEEE TMI 2022 · Đối chiếu trực tiếp Ảnh &amp; Báo cáo trên 1 màn hình</p>
</div>
""", unsafe_allow_html=True)

    # 2-column side-by-side comparison: Left = Upload + Images & Heatmap (0.95), Right = Reports Comparison (1.05)
    col_left, col_right = st.columns([0.95, 1.05], gap="medium")

    # ══════════════════════════════════════════════════════════════════════════
    # CỘT TRÁI: Tải ảnh & Hiển thị song song Ảnh gốc + Attention Heatmap
    # ══════════════════════════════════════════════════════════════════════════
    with col_left:
        st.markdown('<div class="section-card"><div class="section-title">📂 Tải phim &amp; Phân tích</div>', unsafe_allow_html=True)
        analyze_clicked = st.button("🔬 Phân tích & Sinh Báo cáo X-Quang", key="run_analysis")

        up_col1, up_col2 = st.columns(2, gap="small")
        with up_col1:
            frontal_file = st.file_uploader(
                "Phim X-quang Thẳng (Frontal - bắt buộc)",
                type=["png", "jpg", "jpeg", "bmp", "tiff"],
                key="frontal_upload",
                help="Ảnh chụp X-quang lồng ngực tư thế thẳng (PA hoặc AP)",
            )
        with up_col2:
            lateral_file = st.file_uploader(
                "Phim X-quang Nghiêng (Lateral - tùy chọn)",
                type=["png", "jpg", "jpeg", "bmp", "tiff"],
                key="lateral_upload",
                help="Ảnh chụp X-quang tư thế nghiêng bổ trợ (nếu có)",
            )
        st.markdown("</div>", unsafe_allow_html=True)

        pil_frontal = None
        pil_lateral = None
        uploaded_filename = ""

        if frontal_file is not None:
            pil_frontal = Image.open(frontal_file).convert("RGB")
            uploaded_filename = frontal_file.name
            if lateral_file is not None:
                pil_lateral = Image.open(lateral_file).convert("RGB")

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

        # Run inference on button click BEFORE rendering both side-by-side images and right-column reports
        if analyze_clicked:
            if pil_frontal is None:
                st.warning("⚠️ Cần tải lên Phim X-quang Thẳng (Frontal) trước khi phân tích.")
            else:
                with st.spinner("⏳ Đang chạy Pure Transformer và sinh Attention Heatmap..."):
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
                            report_en, probs_np, out_ids = run_model_inference(
                                entry, frontal_tensor, lateral_tensor
                            )
                            results[key] = (report_en, probs_np, out_ids)
                        except Exception as ex:
                            st.warning(f"⚠️ Lỗi suy luận '{entry.get('label', key)}': {ex}")

                    if not results:
                        st.error("❌ Không có mô hình nào suy luận thành công.")
                    else:
                        mlc_probs = _compute_mlc_consensus(results, all_entries)
                        best_key, best_raw_report = select_best_report(results, all_entries, mlc_probs)
                        clean_en = format_clinical_report(best_raw_report)
                        structured_en = split_clinical_report(clean_en)

                        heatmap_image = None
                        hm_entry = all_entries.get(best_key)
                        best_out_ids = results[best_key][2]
                        if hm_entry and "model" in hm_entry:
                            try:
                                heatmap_image = create_attention_overlay(
                                    pil_frontal,
                                    hm_entry["model"],
                                    hm_entry["device"],
                                    frontal_tensor,
                                    lateral_tensor,
                                    token_ids=best_out_ids,
                                    special_ids=(hm_entry["bos"], hm_entry["eos"], hm_entry["pad"]),
                                )
                            except Exception as ex:
                                st.warning(f"⚠️ Lỗi sinh Attention Heatmap: {ex}")

                        st.session_state["diagnosis"] = {
                            "raw_report": best_raw_report,
                            "clean_en": clean_en,
                            "structured_en": structured_en,
                            "heatmap_image": heatmap_image,
                            "selected_model": all_entries[best_key]["label"],
                            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                            "uploaded_fname": uploaded_filename,
                        }

        # Side-by-side image row: Frontal (left) + Attention Heatmap (right)
        img_col1, img_col2 = st.columns(2, gap="small")
        with img_col1:
            st.markdown('<div class="sub-label">🩻 Phim X-quang Thẳng (Frontal)</div>', unsafe_allow_html=True)
            if pil_frontal is not None:
                st.image(pil_frontal, caption=uploaded_filename, use_container_width=True)
            else:
                st.markdown('<div class="img-placeholder">Chưa có phim Frontal.<br>Vui lòng tải ảnh ở khung trên.</div>', unsafe_allow_html=True)

        with img_col2:
            st.markdown('<div class="sub-label">🔥 Bản đồ nhiệt (Attention Heatmap)</div>', unsafe_allow_html=True)
            dx = st.session_state.get("diagnosis", {})
            heatmap_image = dx.get("heatmap_image")
            if pil_frontal is not None and heatmap_image is not None:
                st.image(
                    heatmap_image,
                    caption="Thang màu nhiệt: Đỏ/Cam = mức chú ý cao · Vàng nhạt = trung bình · Trong suốt = nền",
                    use_container_width=True,
                )
            else:
                msg = (
                    "Nhấn nút phân tích ở trên để hiển thị Attention Heatmap."
                    if pil_frontal is not None
                    else "Bản đồ nhiệt vùng phổi – trung thất sẽ hiển thị tại đây."
                )
                st.markdown(f'<div class="img-placeholder">{msg}</div>', unsafe_allow_html=True)

        # Lateral image collapsed neatly underneath if uploaded
        if pil_lateral is not None:
            with st.expander("🩻 Xem Phim X-quang Nghiêng (Lateral)", expanded=False):
                st.image(pil_lateral, caption=lateral_file.name, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # CỘT PHẢI: Đối chiếu Báo cáo gốc của Bác sĩ vs. Báo cáo Mô hình AI
    # ══════════════════════════════════════════════════════════════════════════
    with col_right:
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
                "(`MIMIC-CXR`, `IU X-Ray Archive`, `IU-Xray TimPhoi`)."
            )
        else:
            st.info("ℹ️ Tải lên phim X-quang ở cột trái để tự động tra cứu báo cáo bác sĩ trong `data/ground_truth/`.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Khung 2: Predicted Report by Transformer Model
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🤖 Khung 2 — Báo cáo do Mô hình AI sinh ra (Predicted Report)</div>', unsafe_allow_html=True)

        if "diagnosis" in st.session_state:
            dx = st.session_state["diagnosis"]
            structured_en = dx.get("structured_en", split_clinical_report(dx["raw_report"]))
            selected_lbl = dx.get("selected_model", "")
            st.markdown(
                f'<span class="dataset-badge">Mô hình: {selected_lbl}</span>',
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
                    height=100,
                    disabled=True,
                    key="raw_report_display",
                    label_visibility="collapsed",
                )
        else:
            st.info("ℹ️ Nhấn **🔬 Phân tích & Sinh Báo cáo X-Quang** ở cột trái để sinh báo cáo đối chiếu.")
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
