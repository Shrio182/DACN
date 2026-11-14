"""
demo.py — Ứng dụng Streamlit chẩn đoán và sinh báo cáo X-quang y tế tự động.
============================================================================
Tái hiện paper IEEE TMI 2022:
"Automated Radiographic Report Generation Purely on Transformer:
 A Multicriteria Supervised Approach"

Hỗ trợ:
  - Lựa chọn linh hoạt mô hình trong artifacts/ (IU-Xray Archive, TimPhoi, MIMIC-CXR).
  - Tự động nhận diện thiết bị (CUDA/CPU) và kích thước từ vựng (vocab_size).
  - Nhập 2 góc chụp X-quang (Frontal + Lateral) với độ phân giải chuẩn 448×448.
  - Sinh báo cáo y khoa (Generated Report) hỗ trợ Beam Search & Repetition Penalty.
  - Hiển thị xác suất phân loại đa nhãn bệnh (Multi-Label Disease Probabilities).

Chạy:
  streamlit run demo.py
"""

import os
import sys
import time
import json
import torch
import streamlit as st
from PIL import Image
from torchvision import transforms

# Đảm bảo import được module dự án
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from config import ModelConfig, TrainConfig
from model.full_model import MulticriteriaTransformer
from data.tokenizer import SimpleTokenizer, BOS, EOS, PAD

# ── Danh sách 65 MeSH Root Tags tiêu chuẩn của IU-Xray ─────────────────────────
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
    "trachea", "tuberculosis", "vascular calcification", "vertebrae", "wire", "bone", "ribs"
]


# ── Cấu hình trang Streamlit ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Radiology Report AI (IEEE TMI 2022)",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Phát hiện các mô hình trong artifacts/ ─────────────────────────────────────
def discover_models():
    models = {}

    # 1. IU-Xray Archive
    arc_pt = os.path.join(CURRENT_DIR, "artifacts", "archive", "best_archive.pt")
    arc_vocab = os.path.join(CURRENT_DIR, "artifacts", "archive", "vocab_archive.json")
    if os.path.exists(arc_pt) and os.path.exists(arc_vocab):
        models["IU-Xray Archive"] = {
            "checkpoint": arc_pt,
            "vocab": arc_vocab,
            "desc": "Tập IU-Xray đầy đủ (2.937 ca, vocab=1.690)",
        }

    # 2. TimPhoi
    tp_pt = os.path.join(CURRENT_DIR, "artifacts", "timphoi", "best_timphoi.pt")
    tp_vocab = os.path.join(CURRENT_DIR, "artifacts", "timphoi", "vocab_timphoi.json")
    if os.path.exists(tp_pt) and os.path.exists(tp_vocab):
        models["TimPhoi"] = {
            "checkpoint": tp_pt,
            "vocab": tp_vocab,
            "desc": "Tập IU-Xray Tim Phổi (CIDEr=0.449, vocab=873)",
        }

    # 3. Tự động kiểm tra MIMIC-CXR
    mimic_pt = os.path.join(CURRENT_DIR, "artifacts", "mimic", "best_mimic.pt")
    mimic_vocab = os.path.join(CURRENT_DIR, "artifacts", "mimic", "vocab_mimic.json")
    if os.path.exists(mimic_pt) and os.path.exists(mimic_vocab):
        models["MIMIC-CXR"] = {
            "checkpoint": mimic_pt,
            "vocab": mimic_vocab,
            "desc": "Mô hình huấn luyện trên MIMIC-CXR",
        }

    # 4. Fallback checkpoints cục bộ nếu có
    local_pt = os.path.join(CURRENT_DIR, "checkpoints", "best.pt")
    local_vocab = os.path.join(CURRENT_DIR, "data", "vocab.json")
    if os.path.exists(local_pt) and os.path.exists(local_vocab):
        models["Checkpoints Local"] = {
            "checkpoint": local_pt,
            "vocab": local_vocab,
            "desc": "Checkpoint cục bộ (checkpoints/best.pt)",
        }

    # Tuỳ chọn nhập thủ công
    models["Tùy chỉnh (Custom)..."] = {
        "checkpoint": "",
        "vocab": "",
        "desc": "Tự nhập đường dẫn file .pt và vocab.json",
    }

    return models


# ── Hàm nạp mô hình có bộ nhớ đệm ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Đang nạp mô hình và từ điển vào bộ nhớ...")
def load_trained_model(checkpoint_path: str, vocab_path: str):
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Không tìm thấy file từ vựng: {vocab_path}")

    tokenizer = SimpleTokenizer.load(vocab_path)
    mcfg = ModelConfig()
    tcfg = TrainConfig()

    mcfg.vocab_size = len(tokenizer)
    mcfg.pad_idx = tokenizer.vocab.get(PAD, 0)
    mcfg.use_paired_views = True
    mcfg.img_size = 448
    mcfg.patch_size = 32

    # Đọc số nhãn bệnh từ checkpoint nếu có
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "mlc_head.proj.weight" in state_dict:
        mcfg.num_mlc_labels = state_dict["mlc_head.proj.weight"].shape[0]

    model = MulticriteriaTransformer(mcfg, tcfg)
    model.load_state_dict(state_dict)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Thử nạp nhãn MeSH từ file nếu có
    labels = DEFAULT_MESH_LABELS
    possible_mesh_paths = [
        os.path.join(os.path.dirname(vocab_path), "mesh_labels.json"),
        os.path.join(CURRENT_DIR, "data", "mesh_labels.json"),
    ]
    for pmp in possible_mesh_paths:
        if os.path.exists(pmp):
            try:
                with open(pmp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    labels = data.get("mesh_classes", labels)
                break
            except Exception:
                pass

    return model, tokenizer, device, mcfg, labels


def preprocess_xray_image(img: Image.Image, img_size: int = 448) -> torch.Tensor:
    """Chuẩn hóa ảnh X-quang theo đúng chuẩn đầu vào ViT Encoder (Grayscale 448×448)."""
    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    return tfm(img)


# ── GIAO DIỆN CHÍNH ───────────────────────────────────────────────────────────
def main():
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                    padding: 24px 30px; border-radius: 10px; margin-bottom: 25px; color: white;">
            <h1 style="color: white; margin: 0; font-size: 2.2rem;">🩻 Radiology Report Generation AI</h1>
            <p style="margin: 8px 0 0 0; opacity: 0.9; font-size: 1.05rem;">
                Hệ thống tự động sinh báo cáo X-quang & dự đoán bệnh lý dựa trên kiến trúc thuần Transformer (IEEE TMI 2022)
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    available_models = discover_models()

    # ── SIDEBAR CÀI ĐẶT ────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Cấu hình Mô hình")

        model_choice = st.selectbox(
            "Chọn mô hình đã huấn luyện:",
            list(available_models.keys()),
            index=0,
        )

        selected_info = available_models[model_choice]
        st.caption(f"ℹ️ {selected_info['desc']}")

        if model_choice == "Tùy chỉnh (Custom)...":
            checkpoint_path = st.text_input("Đường dẫn Checkpoint (.pt):", value="")
            vocab_path = st.text_input("Đường dẫn Vocab (.json):", value="")
        else:
            checkpoint_path = selected_info["checkpoint"]
            vocab_path = selected_info["vocab"]

        st.divider()
        st.subheader("📋 Thông số Mô hình (IEEE TMI 2022)")
        st.markdown(
            "- **Đầu vào**: Cặp ảnh Frontal + Lateral\n"
            "- **Kích thước ảnh**: `448 × 448` (Grayscale)\n"
            "- **Patch size**: `32 × 32` (196 patches/view)\n"
            "- **Vision Transformer**: 3 lớp, chia sẻ trọng số\n"
            "- **Text Decoder**: 3 lớp + Memory-Driven LayerNorm\n"
            "- **Độ dài báo cáo tối đa**: `60 từ`"
        )

    # ── KIỂM TRA FILE VÀ NẠP MODEL ─────────────────────────────────────────────
    if not checkpoint_path or not vocab_path or not os.path.exists(checkpoint_path) or not os.path.exists(vocab_path):
        st.warning(
            f"⚠️ Chưa tìm thấy file checkpoint hoặc vocab tương ứng.\n\n"
            f"- **Checkpoint**: `{checkpoint_path}`\n"
            "- **Vocab**: `{vocab_path}`\n\n"
            "Hãy đảm bảo các file đã có trong thư mục `artifacts/`."
        )
        return

    try:
        model, tokenizer, device, mcfg, mesh_labels = load_trained_model(checkpoint_path, vocab_path)
    except Exception as e:
        st.error(f"❌ Lỗi khi khởi tạo mô hình: {e}")
        return

    # Thông tin model trạng thái hiện tại
    dev_color = "#28a745" if device.type == "cuda" else "#ffc107"
    st.markdown(
        f"""
        <div style="background:#f8f9fa; border:1px solid #e9ecef; border-radius:8px; padding:12px 18px; margin-bottom:20px; font-size:0.95rem;">
            <b>Mô hình hoạt động:</b> <code>{model_choice}</code> &nbsp;|&nbsp;
            <b>Thiết bị tính toán:</b> <span style="background:{dev_color}; color:white; padding:2px 8px; border-radius:4px; font-weight:bold;">{str(device).upper()}</span> &nbsp;|&nbsp;
            <b>Kích thước từ điển:</b> <code>{len(tokenizer):,} từ</code> &nbsp;|&nbsp;
            <b>Số nhãn bệnh lý (MeSH):</b> <code>{mcfg.num_mlc_labels} nhãn</code>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── KHU VỰC TẢI LÊN ẢNH ─────────────────────────────────────────────────────
    st.subheader("1. Tải lên cặp ảnh X-quang lồng ngực (Frontal + Lateral)")
    col_up1, col_up2 = st.columns(2)

    with col_up1:
        front_file = st.file_uploader(
            "📷 Ảnh Frontal (Chính diện PA / AP) — Bắt buộc",
            type=["png", "jpg", "jpeg", "dcm"],
            key="file_front",
            help="Ảnh chụp X-quang hướng trước-sau hoặc sau-trước.",
        )

    with col_up2:
        lat_file = st.file_uploader(
            "📷 Ảnh Lateral (Nghiêng) — Tùy chọn",
            type=["png", "jpg", "jpeg", "dcm"],
            key="file_lat",
            help="Nếu không có ảnh nghiêng, hệ thống sẽ tự động nhân bản ảnh chính diện làm góc nhìn bổ trợ.",
        )

    if not front_file:
        st.info("👆 Vui lòng tải lên ảnh X-quang chính diện (Frontal) để tiến hành chẩn đoán.")
        return

    # Hiển thị ảnh xem trước
    img_front = Image.open(front_file).convert("RGB")
    img_lat = Image.open(lat_file).convert("RGB") if lat_file else None

    prev_c1, prev_c2 = st.columns(2)
    with prev_c1:
        st.image(img_front, caption="Góc chụp Frontal (Input)", use_container_width=True)
    with prev_c2:
        if img_lat:
            st.image(img_lat, caption="Góc chụp Lateral (Input)", use_container_width=True)
        else:
            st.image(img_front, caption="Góc chụp Lateral (Tự động dùng Frontal thay thế)", use_container_width=True)

    st.write("")
    btn_predict = st.button("🚀 Thực hiện Chẩn đoán & Sinh Báo Cáo Y Khoa", type="primary", use_container_width=True)

    if btn_predict:
        # Chuẩn bị Tensor ảnh
        t_front = preprocess_xray_image(img_front, mcfg.img_size)
        t_lat = preprocess_xray_image(img_lat if img_lat else img_front, mcfg.img_size)

        # Shape: (1, 2, 1, 448, 448)
        batch_images = torch.stack([t_front, t_lat], dim=0).unsqueeze(0).to(device)

        bos_idx = tokenizer.vocab.get(BOS, 1)
        eos_idx = tokenizer.vocab.get(EOS, 2)

        with st.spinner("Đang xử lý ảnh qua Vision Transformer Encoder và sinh báo cáo..."):
            t_start = time.time()
            with torch.no_grad():
                # 1. Sinh báo cáo văn bản theo cơ chế chuẩn
                out_ids = model.generate(
                    batch_images,
                    bos_idx=bos_idx,
                    eos_idx=eos_idx,
                    max_len=mcfg.max_seq_len,
                )
                generated_text = tokenizer.decode(out_ids[0].tolist())

                # 2. Dự đoán xác suất nhãn bệnh từ CLS token
                mlc_probs = model.predict_mlc(batch_images)[0].cpu().numpy()

            elapsed = time.time() - t_start

        st.success(f"✅ Hoàn tất xử lý trong {elapsed:.2f} giây trên {str(device).upper()}")

        # ── KẾT QUẢ SINH BÁO CÁO VÀ PHÂN LOẠI BỆNH LÝ ─────────────────────────
        res_col1, res_col2 = st.columns([1.1, 0.9])

        with res_col1:
            st.subheader("📋 Báo cáo Y khoa Tự động (Generated Report)")
            st.markdown(
                f"""
                <div style="background:#f4f9ff; border: 2px solid #b8daff; border-left: 6px solid #0056b3;
                            padding: 20px; border-radius: 8px; font-size: 1.08rem; line-height: 1.8;
                            color: #1a252f; font-family: monospace;">
                    {generated_text}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")
            st.code(generated_text, language=None)

            st.caption(
                f"Độ dài báo cáo tạo sinh: {len(generated_text.split())} từ (Giới hạn: {mcfg.max_seq_len} từ)"
            )

        with res_col2:
            st.subheader("🩺 Dự đoán Xác suất Bệnh lý (MeSH MLC)")

            # Ghép tên nhãn và xác suất
            label_probs = []
            for i, p in enumerate(mlc_probs):
                name = mesh_labels[i] if i < len(mesh_labels) else f"Tag #{i}"
                label_probs.append((name, float(p)))

            label_probs.sort(key=lambda x: x[1], reverse=True)

            # Hiển thị Top 8 bệnh lý có xác suất cao nhất
            st.markdown("Xác suất từ **Multi-Label Classification Head**:")
            for tag_name, prob in label_probs[:8]:
                prob_pct = prob * 100
                st.write(f"**{tag_name.capitalize()}**: `{prob_pct:.1f}%`")
                # Tô màu thanh tiến trình
                st.progress(min(prob, 1.0))

            # Bảng tra cứu mở rộng
            with st.expander("🔍 Xem toàn bộ 65 nhãn MeSH và xác suất"):
                import pandas as pd
                df_all = pd.DataFrame(label_probs, columns=["Tên nhãn MeSH", "Xác suất (0-1)"])
                df_all["Tỷ lệ (%)"] = (df_all["Xác suất (0-1)"] * 100).round(2)
                st.dataframe(df_all, use_container_width=True, height=280)

        st.divider()
        st.warning(
            "⚠️ **Lưu ý y khoa quan trọng:** Kết quả trên được sinh ra từ mô hình trí tuệ nhân tạo "
            "trong khuôn khổ nghiên cứu thực nghiệm học thuật (IEEE TMI 2022). "
            "Hệ thống không thay thế các chẩn đoán chuyên môn của bác sĩ chẩn đoán hình ảnh."
        )


if __name__ == "__main__":
    main()
