# demo.py
# Ứng dụng giao diện web Streamlit phục vụ chẩn đoán hình ảnh và tự động sinh báo cáo X-quang y tế.
# Tái hiện bài báo: "Automated Radiographic Report Generation Purely on Transformer:
#                   A Multicriteria Supervised Approach" (IEEE TMI 2022).

import os
import sys
import glob
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

st.set_page_config(
    page_title="Radiology Report AI (IEEE TMI 2022)",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Tự động quét và phát hiện TẤT CẢ các mô hình trong artifacts/ ──────────────
def discover_all_models():
    models = {}
    artifacts_dir = os.path.join(CURRENT_DIR, "artifacts")
    
    if os.path.exists(artifacts_dir):
        for item in sorted(os.listdir(artifacts_dir)):
            sub_path = os.path.join(artifacts_dir, item)
            if not os.path.isdir(sub_path):
                continue
            
            # Tìm file trọng số .pt hoặc .pth
            pt_files = glob.glob(os.path.join(sub_path, "*.pt")) + glob.glob(os.path.join(sub_path, "*.pth"))
            # Tìm file từ điển vocab*.json
            vocab_files = glob.glob(os.path.join(sub_path, "vocab*.json"))
            if not vocab_files:
                all_jsons = glob.glob(os.path.join(sub_path, "*.json"))
                vocab_files = [j for j in all_jsons if not os.path.basename(j).startswith("mesh_")]

            if pt_files and vocab_files:
                ckpt_file = pt_files[0]
                vocab_file = vocab_files[0]
                
                # Đọc kết quả test nếu có
                metrics_summary = ""
                test_txts = glob.glob(os.path.join(sub_path, "test_results*.txt"))
                if test_txts:
                    try:
                        with open(test_txts[0], "r", encoding="utf-8", errors="ignore") as f:
                            lines = f.readlines()
                            cider_val = ""
                            b4_val = ""
                            for line in lines:
                                if "CIDEr" in line:
                                    cider_val = line.split()[-1]
                                elif "BLEU-4" in line:
                                    b4_val = line.split()[-1]
                            if cider_val:
                                metrics_summary = f"CIDEr: {cider_val}" + (f" | BLEU-4: {b4_val}" if b4_val else "")
                    except Exception:
                        pass

                display_name = item.upper() if item in ["mimic", "archive"] else item.capitalize()
                if item.lower() == "timphoi":
                    display_name = "IU-Xray TimPhoi (Khuyên dùng)"
                elif item.lower() == "archive":
                    display_name = "IU-Xray Archive (Tập đầy đủ)"
                elif item.lower() == "mimic":
                    display_name = "MIMIC-CXR"

                models[display_name] = {
                    "checkpoint": ckpt_file,
                    "vocab": vocab_file,
                    "folder": item,
                    "summary": metrics_summary or "Mô hình đã huấn luyện hoàn thiện",
                }

    # Thêm tùy chọn nhập đường dẫn tùy chỉnh
    models["Tùy chỉnh (Đường dẫn thủ công)..."] = {
        "checkpoint": "",
        "vocab": "",
        "folder": "custom",
        "summary": "Tự nhập đường dẫn file .pt và vocab.json",
    }

    return models


# ── Hàm nạp mô hình dùng cache tài nguyên Streamlit ────────────────────────────
@st.cache_resource(show_spinner="Đang nạp trọng số mô hình và từ điển vào bộ nhớ...")
def load_trained_model(checkpoint_path: str, vocab_path: str):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Không tìm thấy file checkpoint: {checkpoint_path}")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Không tìm thấy file từ vựng: {vocab_path}")

    tokenizer = SimpleTokenizer.load(vocab_path)
    mcfg = ModelConfig()
    tcfg = TrainConfig()

    mcfg.vocab_size = len(tokenizer)
    mcfg.pad_idx = tokenizer.vocab.get(PAD, tokenizer.vocab.get("<pad>", 0))
    mcfg.use_paired_views = True
    mcfg.img_size = 448
    mcfg.patch_size = 32

    # Nạp state_dict an toàn, hỗ trợ bóc tách wrapper
    raw_ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(raw_ckpt, dict):
        if "model_state_dict" in raw_ckpt:
            state_dict = raw_ckpt["model_state_dict"]
        elif "state_dict" in raw_ckpt:
            state_dict = raw_ckpt["state_dict"]
        else:
            state_dict = raw_ckpt
    else:
        state_dict = raw_ckpt

    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Cập nhật số nhãn bệnh lý MLC từ trọng số checkpoint
    if "mlc_head.proj.weight" in clean_state_dict:
        mcfg.num_mlc_labels = clean_state_dict["mlc_head.proj.weight"].shape[0]

    model = MulticriteriaTransformer(mcfg, tcfg)
    model.load_state_dict(clean_state_dict, strict=False)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Nạp danh sách nhãn MeSH
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
    # Chuẩn hóa ảnh X-quang đúng chuẩn huấn luyện: Grayscale 1 kênh, resize 448x448, mean=[0.5], std=[0.5]
    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    return tfm(img)


# ── GIAO DIỆN CHÍNH ───────────────────────────────────────────────────────────
def main():
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                    padding: 22px 28px; border-radius: 10px; margin-bottom: 22px; color: white;">
            <h1 style="color: white; margin: 0; font-size: 2.1rem;">🩻 Radiology Report Generation AI</h1>
            <p style="margin: 8px 0 0 0; opacity: 0.92; font-size: 1.02rem;">
                Hệ thống tự động sinh báo cáo X-quang & dự đoán bệnh lý dựa trên kiến trúc thuần Transformer (IEEE TMI 2022)
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    available_models = discover_all_models()

    # ── VỊ TRÍ ĐẦU TIÊN THANH BÊN (SIDEBAR): CHỌN MÔ HÌNH ──────────────────────
    with st.sidebar:
        st.header("🎯 Lựa chọn Mô hình")

        model_choice = st.selectbox(
            "Chọn mô hình chẩn đoán:",
            list(available_models.keys()),
            index=0,
            help="Danh sách quét tự động tất cả các mô hình đã huấn luyện trong thư mục artifacts/.",
        )

        selected_info = available_models[model_choice]
        st.info(f"📊 **Hiệu năng:** {selected_info['summary']}")

        if model_choice.startswith("Tùy chỉnh"):
            checkpoint_path = st.text_input("Đường dẫn Checkpoint (.pt):", value="")
            vocab_path = st.text_input("Đường dẫn Vocab (.json):", value="")
        else:
            checkpoint_path = selected_info["checkpoint"]
            vocab_path = selected_info["vocab"]

        st.divider()
        st.subheader("🎛️ Tinh chỉnh Giải mã (Visual Guidance)")

        visual_guidance = st.slider(
            "Độ nhạy thị giác (Visual Guidance):",
            min_value=0.0,
            max_value=1.5,
            value=0.8,
            step=0.1,
            help="Cơ chế Contrastive Guidance: Giá trị > 0 giúp triệt tiêu mẫu câu bình thường mặc định và buộc mô hình tập trung vào đặc trưng dị thường trên ảnh.",
        )

        use_beam = st.toggle("Kích hoạt Beam Search", value=True)
        beam_size = st.slider("Beam Size (k):", min_value=2, max_value=5, value=3, step=1) if use_beam else 1
        repetition_penalty = st.slider("Phạt lặp từ (Repetition Penalty):", min_value=1.0, max_value=1.6, value=1.25, step=0.05)

        st.divider()
        st.subheader("📋 Cấu hình Kỹ thuật (IEEE TMI 2022)")
        st.markdown(
            "- **Đầu vào**: Cặp ảnh Frontal + Lateral\n"
            "- **Độ phân giải**: `448 × 448` (Grayscale 1 kênh)\n"
            "- **Patch Size**: `32 × 32` (196 vùng/view)\n"
            "- **Vision Transformer**: 3 lớp, trọng số chia sẻ\n"
            "- **Decoder**: 3 lớp + Memory-Driven LayerNorm"
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
    dev_color = "#28a745" if device.type == "cuda" else "#007bff"
    st.markdown(
        f"""
        <div style="background:#f8f9fa; border:1px solid #e9ecef; border-radius:8px; padding:10px 16px; margin-bottom:18px; font-size:0.92rem;">
            <b>Mô hình hoạt động:</b> <code>{model_choice}</code> &nbsp;|&nbsp;
            <b>Thiết bị tính toán:</b> <span style="background:{dev_color}; color:white; padding:2px 8px; border-radius:4px; font-weight:bold;">{str(device).upper()}</span> &nbsp;|&nbsp;
            <b>Từ điển từ vựng:</b> <code>{len(tokenizer):,} từ</code> &nbsp;|&nbsp;
            <b>Nhãn bệnh lý (MeSH):</b> <code>{mcfg.num_mlc_labels} nhãn</code>
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
            help="Ảnh chụp X-quang hướng trước-sau (PA/AP).",
        )

    with col_up2:
        lat_file = st.file_uploader(
            "📷 Ảnh Lateral (Nghiêng) — Tùy chọn",
            type=["png", "jpg", "jpeg", "dcm"],
            key="file_lat",
            help="Nếu không có ảnh nghiêng, hệ thống sẽ tự động dùng ảnh chính diện làm góc nhìn bổ trợ.",
        )

    if not front_file:
        st.info("👆 Vui lòng tải lên ảnh X-quang chính diện (Frontal) để tiến hành chẩn đoán.")
        return

    img_front = Image.open(front_file).convert("RGB")
    img_lat = Image.open(lat_file).convert("RGB") if lat_file else None

    prev_c1, prev_c2 = st.columns(2)
    with prev_c1:
        st.image(img_front, caption="Góc chụp Frontal (Input)", use_container_width=True)
    with prev_c2:
        if img_lat:
            st.image(img_lat, caption="Góc chụp Lateral (Input)", use_container_width=True)
        else:
            st.image(img_front, caption="Góc chụp Lateral (Tự động nhân bản Frontal)", use_container_width=True)

    st.write("")
    btn_predict = st.button("🚀 Thực hiện Chẩn đoán & Sinh Báo Cáo Y Khoa", type="primary", use_container_width=True)

    if btn_predict:
        # Chuẩn hóa tensor ảnh đầu vào (1, 2, 1, 448, 448)
        t_front = preprocess_xray_image(img_front, mcfg.img_size)
        t_lat = preprocess_xray_image(img_lat if img_lat else img_front, mcfg.img_size)
        batch_images = torch.stack([t_front, t_lat], dim=0).unsqueeze(0).to(device)

        bos_idx = tokenizer.vocab.get(BOS, tokenizer.vocab.get("<bos>", 1))
        eos_idx = tokenizer.vocab.get(EOS, tokenizer.vocab.get("<eos>", 2))

        with st.spinner("Đang trích xuất đặc trưng qua Vision Transformer và giải mã báo cáo..."):
            t_start = time.time()
            with torch.no_grad():
                # 1. Sinh báo cáo tự hồi quy có định hướng thị giác
                out_ids = model.generate(
                    batch_images,
                    bos_idx=bos_idx,
                    eos_idx=eos_idx,
                    max_len=mcfg.max_seq_len,
                    beam_size=beam_size,
                    repetition_penalty=repetition_penalty,
                    visual_guidance=visual_guidance,
                )
                generated_text = tokenizer.decode(out_ids[0].tolist())

                # 2. Dự đoán xác suất nhãn bệnh lý từ CLS token
                mlc_probs = model.predict_mlc(batch_images)[0].cpu().numpy()

            elapsed = time.time() - t_start

        st.success(f"✅ Hoàn tất xử lý trong {elapsed:.2f} giây trên thiết bị {str(device).upper()}")

        # ── KẾT QUẢ SINH BÁO CÁO VÀ PHÂN LOẠI BỆNH LÝ ─────────────────────────
        res_col1, res_col2 = st.columns([1.15, 0.85])

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
                f"Độ dài: {len(generated_text.split())} từ | Guidance: {visual_guidance} | Beam Size: {beam_size}"
            )

        with res_col2:
            st.subheader("🩺 Top 5 Bệnh lý Có Xác suất Cao nhất (MeSH MLC)")

            # Ghép tên nhãn và xác suất
            label_probs = []
            for i, p in enumerate(mlc_probs):
                name = mesh_labels[i] if i < len(mesh_labels) else f"Tag #{i}"
                label_probs.append((name, float(p)))

            label_probs.sort(key=lambda x: x[1], reverse=True)

            # Hiển thị Top 5 bệnh lý
            st.markdown("Xác thực từ **Multi-Label Classification Head**:")
            for tag_name, prob in label_probs[:5]:
                prob_pct = prob * 100
                st.write(f"**{tag_name.capitalize()}**: `{prob_pct:.2f}%`")
                st.progress(min(max(prob, 0.0), 1.0))

            with st.expander("🔍 Xem toàn bộ 65 nhãn MeSH và xác suất"):
                import pandas as pd
                df_all = pd.DataFrame(label_probs, columns=["Tên nhãn MeSH", "Xác suất (0-1)"])
                df_all["Tỷ lệ (%)"] = (df_all["Xác suất (0-1)"] * 100).round(3)
                st.dataframe(df_all, use_container_width=True, height=280)

        st.divider()
        st.warning(
            "⚠️ **Lưu ý y khoa quan trọng:** Báo cáo trên được sinh tự động từ mô hình trí tuệ nhân tạo "
            "trong khuôn khổ nghiên cứu thực nghiệm học thuật (IEEE TMI 2022). "
            "Kết quả không thay thế chẩn đoán chuyên môn của bác sĩ chẩn đoán hình ảnh."
        )


if __name__ == "__main__":
    main()
