"""
demo.py — Streamlit demo cho Multicriteria Radiology Report Generation.
========================================================================
Chạy:
    streamlit run demo.py

Yêu cầu:
    - Đã train xong: checkpoints/best.pt
    - Đã preprocess:  iu_xray_data/vocab.json
    - pip install streamlit
"""
import os
import sys
import time
import torch
import streamlit as st
from PIL import Image
from torchvision import transforms

# ── Cấu hình trang ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Radiology Report Generator",
    page_icon="🩻",
    layout="centered",
)

# ── Đường dẫn mặc định ──────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = "checkpoints/best.pt"
DEFAULT_VOCAB      = "iu_xray_data/vocab.json"
DEFAULT_DATA_DIR   = "iu_xray_data"

# ── Load model (cache để không load lại mỗi lần) ────────────────────────────
@st.cache_resource(show_spinner="Đang load model...")
def load_model(checkpoint_path: str, vocab_path: str):
    sys.path.insert(0, os.path.dirname(__file__))
    from config import ModelConfig, TrainConfig
    from model.full_model import MulticriteriaTransformer
    from data.tokenizer import SimpleTokenizer

    tokenizer = SimpleTokenizer.load(vocab_path)
    mcfg = ModelConfig()
    tcfg = TrainConfig()
    mcfg.vocab_size    = len(tokenizer)
    mcfg.pad_idx       = 0
    mcfg.use_paired_views = True

    model = MulticriteriaTransformer(mcfg, tcfg)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    return model, tokenizer, device, mcfg


def preprocess_image(img: Image.Image, img_size: int = 224) -> torch.Tensor:
    """Chuẩn hoá ảnh X-quang về tensor (1, H, W)."""
    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    return tfm(img)


def generate_report(
    model, tokenizer, device, mcfg,
    frontal: Image.Image,
    lateral: Image.Image | None,
) -> tuple[str, float]:
    """Sinh báo cáo từ 1 hoặc 2 ảnh, trả về (text, thời_gian_giây)."""
    from data.tokenizer import BOS, EOS

    bos_idx = tokenizer.vocab[BOS]
    eos_idx = tokenizer.vocab[EOS]

    front_t = preprocess_image(frontal, mcfg.img_size)
    lat_t   = preprocess_image(lateral if lateral else frontal, mcfg.img_size)

    # Stack 2 view: (1, 2, 1, H, W)
    images = torch.stack([front_t, lat_t], dim=0).unsqueeze(0).to(device)

    t0 = time.time()
    with torch.no_grad():
        out_ids = model.generate(images, bos_idx, eos_idx)
    elapsed = time.time() - t0

    report = tokenizer.decode(out_ids[0].tolist())
    return report, elapsed


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🩻 Radiology Report Generator")
st.caption(
    "Tái hiện paper **IEEE TMI 2022** — "
    "_Automated Radiographic Report Generation Purely on Transformer_"
)
st.divider()

# Sidebar: cài đặt đường dẫn
with st.sidebar:
    st.header("⚙️ Cài đặt")
    checkpoint_path = st.text_input("Checkpoint (.pt)", value=DEFAULT_CHECKPOINT)
    vocab_path      = st.text_input("Vocab (vocab.json)", value=DEFAULT_VOCAB)
    st.divider()
    st.markdown(
        "**Lệnh train:**\n"
        "```bash\n"
        "python train.py --data_dir ./iu_xray_data \\\n"
        "  --epochs 30 --batch_size 8 --amp\n"
        "```"
    )
    st.markdown(
        "**Lệnh eval:**\n"
        "```bash\n"
        "python eval.py --data_dir ./iu_xray_data \\\n"
        "  --checkpoint checkpoints/best.pt\n"
        "```"
    )

# Kiểm tra file tồn tại
checkpoint_ok = os.path.exists(checkpoint_path)
vocab_ok      = os.path.exists(vocab_path)

if not vocab_ok:
    st.error(f"❌ Không tìm thấy vocab: `{vocab_path}`  \n"
             "Chạy: `python data/preprocess_iu_xray.py --data_dir ./iu_xray_data`")
    st.stop()

if not checkpoint_ok:
    st.warning(
        f"⚠️ Chưa có checkpoint: `{checkpoint_path}`  \n"
        "Model sẽ dùng trọng số **ngẫu nhiên** (chỉ để test UI).  \n"
        "Train xong mới có kết quả có ý nghĩa."
    )

# Load model
try:
    model, tokenizer, device, mcfg = load_model(
        checkpoint_path if checkpoint_ok else None,
        vocab_path,
    )
    st.success(f"✅ Model loaded — device: `{device}` — vocab: {len(tokenizer)} từ")
except Exception as e:
    # Nếu chưa có checkpoint, load model không có weight
    st.info("ℹ️ Load model với trọng số ngẫu nhiên (chưa có checkpoint).")
    try:
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(__file__))
        from config import ModelConfig, TrainConfig
        from model.full_model import MulticriteriaTransformer
        from data.tokenizer import SimpleTokenizer
        tokenizer = SimpleTokenizer.load(vocab_path)
        mcfg = ModelConfig()
        tcfg = TrainConfig()
        mcfg.vocab_size = len(tokenizer)
        mcfg.pad_idx = 0
        mcfg.use_paired_views = True
        model = MulticriteriaTransformer(mcfg, tcfg)
        model.eval()
        device = torch.device("cpu")
        st.success(f"✅ Model (random weights) — device: `{device}` — vocab: {len(tokenizer)} từ")
    except Exception as e2:
        st.error(f"❌ Không thể load model: {e2}")
        st.stop()

st.divider()

# Upload ảnh
st.subheader("📤 Tải lên ảnh X-quang")
col1, col2 = st.columns(2)
with col1:
    frontal_file = st.file_uploader(
        "Frontal (PA / AP) — bắt buộc",
        type=["png", "jpg", "jpeg", "dcm"],
        key="frontal",
    )
with col2:
    lateral_file = st.file_uploader(
        "Lateral — tuỳ chọn (nếu không có sẽ dùng frontal×2)",
        type=["png", "jpg", "jpeg", "dcm"],
        key="lateral",
    )

if frontal_file:
    frontal_img = Image.open(frontal_file).convert("RGB")
    lateral_img = Image.open(lateral_file).convert("RGB") if lateral_file else None

    # Hiển thị ảnh preview
    st.subheader("🖼️ Preview")
    prev_col1, prev_col2 = st.columns(2)
    with prev_col1:
        st.image(frontal_img, caption="Frontal", use_container_width=True)
    with prev_col2:
        if lateral_img:
            st.image(lateral_img, caption="Lateral", use_container_width=True)
        else:
            st.info("Không có ảnh lateral — sẽ dùng frontal × 2")

    st.divider()

    # Nút sinh báo cáo
    if st.button("🔬 Sinh báo cáo", type="primary", use_container_width=True):
        with st.spinner("Đang sinh báo cáo..."):
            try:
                report, elapsed = generate_report(
                    model, tokenizer, device, mcfg,
                    frontal_img, lateral_img,
                )
                st.subheader("📋 Báo cáo sinh được")
                st.markdown(
                    f"""<div style="
                        background:#f0f8ff;
                        border-left:4px solid #1f77b4;
                        padding:16px 20px;
                        border-radius:6px;
                        font-size:16px;
                        line-height:1.7;
                    ">{report}</div>""",
                    unsafe_allow_html=True,
                )
                st.caption(f"⏱ Thời gian sinh: {elapsed:.2f}s | Device: {device}")

                # Nút copy
                st.code(report, language=None)
                st.info(
                    "💡 **Lưu ý:** Đây là báo cáo do AI tạo ra, "
                    "chỉ dùng cho mục đích nghiên cứu học thuật. "
                    "Không dùng cho chẩn đoán lâm sàng."
                )
            except Exception as e:
                st.error(f"❌ Lỗi khi sinh báo cáo: {e}")
else:
    st.info("👆 Tải lên ảnh X-quang để bắt đầu.")

# Footer
st.divider()
st.caption(
    "📄 Tái hiện: _Automated Radiographic Report Generation Purely on Transformer: "
    "A Multicriteria Supervised Approach_ — IEEE TMI 2022  \n"
    "Dataset: IU-Xray (Indiana University)"
)

