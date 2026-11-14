# 🩻 Multicriteria Transformer — Sinh báo cáo X-quang tự động

Tái hiện (reimplementation) bài báo khoa học: **"Automated Radiographic Report Generation Purely on Transformer: A Multicriteria Supervised Approach"** — *Zhanyu Wang et al., IEEE Transactions on Medical Imaging (TMI), 2022.*

Dự án triển khai kiến trúc **thuần Transformer** (không sử dụng mạng CNN trích xuất đặc trưng có sẵn) nhằm tự động sinh báo cáo y khoa hoàn chỉnh từ ảnh X-quang lồng ngực kết hợp cơ chế giám sát đa tiêu chí (Multi-Criteria Supervision).

---

## 📐 Kiến trúc Mô hình và Luồng Xử lý

```
Ảnh X-quang cặp (Frontal + Lateral)
                  │
                  ▼
  ┌───────────────────────────────┐
  │   Vision Transformer (ViT)    │   3 tầng Encoder (trọng số chia sẻ)
  │    Encoder (from scratch)     │   Ảnh 448×448, Patch 32×32
  │                               │   → 392 Regional Tokens + 1 [CLS]
  └───────────────┬───────────────┘
                  │
        ┌─────────┼─────────────────┐
        │         │                 │
        ▼         ▼                 ▼
  ┌──────────┐ ┌──────────┐ ┌───────────────┐
  │ MLC Head │ │ ITM Head │ │ Text Decoder  │  3 tầng Transformer Decoder
  │ (65 nhãn │ │ (So khớp │ │ (Tích hợp     │  tích hợp Memory-Driven
  │   MeSH)  │ │ ảnh-chữ) │ │     MLN)      │  LayerNorm (MLN)
  └──────────┘ └──────────┘ └───────┬───────┘
                                    │
                                    ▼
                          Báo cáo Y khoa Tự động
                           (Generated Report)
```

### Các thành phần cốt lõi:
1. **ViT Encoder**: Xử lý đồng thời cặp ảnh Frontal (chính diện) và Lateral (nghiêng) kích thước 448×448 với patch 32×32 (không dùng pretrained CNN). Trích xuất 392 token đặc trưng vùng và 1 token `[CLS]` toàn cục.
2. **Memory-Augmented Decoder**: Gồm 3 tầng Decoder tích hợp **Memory-Driven LayerNorm (MLN)**, sử dụng một vector bộ nhớ học được để điều biến thích ứng tham số scale/shift của quá trình chuẩn hóa.
3. **Giám sát Đa tiêu chí (Multi-Criteria Loss)**:
   $$\mathcal{L}_{total} = \lambda_{RG} \cdot \mathcal{L}_{RG}^{tw} + \lambda_{MLC} \cdot \mathcal{L}_{MLC} + \lambda_{ITM} \cdot \mathcal{L}_{ITM}$$
   - **$\mathcal{L}_{RG}^{tw}$**: Hàm mất mát sinh từ vựng có đánh trọng số TF-IDF, ưu tiên các thuật ngữ bệnh lý hiếm gặp.
   - **$\mathcal{L}_{MLC}$**: Phân loại đa nhãn 65 bệnh lý MeSH root tags từ vector `[CLS]`.
   - **$\mathcal{L}_{ITM}$**: Khớp nối ảnh - văn bản với trọng số thời gian $\lambda_i = (N_r - i + 1)^{-\alpha}$ ($\alpha = 0.5$) và Triplet Loss margin $\varepsilon = 0.05$.

---

## 🗂 Cấu trúc Thư mục Dự án

```
DACN/
├── model/                          # Kiến trúc mô hình thuần Transformer
│   ├── vit_encoder.py              #   Vision Transformer Encoder (3 lớp, shared weights)
│   ├── decoder.py                  #   Memory-Augmented Text Decoder
│   ├── memory_layernorm.py         #   Khối Memory-Driven LayerNorm (MLN)
│   ├── heads.py                    #   Đầu phụ trợ MLC Head và ITM Head
│   ├── losses.py                   #   Hàm tổn thất RG, MLC, ITM Loss
│   └── full_model.py               #   Mô hình hoàn chỉnh tích hợp & thuật toán giải mã
├── data/                           # Dữ liệu và các script xử lý
│   ├── preprocess_iu_xray.py       #   Tiền xử lý toàn diện: tách từ, chia split, gán nhãn MeSH
│   ├── iu_xray_dataset.py          #   PyTorch Dataset nạp ảnh cặp 448x448 & nhãn
│   ├── tokenizer.py                #   Bộ tách từ mã hóa word-level tokenizer
│   ├── mesh_labeler.py             #   Trích xuất 65 nhãn bệnh MeSH root tags từ CSV
│   ├── chexpert_labeler.py         #   Bộ gán nhãn luật 14 bệnh CheXpert
│   ├── iu_xray/                    #   Dữ liệu IU-Xray gốc (images/ + reports CSV)
│   └── archive/                    #   Dữ liệu IU-Xray Archive (metadata.csv + images/)
├── artifacts/                      # Checkpoints và kết quả huấn luyện chuẩn
│   ├── archive/                    #   Mô hình IU-Xray Archive (full 2.937 ca)
│   │   ├── vocab_archive.json      #     Từ điển từ vựng (1.690 từ)
│   │   ├── train_log_archive.csv   #     Nhật ký huấn luyện 19 epochs
│   │   └── test_results_archive.txt#     Kết quả đánh giá trên tập Test
│   └── timphoi/                    #   Mô hình TimPhoi (lọc ca bệnh tim - phổi)
│       ├── vocab_timphoi.json      #     Từ điển từ vựng (873 từ)
│       ├── train_log_timphoi.csv   #     Nhật ký huấn luyện 13 epochs
│       └── test_results_timphoi.txt#     Kết quả đánh giá trên tập Test (CIDEr=0.4486)
├── config.py                       # Toàn bộ siêu tham số cấu hình mô hình & train
├── train.py                        # Vòng lặp huấn luyện, Early Stopping, AMP, Logging
├── eval.py                         # Đánh giá chỉ số BLEU-1..4, ROUGE-L, CIDEr, METEOR
├── demo.py                         # Giao diện web demo Streamlit chẩn đoán tự động
├── smoke_test.py                   # Kiểm thử toàn diện pipeline mô hình trước khi train
├── dataset_filter_audit.py         # Đối soát số lượng mẫu lọc dữ liệu
├── plot_comparison.py              # Vẽ biểu đồ so sánh Training Curves & Test Metrics
├── plot_paper_comparison.py        # Vẽ biểu đồ đối chiếu với bài báo gốc (IEEE TMI 2022)
├── requirements.txt                # Danh sách thư viện Python cần thiết
└── .gitignore                      # Chặn push dữ liệu ảnh và file weights nặng lên Git
```

---

## 📊 Kết quả Thực nghiệm

Mô hình được huấn luyện trên bộ dữ liệu **IU-Xray** (2.937 ca hợp lệ có đủ cặp ảnh Frontal + Lateral, chia 80% Train / 10% Val / 10% Test):

| Mô hình | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L | CIDEr |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **IU-Xray Archive** | 0.3110 | 0.1862 | 0.1234 | 0.0845 | 0.2794 | 0.3273 |
| **TimPhoi** | 0.3139 | 0.2051 | 0.1462 | 0.0933 | 0.2931 | **0.4486** |
| **Bài báo gốc (TMI 2022)** | 0.4960 | 0.3270 | 0.2380 | 0.1780 | 0.3810 | **0.4450** |

*Ghi chú: Mô hình `TimPhoi` đạt điểm **CIDEr = 0.4486**, bám rất sát mốc **0.4450** của bài báo gốc.*

---

## 🚀 HƯỚNG DẪN CHẠY BÀI CHI TIẾT (HOW TO RUN)

### 1. Chuẩn bị Môi trường và Cài đặt

Yêu cầu máy tính cài đặt **Python 3.10+**. Khuyến nghị sử dụng môi trường ảo:

```bash
# Clone dự án từ GitHub
git clone https://github.com/Shrio182/DACN.git
cd DACN

# Cài đặt toàn bộ các thư viện phụ thuộc
pip install -r requirements.txt
```

Nếu muốn tính toán chỉ số `METEOR` trong `eval.py`, cần cài đặt thêm **Java JRE** trên máy.

---

### 2. Chuẩn bị Dữ liệu và Tiền xử lý

Cấu trúc thư mục dữ liệu IU-Xray cần chuẩn bị (ví dụ đặt tại `data/iu_xray/`):
```
data/iu_xray/
├── indiana_reports.csv
├── indiana_projections.csv
└── images/
    ├── 1_1.png
    ├── ...
```

Chạy script tiền xử lý để tự động lọc dữ liệu, xây dựng từ điển và trích xuất 65 nhãn MeSH:
```bash
python data/preprocess_iu_xray.py --data_dir ./data/iu_xray
```
Sau khi chạy xong, thư mục dữ liệu sẽ tự động có các file:
- `annotation.json`
- `vocab.json`
- `term_weights.pt`
- `mesh_labels.json`
- `processed_train.json`, `processed_val.json`, `processed_test.json`

---

### 3. Kiểm thử Pipeline Trước khi Train (Smoke Test)

Để chắc chắn luồng dữ liệu, forward pass, gradient và hàm mất mát hoạt động chính xác trước khi train nhiều giờ:
```bash
python smoke_test.py --data_dir ./data/iu_xray
```

---

### 4. Huấn luyện Mô hình (Training)

Chạy vòng lặp huấn luyện chính thức với các tham số tối ưu:
```bash
python train.py --data_dir ./data/iu_xray --epochs 30 --batch_size 16 --amp --patience 5
```

**Các tham số quan trọng:**
- `--data_dir`: Đường dẫn thư mục dữ liệu đã tiền xử lý.
- `--epochs`: Số epoch tối đa (mặc định: 30 epochs theo bài báo).
- `--batch_size`: Kích thước batch trên GPU (nếu dùng GPU có VRAM nhỏ như 8GB-12GB, giảm về 8).
- `--accum_steps`: Tích lũy gradient (ví dụ: `--batch_size 8 --accum_steps 2` tương đương batch 16).
- `--amp`: Bật Mixed Precision (tăng tốc 2x và tiết kiệm 50% VRAM).
- `--patience`: Dừng sớm (Early Stopping) nếu điểm CIDEr trên tập val không tăng sau số epoch này.

*Checkpoint tốt nhất được tự động lưu tại `checkpoints/best.pt`.*

---

### 5. Đánh giá Mô hình trên Tập Test (Evaluation)

Sau khi huấn luyện xong, chạy đánh giá toàn bộ các thước đo BLEU-1..4, ROUGE-L, CIDEr:
```bash
python eval.py --data_dir ./data/iu_xray --checkpoint checkpoints/best.pt --split test
```

---

### 6. Chạy Ứng dụng Giao diện Web (Streamlit Demo)

Khởi chạy ứng dụng web trực quan để tương tác và chẩn đoán ca bệnh:
```bash
streamlit run demo.py
```

**Các thao tác trên giao diện:**
1. Chọn mô hình từ thanh bên trái: `TimPhoi` hoặc `IU-Xray Archive`.
2. Tải lên ảnh chụp chính diện **Frontal** (và tùy chọn ảnh nghiêng **Lateral**).
3. Nhấn **"Thực hiện Chẩn đoán & Sinh Báo Cáo Y Khoa"**.
4. Xem kết quả văn bản sinh tự động và biểu đồ xác suất 65 nhóm bệnh lý MeSH.

---

### 7. Xuất Biểu đồ So sánh và Đối chiếu Thực nghiệm

Để vẽ biểu đồ phân tích training loss, điểm kiểm thử và so sánh với bài báo gốc:

```bash
# 1. Vẽ biểu đồ so sánh các checkpoint trong thư mục artifacts/
python plot_comparison.py

# 2. Vẽ biểu đồ đối chiếu hiệu năng với bài báo gốc IEEE TMI 2022
python plot_paper_comparison.py
```

Các ảnh biểu đồ độ phân giải cao (300 DPI) sẽ được lưu tự động vào thư mục `outputs/charts/`.

---

## 🛠 Công nghệ Sử dụng

- **Ngôn ngữ**: Python 3.10+
- **Deep Learning Framework**: PyTorch
- **Thị giác máy tính**: torchvision, Pillow
- **Giao diện người dùng**: Streamlit
- **Xử lý số liệu & Đồ thị**: NumPy, Pandas, Matplotlib
- **Đánh giá văn bản y khoa**: pycocoevalcap (BLEU, ROUGE, CIDEr, METEOR)
- **Môi trường huấn luyện**: Kaggle GPU (Tesla P100 / T4)

---

## ⚠️ Tuyên bố Miễn trừ Y tế (Medical Disclaimer)

Dự án này được xây dựng và phục vụ mục đích **nghiên cứu khoa học học thuật**. Các chẩn đoán và văn bản báo cáo tạo sinh từ mô hình trí tuệ nhân tạo **không thay thế** kết luận chuyên môn của bác sĩ chẩn đoán hình ảnh và chuyên gia y tế.
