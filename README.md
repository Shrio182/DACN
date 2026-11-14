# 🩻 Multicriteria Transformer — Sinh báo cáo X-quang tự động

Tái hiện (reimplementation) bài báo **"Automated Radiographic Report Generation Purely on Transformer: A Multicriteria Supervised Approach"** — *Zhanyu Wang et al., IEEE Transactions on Medical Imaging (TMI), 2022.*

Hệ thống sử dụng kiến trúc **thuần Transformer** (không dùng CNN pretrained) để tự động sinh báo cáo y khoa từ ảnh X-quang lồng ngực, kết hợp giám sát đa tiêu chí giúp cải thiện chất lượng ngôn ngữ và độ chính xác lâm sàng.

---

## 📐 Kiến trúc Mô hình

```
Ảnh X-quang (Frontal + Lateral)
        │
        ▼
┌──────────────────────────────┐
│   Vision Transformer (ViT)   │   3 lớp Encoder, chia sẻ trọng số
│   Encoder (from scratch)     │   Patch 32×32, Ảnh 448×448
│                              │   → 392 Regional Tokens + 1 [CLS]
└──────────┬───────────────────┘
           │
     ┌─────┼─────────────────┐
     │     │                 │
     ▼     ▼                 ▼
┌────────┐ ┌──────────┐ ┌─────────┐
│MLC Head│ │ITM Head  │ │Decoder  │   Memory-Augmented Transformer
│(65 MeSH│ │(Ảnh-Văn  │ │(3 lớp + │   Decoder với Memory-Driven
│ labels)│ │ bản)     │ │ MLN)    │   LayerNorm (MLN)
└────────┘ └──────────┘ └────┬────┘
                             │
                             ▼
                    Báo cáo Y khoa
                    (Generated Report)
```

### Thành phần chính

| Thành phần | Mô tả |
|---|---|
| **ViT Encoder** | 3 lớp Vision Transformer chia sẻ trọng số xử lý cặp ảnh Frontal + Lateral (448×448, patch 32×32, grayscale). Trích xuất 392 regional tokens và 1 `[CLS]` token toàn cục. |
| **Memory-Augmented Decoder** | 3 lớp Transformer Decoder tích hợp Memory-Driven LayerNorm (MLN) — cơ chế điều chế thống kê chuẩn hóa dựa trên bộ nhớ học được. |
| **Multi-Label Classification (MLC)** | Phân loại 65 nhãn bệnh MeSH root tags từ `[CLS]` token. |
| **Image-Text Matching (ITM)** | Khớp nối ảnh-văn bản bằng Triplet Loss với trọng số thời gian (temporal weighting). |

### Giám sát Đa tiêu chí (Multi-Criteria Loss)

$$\mathcal{L}_{total} = \lambda_{RG} \cdot \mathcal{L}_{RG}^{tw} + \lambda_{MLC} \cdot \mathcal{L}_{MLC} + \lambda_{ITM}(e) \cdot \mathcal{L}_{ITM}$$

- **Term-Weighted RG Loss** ($\mathcal{L}_{RG}^{tw}$): Cross-entropy đánh trọng số từ vựng dựa trên TF-IDF.
- **MLC Loss** ($\mathcal{L}_{MLC}$): Binary Cross-Entropy cho 65 nhãn MeSH.
- **Temporal-Weighted ITM Loss** ($\mathcal{L}_{ITM}$): Triplet loss margin ε=0.05, L₂-reg μ=0.025, trọng số thời gian α=0.5.

---

## 🗂 Cấu trúc Dự án

```
DACN/
├── model/                          # Kiến trúc mô hình
│   ├── vit_encoder.py              #   ViT Encoder (3 lớp, patch embed, 2-view shared)
│   ├── decoder.py                  #   Memory-Augmented Decoder (causal mask, cross-attn)
│   ├── memory_layernorm.py         #   Memory-Driven LayerNorm (MLN)
│   ├── heads.py                    #   MLC Head + ITM Head
│   ├── losses.py                   #   RG, MLC, ITM losses
│   └── full_model.py               #   Gộp tất cả + generate (Beam Search)
├── data/                           # Tiền xử lý & dataset
│   ├── preprocess_iu_xray.py       #   Build vocab, term-weights, nhãn MeSH, chia split
│   ├── iu_xray_dataset.py          #   PyTorch Dataset cho IU-Xray
│   ├── tokenizer.py                #   SimpleTokenizer (word-level)
│   ├── mesh_labeler.py             #   Trích xuất 65 MeSH root tags từ báo cáo
│   ├── chexpert_labeler.py         #   Rule-based labeler xử lý phủ định
│   ├── iu_xray/                    #   Dữ liệu IU-Xray gốc (images/ + reports)
│   └── archive/                    #   Dữ liệu IU-Xray Archive (images/ + metadata.csv)
├── artifacts/                      # Kết quả huấn luyện đã hoàn thành
│   ├── archive/                    #   Mô hình IU-Xray Archive
│   │   ├── vocab_archive.json      #     Từ điển (1.690 từ)
│   │   ├── train_log_archive.csv   #     Log train 19 epochs
│   │   └── test_results_archive.txt#     Kết quả test
│   └── timphoi/                    #   Mô hình TimPhoi
│       ├── vocab_timphoi.json      #     Từ điển (873 từ)
│       ├── train_log_timphoi.csv   #     Log train 13 epochs
│       └── test_results_timphoi.txt#     Kết quả test
├── config.py                       # Siêu tham số (d=512, 8 heads, patch 32)
├── train.py                        # Vòng lặp train + Early Stopping + Log CSV
├── eval.py                         # Đánh giá BLEU / ROUGE-L / CIDEr / METEOR
├── demo.py                         # Ứng dụng Streamlit demo sinh báo cáo
├── smoke_test.py                   # Kiểm thử forward pass toàn bộ pipeline
├── dataset_filter_audit.py         # Kiểm tra & lọc dữ liệu Archive
├── plot_comparison.py              # Vẽ biểu đồ so sánh Training & Test metrics
├── plot_paper_comparison.py        # So sánh kết quả với bài báo gốc
├── requirements.txt                # Thư viện Python cần cài đặt
└── .gitignore
```

> **Lưu ý:** File checkpoint (`*.pt`) và thư mục ảnh (`images/`) không được push lên Git. Huấn luyện được thực hiện trên Kaggle.

---

## 📊 Kết quả Thực nghiệm

Huấn luyện trên tập **IU-Xray** (2.937 mẫu hợp lệ, chia 80/10/10) với Kaggle GPU.

| Mô hình | BLEU-1 | BLEU-4 | ROUGE-L | CIDEr |
|---|:---:|:---:|:---:|:---:|
| **IU-Xray Archive** | 0.3110 | 0.0845 | 0.2794 | 0.3273 |
| **TimPhoi** | 0.3139 | 0.0933 | 0.2931 | **0.4486** |
| Bài báo gốc (TMI 2022) | 0.486 | 0.138 | 0.365 | 0.449 |

---

## 🚀 Hướng dẫn Sử dụng

### Cài đặt

```bash
git clone https://github.com/Shrio182/DACN.git
cd DACN
pip install -r requirements.txt
```

### Tiền xử lý dữ liệu

```bash
python data/preprocess_iu_xray.py --data_dir ./data/iu_xray
```

Tự động tạo: `vocab.json`, `term_weights.pt`, `mesh_labels.json`, `processed_train/val/test.json`.

### Huấn luyện

```bash
python train.py --data_dir ./data/iu_xray --epochs 30 --batch_size 16 --amp --patience 5
```

Hỗ trợ: Mixed Precision (`--amp`), Gradient Accumulation (`--accum_steps`), Early Stopping theo CIDEr.

### Đánh giá

```bash
python eval.py --data_dir ./data/iu_xray --checkpoint checkpoints/best.pt
```

### Demo Web App (Streamlit)

```bash
streamlit run demo.py
```

Giao diện cho phép:
- Chọn mô hình từ `artifacts/` (IU-Xray Archive, TimPhoi, tự động phát hiện MIMIC-CXR).
- Tải lên cặp ảnh Frontal + Lateral.
- Điều chỉnh Beam Search (k=2–8) và Repetition Penalty.
- Xem báo cáo sinh tự động và xác suất 65 nhãn bệnh MeSH.

### Vẽ biểu đồ so sánh

```bash
python plot_comparison.py
python plot_paper_comparison.py
```

---

## 🛠 Công nghệ

- **Python 3.10+**
- **PyTorch** — Xây dựng và huấn luyện mô hình
- **torchvision** — Augmentation và transform ảnh
- **Streamlit** — Giao diện demo tương tác
- **pycocoevalcap** — Tính BLEU, ROUGE-L, CIDEr, METEOR
- **matplotlib** — Vẽ biểu đồ phân tích
- **Kaggle GPU** — Môi trường huấn luyện

---

## 📖 Tham khảo

```bibtex
@article{wang2022automated,
  title={Automated Radiographic Report Generation Purely on Transformer:
         A Multicriteria Supervised Approach},
  author={Wang, Zhanyu and others},
  journal={IEEE Transactions on Medical Imaging},
  volume={41},
  number={10},
  pages={2803--2814},
  year={2022}
}
```

---

## ⚠️ Tuyên bố miễn trừ

Dự án này được thực hiện trong khuôn khổ **nghiên cứu học thuật**. Kết quả sinh ra từ mô hình AI **không thay thế** chẩn đoán chuyên môn của bác sĩ chẩn đoán hình ảnh.
