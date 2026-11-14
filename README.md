# 🩻 Multicriteria Transformer — Sinh báo cáo X-quang Y tế Tự động

Tái hiện (reimplementation) bài báo khoa học: **"Automated Radiographic Report Generation Purely on Transformer: A Multicriteria Supervised Approach"** — *Zhanyu Wang et al., IEEE Transactions on Medical Imaging (TMI), 2022.*

Dự án triển khai kiến trúc **thuần Transformer** (không sử dụng mạng CNN trích xuất đặc trưng có sẵn) nhằm tự động sinh báo cáo y khoa hoàn chỉnh từ ảnh X-quang lồng ngực kết hợp cơ chế giám sát đa tiêu chí (Multi-Criteria Supervision). 

Dự án đã tích hợp sẵn **các trọng số mô hình đã huấn luyện hoàn thiện** trong thư mục `artifacts/`. Bất kỳ ai clone repository về đều có thể **chạy ngay lập tức** mà không cần tải thêm trọng số bên ngoài hay cấu hình phức tạp.

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
1. **ViT Encoder**: Xử lý đồng thời cặp ảnh Frontal (chính diện) và Lateral (nghiêng) kích thước 448×448 với patch 32×32 (không dùng CNN pretrained). Trích xuất 392 token đặc trưng vùng và 1 token `[CLS]` toàn cục đại diện ngữ cảnh.
2. **Memory-Augmented Decoder**: Gồm 3 tầng Decoder tích hợp **Memory-Driven LayerNorm (MLN)**, sử dụng một vector bộ nhớ học được để điều biến thích ứng tham số chuẩn hóa.
3. **Giám sát Đa tiêu chí (Multi-Criteria Supervision)**: Tối ưu đồng thời 3 mục tiêu qua hàm mất mát tổng hợp:

$$
\mathcal{L}_{total} = \lambda_{RG} \cdot \mathcal{L}_{RG}^{tw} + \lambda_{MLC} \cdot \mathcal{L}_{MLC} + \lambda_{ITM} \cdot \mathcal{L}_{ITM}
$$

   - **Term-Weighted RG Loss** ($\mathcal{L}_{RG}^{tw}$): Hàm mất mát sinh từ vựng có đánh trọng số TF-IDF, phạt nặng khi sinh sai các thuật ngữ bệnh lý hiếm.
   - **MLC Loss** ($\mathcal{L}_{MLC}$): Phân loại đa nhãn 65 bệnh lý MeSH root tags từ vector `[CLS]`.
   - **Temporal-Weighted ITM Loss** ($\mathcal{L}_{ITM}$): Khớp nối ảnh - văn bản với trọng số thời gian $\lambda_i = (N_r - i + 1)^{-\alpha}$ ($\alpha = 0.5$) và Triplet Loss margin $\varepsilon = 0.05$.

---

## 📊 Bảng Đối Chiếu Hiệu Năng Thực Nghiệm với Bài Báo Gốc

Kết quả thực nghiệm trên tập kiểm thử chuẩn **IU-Xray** (2.937 mẫu hợp lệ có đủ 2 góc chụp Frontal + Lateral, chia 80% Train / 10% Val / 10% Test) đối chiếu cùng bài báo gốc (IEEE TMI 2022) và các mô hình nền tảng (Baselines):

| Mô hình | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L | CIDEr |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **M2Transformer** (Baseline) | 0.453 | 0.290 | 0.208 | 0.156 | 0.365 | 0.382 |
| **R2Gen** (EMNLP 2020) | 0.470 | 0.304 | 0.219 | 0.165 | 0.371 | 0.398 |
| **IEEE TMI 2022 (Bài báo gốc)** | **0.496** | **0.327** | **0.238** | **0.178** | **0.381** | **0.445** |
| 🩻 **DACN — TimPhoi (Đã tích hợp)** | 0.314 | 0.205 | 0.146 | 0.093 | 0.293 | **0.449** |
| 🩻 **DACN — Archive (Đã tích hợp)** | 0.311 | 0.186 | 0.123 | 0.085 | 0.279 | 0.327 |

> 🌟 **Điểm nổi bật:** Mô hình **TimPhoi** của dự án đạt điểm **CIDEr = 0.4486** (làm tròn **0.449**), hoàn toàn tương đương mốc **0.445** được công bố trong bài báo gốc IEEE TMI 2022.

---

## 🗂 Cấu trúc Thư mục Dự án

```
DACN/
├── artifacts/                      # CÁC MÔ HÌNH ĐÃ HUẤN LUYỆN SẴN (ĐẦY ĐỦ TRỌNG SỐ)
│   ├── archive/                    #   Mô hình IU-Xray Archive (Full 2.937 ca)
│   │   ├── best_archive.pt         #     Trọng số PyTorch checkpoint (sẵn sàng chạy)
│   │   ├── vocab_archive.json      #     Từ điển từ vựng (1.690 từ)
│   │   ├── train_log_archive.csv   #     Nhật ký huấn luyện 19 epochs
│   │   └── test_results_archive.txt#     Kết quả đánh giá trên tập Test
│   └── timphoi/                    #   Mô hình TimPhoi (CIDEr = 0.449)
│       ├── best_timphoi.pt         #     Trọng số PyTorch checkpoint (sẵn sàng chạy)
│       ├── vocab_timphoi.json      #     Từ điển từ vựng (873 từ)
│       ├── train_log_timphoi.csv   #     Nhật ký huấn luyện 13 epochs
│       └── test_results_timphoi.txt#     Kết quả đánh giá trên tập Test
├── model/                          # Kiến trúc thuần Transformer
│   ├── vit_encoder.py              #   Vision Transformer Encoder (3 lớp, shared weights)
│   ├── decoder.py                  #   Memory-Augmented Text Decoder
│   ├── memory_layernorm.py         #   Khối Memory-Driven LayerNorm (MLN)
│   ├── heads.py                    #   Đầu phụ trợ MLC Head và ITM Head
│   ├── losses.py                   #   Hàm tổn thất RG, MLC, ITM Loss
│   └── full_model.py               #   Mô hình hoàn chỉnh tích hợp & giải mã
├── data/                           # Xử lý dữ liệu
│   ├── preprocess_iu_xray.py       #   Tiền xử lý: tách từ, chia split, gán nhãn MeSH
│   ├── iu_xray_dataset.py          #   PyTorch Dataset nạp ảnh cặp 448x448
│   ├── tokenizer.py                #   Bộ tách từ mã hóa word-level tokenizer
│   ├── mesh_labeler.py             #   Trích xuất 65 nhãn bệnh MeSH root tags từ CSV
│   └── chexpert_labeler.py         #   Bộ gán nhãn luật 14 bệnh CheXpert
├── config.py                       # Toàn bộ siêu tham số cấu hình mô hình & train
├── train.py                        # Vòng lặp huấn luyện, Early Stopping, AMP, Logging
├── demo.py                         # Ứng dụng giao diện web Streamlit chẩn đoán tự động
├── requirements.txt                # Danh sách thư viện nhẹ, cài đặt ngay không lỗi
└── .gitignore                      # Cấu hình Git
```

---

## 🚀 HƯỚNG DẪN CHẠY BÀI (QUICK START)

Dự án được tối ưu để **bất kỳ ai clone về cũng có thể chạy ngay** chỉ với 3 lệnh:

### Bước 1: Clone dự án và cài đặt môi trường

```bash
git clone https://github.com/Shrio182/DACN.git
cd DACN
pip install -r requirements.txt
```

*File `requirements.txt` chỉ chứa các thư viện chuẩn (`torch`, `torchvision`, `pillow`, `streamlit`, `pandas`, `numpy`), không yêu cầu cài đặt trình biên dịch C++ hay Java.*

### Bước 2: Khởi chạy Ứng dụng Chẩn đoán Giao diện Web (Streamlit)

```bash
python -m streamlit run demo.py
```
*(Hoặc `streamlit run demo.py` nếu đã cấu hình biến môi trường PATH)*

### Bước 3: Thao tác trên giao diện:
1. **Chọn mô hình**: Lựa chọn giữa `TimPhoi` (khuyên dùng, CIDEr=0.449) hoặc `IU-Xray Archive`.
2. **Tải ảnh X-quang**: Chọn ảnh chụp lồng ngực chính diện **Frontal** (và tùy chọn ảnh nghiêng **Lateral**).
3. **Chẩn đoán**: Bấm **"Thực hiện Chẩn đoán & Sinh Báo Cáo Y Khoa"**.
4. **Xem kết quả**:
   - Báo cáo y khoa hoàn chỉnh mô tả tim, trung thất, phế trường hai phổi.
   - Bảng phân phối xác suất 65 nhóm bệnh lý MeSH tương ứng.

---

## 🔄 Huấn luyện lại từ đầu (Tùy chọn)

Nếu bạn muốn tự huấn luyện lại mô hình trên dữ liệu mới:

```bash
# 1. Tiền xử lý dữ liệu từ thư mục chứa CSV và images:
python data/preprocess_iu_xray.py --data_dir ./data/iu_xray

# 2. Chạy huấn luyện:
python train.py --data_dir ./data/iu_xray --epochs 30 --batch_size 16 --amp --patience 5
```

---

## 🛠 Công nghệ Sử dụng

- **Ngôn ngữ**: Python 3.10+
- **Deep Learning Framework**: PyTorch
- **Thị giác máy tính**: torchvision, Pillow
- **Giao diện người dùng**: Streamlit
- **Xử lý số liệu**: NumPy, Pandas

---

## 📖 Trích dẫn Bài báo Gốc (Citation)

```bibtex
@article{wang2022automated,
  title={Automated Radiographic Report Generation Purely on Transformer: A Multicriteria Supervised Approach},
  author={Wang, Zhanyu and Liu, Lingqiao and Wang, Lei and Zhou, Luping},
  journal={IEEE Transactions on Medical Imaging},
  volume={41},
  number={10},
  pages={2803--2814},
  year={2022},
  publisher={IEEE}
}
```

---

## ⚠️ Tuyên bố Miễn trừ Y tế (Medical Disclaimer)

Dự án này được xây dựng phục vụ mục đích **nghiên cứu khoa học học thuật**. Các chẩn đoán và văn bản báo cáo tạo sinh từ mô hình trí tuệ nhân tạo **không thay thế** kết luận chuyên môn của bác sĩ chẩn đoán hình ảnh và chuyên gia y tế.
