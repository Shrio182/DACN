# Prompt để dán vào Antigravity Agent

> Mở thư mục project `mct_skeleton/` trong Antigravity, mở Agent panel, dán toàn bộ
> phần dưới đây. Đọc kỹ phần "Ràng buộc" để AI không phá code đang chạy được.

---

## Bối cảnh dự án

Đây là đồ án chuyên ngành dựng lại paper IEEE TMI 2022:
"Automated Radiographic Report Generation Purely on Transformer: A Multicriteria
Supervised Approach" (sinh báo cáo y tế từ ảnh X-quang ngực).

Project đã có sẵn bộ khung HOÀN CHỈNH và đã được test chạy được:
- `model/`: ViT encoder + memory-augmented decoder + 3 loss (RG+term-weight,
  ITM+temporal-weight, MLC 14 bệnh). Đúng kiến trúc paper.
- `data/`: tokenizer, CheXpert rule-based labeler, preprocess IU-Xray, Dataset class.
- `train.py`, `eval.py`, `config.py`.
- Đã test với data giả và layout CSV+images của bộ Kaggle raddar/chest-xrays-indiana-university.

Dataset thật đã đặt trong `iu_xray_data/` (chứa `indiana_reports.csv`,
`indiana_projections.csv`, thư mục `images/`).

## Nhiệm vụ cho bạn (làm theo thứ tự, cứ mỗi bước xong báo lại)

1. **Kiểm tra môi trường**: tạo virtual env, cài `torch` (phiên bản CUDA nếu máy có GPU),
   `torchvision pillow numpy pandas nltk pycocoevalcap`. Chạy `python test_forward.py`
   để xác nhận model chạy.
2. **Tiền xử lý data**: chạy `python data/preprocess_iu_xray.py --data_dir ./iu_xray_data`.
   Nếu báo lỗi không tìm thấy ảnh, kiểm tra tên thư mục ảnh và sửa đường dẫn.
   Xác nhận in ra số mẫu train/val/test (~3955 báo cáo, ~7466 ảnh).
3. **Thêm vào train.py**: early stopping theo CIDEr trên tập val, lưu checkpoint tốt nhất
   thành `checkpoints/best.pt`, in loss mỗi 10 step, log tensorboard hoặc CSV đơn giản.
   GIỮ NGUYÊN 2 nhóm learning rate (encoder 5e-5, còn lại 1e-4) và 30 epochs theo paper.
4. **Hoàn thiện eval.py**: tích hợp đầy đủ `pycocoevalcap` để tính BLEU-1→4, METEOR,
   ROUGE-L, CIDEr trên tập test, in bảng kết quả.
5. **Viết demo Streamlit** (`demo.py`): chọn file ảnh X-quang → load `best.pt` →
   `model.generate()` → hiển thị báo cáo sinh được.
6. **Khuyến nghị**: nếu máy không có GPU, viết `train_kaggle.ipynb` hoặc hướng dẫn
   chạy trên Kaggle GPU (dataset đã có sẵn trên Kaggle).

## Lệnh chính

```bash
python data/preprocess_iu_xray.py --data_dir ./iu_xray_data
python train.py --data_dir ./iu_xray_data --epochs 30
python eval.py --data_dir ./iu_xray_data --checkpoint checkpoints/best.pt
streamlit run demo.py
```

## Ràng buộc (QUAN TRỌNG — không được vi phạm)

- KHÔNG đổi kiến trúc model (encoder 3 lớp, decoder memory-augmented, 3 loss) nếu chưa
  hỏi tôi. Chỉ được sửa phần train/eval/demo.
- Mỗi lần sửa file, giải thích ngắn gọn đã sửa gì.
- Nếu một lệnh chạy >5 phút (như train full), KHÔNG chờ trong phiên agent — khởi chạy
  nền (background) và hướng dẫn tôi theo dõi log.
- Commit git trước mỗi thay đổi lớn để tôi có thể revert.

## Dấu hiệu train hợp lệ

- `rg` giảm dần từ ~6-7 về ~2-3.
- `mlc` ban đầu ~0.69 (random), sau đó giảm.
- `itm` ban đầu ~log(batch_size)=2.77, sau đó giảm.
- `w_itm` tăng dần từ 0 về 1 (temporal weighting).

"Hãy chỉnh sửa lại toàn bộ source code của project (config.py, các file model, loss, và dataloader) để tuân thủ tuyệt đối các thiết lập và thông số kỹ thuật được công bố trong bài báo khoa học gốc (Automated Radiographic Report Generation Purely on Transformer: A Multicriteria Supervised Approach), cụ thể như sau:Cấu hình Model & Ảnh đầu vào (config.py):Resize ảnh X-quang đầu vào về kích thước chính xác $448 \times 448$, patch size $32$.   Bật chế độ dùng 2 góc chụp cho IU-Xray (use_paired_views = True) ghép frontal và lateral lại với nhau.   Số lớp của Vision Encoder = 3, Text Decoder = 3, số heads chú ý = 8.   Trọng số hàm Loss (Equation 12):Cập nhật lại các trọng số cân bằng Loss: $\lambda_{RG} = 1.0$, $\lambda_{MLC} = 2.0$, và $\lambda_{ITM} = 5.0$.   Cài đặt chuẩn các tham số cho ITM loss: biên độ triplet $\epsilon = 0.05$, L2-regularization $\mu_{ITM} = 0.025$, hệ số temporal-weighting $\alpha = 0.5$.   Quá trình Tối ưu hóa (Optimization):Dùng Adam Optimizer với batch size = 16.   Thiết lập learning rate riêng: 5e-5 cho Vision Encoder và 1e-4 cho các phần còn lại.   Huấn luyện đủ 30 epochs.   Tiền xử lý dữ liệu IU-Xray:Đảm bảo script tiền xử lý lọc chính xác các báo cáo có đủ 2 góc chụp và đầy đủ hai phần Findings và Impression (chuẩn 3.195 mẫu), đồng thời chia ngẫu nhiên 10% làm tập test đúng như bài báo." 