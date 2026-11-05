### CÁC ĐIỀU CHỈNH BẮT BUỘC ĐỂ SÁT VỚI PAPER

**1. IU-Xray Dataset**

- Số lượng ban đầu theo paper: 3.955 reports và 7.470 images.
- Sau khi lọc: 3.195 reports và 6.390 images.
- Chỉ giữ các mẫu có đủ ảnh Frontal và Lateral.
- Chỉ giữ các báo cáo có đầy đủ Findings và Impression.
- Ghép dữ liệu theo UID, không ghép nhầm ảnh và báo cáo.
- Báo cáo số lượng thực tế nếu khác số lượng paper.

**2. Paired Image Input**

- Bật `use_paired_views = True`.
- Resize từng ảnh về 448 × 448.
- Patch size = 32 × 32.
- Mã hóa Frontal và Lateral bằng Vision Transformer.
- Kết hợp visual features của hai góc chụp theo phương pháp trong paper.
- Không tự ý ghép hai ảnh thành một ảnh sáu kênh.

**3. Multi-Label Classification**

- Giữ nhiệm vụ MLC.
- Sử dụng CLS feature của Vision Transformer.
- Với IU-Xray, kiểm tra trường Tags/MeSH và xây dựng multi-label targets từ dữ liệu thực tế.
- Không mặc định IU-Xray có 14 lớp.
- 14 structured labels được paper mô tả rõ trong phần MIMIC-CXR.
- Nếu sử dụng CheXpert rule-based labeler cho IU-Xray, phải ghi rõ đây là phương án triển khai thay thế, không phải cấu hình nhãn được paper xác nhận.

**4. Image-Text Matching**

- Giữ Triplet Loss theo Equation 10.
- Margin = 0.05.
- L2 regularization = 0.025.
- Temporal weighting alpha = 0.5.
- Negative reports được lấy từ các mẫu khác trong mini-batch.
- Temporal weighting được áp dụng theo vị trí token trong report, không phải scheduler tăng trọng số theo epoch.
- Không yêu cầu ITM loss khởi đầu phải bằng log(batch_size).

**5. Training**

- Optimizer = Adam.
- Mini-batch size = 16.
- Vision Encoder learning rate = 5e-5.
- Các tham số còn lại learning rate = 1e-4.
- Training = 30 epochs.
- Loss weights: RG = 1.0, MLC = 2.0, ITM = 5.0.
- Nếu phần cứng không đủ, báo cáo rõ batch size thực tế và cách điều chỉnh.

**6. Data Split và Evaluation**

- Giữ 10% reports làm test set.
- Không chia ảnh riêng lẻ khiến cùng UID xuất hiện ở nhiều tập.
- Validation set là phần bổ sung của quá trình tái triển khai, không được ghi là thiết lập nguyên bản của paper.
- Các chỉ số chính: BLEU, ROUGE-L và CIDEr.
- METEOR có thể được bổ sung như chỉ số tham khảo.

**7. Nguyên tắc sửa source code**

Trước khi sửa, hãy kiểm tra và đối chiếu toàn bộ model, loss và dataloader hiện có với paper.

Không viết lại những thành phần đã triển khai đúng.

Chỉ sửa những thành phần sai, thiếu hoặc chưa tương thích với dataset IU-Xray. Mỗi thay đổi cần nêu rõ lý do và cách kiểm chứng.