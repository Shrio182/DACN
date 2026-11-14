# iu_xray_dataset.py
# Lớp Dataset nạp dữ liệu IU-Xray cho PyTorch DataLoader
# Đọc file danh sách processed_<split>.json và thư mục ảnh images/
# Tự động nạp đồng thời 2 góc chụp X-quang (Frontal + Lateral) với kích thước chuẩn 448x448

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class IUXrayDataset(Dataset):
    def __init__(self, processed_json: str, image_root: str, max_len: int = 60, img_size: int = 448):
        with open(processed_json, "r", encoding="utf-8") as f:
            self.items = json.load(f)
        self.image_root = image_root
        self.max_len = max_len
        # Chuẩn hóa ảnh: chuyển sang Grayscale 1 kênh, resize 448x448 và đưa về khoảng [-1, 1]
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __len__(self):
        return len(self.items)

    def _load_image(self, path):
        # Đọc ảnh từ đĩa và áp dụng biến đổi tensor
        full = os.path.join(self.image_root, path)
        img = Image.open(full).convert("L")
        return self.transform(img)

    def __getitem__(self, idx):
        d = self.items[idx]
        # Luôn đảm bảo nạp đủ 2 góc chụp (frontal + lateral); nếu chỉ có 1 ảnh thì tự động nhân bản
        paths = d["image_path"][:2]
        if len(paths) == 1:
            paths = [paths[0], paths[0]]
        # Kích thước tensor ảnh: (2, 1, 448, 448)
        images = torch.stack([self._load_image(p) for p in paths], dim=0)

        # Cắt hoặc đệm padding chuỗi report_ids về độ dài cố định max_len (pad_idx = 0)
        ids = d["report_ids"][: self.max_len]
        ids = ids + [0] * (self.max_len - len(ids))

        # Vector nhị phân nhãn bệnh lý MeSH (float tensor)
        mlc = torch.tensor(d["mlc_labels"], dtype=torch.float)
        return images, torch.tensor(ids, dtype=torch.long), mlc
