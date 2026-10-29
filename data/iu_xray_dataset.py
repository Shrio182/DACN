"""
iu_xray_dataset.py — PyTorch Dataset cho IU-Xray (đã tiền xử lý).
Đọc processed_<split>.json, ảnh từ thư mục images/, trả về batch cố định kích thước.
"""
import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class IUXrayDataset(Dataset):
    def __init__(self, processed_json: str, image_root: str, max_len: int = 60, img_size: int = 448):
        with open(processed_json) as f:
            self.items = json.load(f)
        self.image_root = image_root
        self.max_len = max_len
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),   # paper: 448×448
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),  # chuẩn hoá về [-1,1]
        ])

    def __len__(self):
        return len(self.items)

    def _load_image(self, path):
        full = os.path.join(self.image_root, path)
        img = Image.open(full).convert("L")
        return self.transform(img)  # (1, H, W)

    def __getitem__(self, idx):
        d = self.items[idx]
        # Luôn stack 2 view (frontal + lateral); nếu chỉ có 1 ảnh thì lặp lại
        paths = d["image_path"][:2]
        if len(paths) == 1:
            paths = [paths[0], paths[0]]
        images = torch.stack([self._load_image(p) for p in paths], dim=0)  # (2,1,H,W)

        # Padding report_ids về max_len (pad_idx=0)
        ids = d["report_ids"][: self.max_len]
        ids = ids + [0] * (self.max_len - len(ids))

        mlc = torch.tensor(d["mlc_labels"], dtype=torch.float)
        return images, torch.tensor(ids, dtype=torch.long), mlc
