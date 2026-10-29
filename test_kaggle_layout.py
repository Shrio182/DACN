"""Test với cấu trúc y hệt bộ Kaggle raddar: CSV + images/ folder."""
import os, csv, subprocess, sys
import numpy as np
from PIL import Image

MOCK = "/tmp/mock_raddar"
os.makedirs(f"{MOCK}/images", exist_ok=True)

# Ảnh giả
for name in ["1001.png", "1002.png", "2001.png"]:
    arr = (np.random.rand(512, 512) * 255).astype(np.uint8)
    Image.fromarray(arr, "L").save(f"{MOCK}/images/{name}")

# indiana_reports.csv (dùng tên cột có chữ hoa để test robust)
with open(f"{MOCK}/indiana_reports.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["uid", "MeSH", "Projection", "findings", "impression"])
    w.writerow([1, "diseases", "Frontal/Lateral", "the heart is enlarged. cardiomegaly present.", "no acute disease."])
    w.writerow([2, "normal", "Frontal", "lungs are clear bilaterally.", "normal chest x-ray."])

# indiana_projections.csv
with open(f"{MOCK}/indiana_projections.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["uid", "projection", "filename"])
    w.writerow([1, "Frontal", "1001.png"])
    w.writerow([1, "Lateral", "1002.png"])
    w.writerow([2, "Frontal", "2001.png"])

# Chạy preprocess (phải tự nhận CSV, không có annotation.json)
print("=== Preprocess (tự nhận CSV) ===")
r = subprocess.run([sys.executable, "data/preprocess_iu_xray.py", "--data_dir", MOCK],
                   capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    print("STDERR:", r.stderr); sys.exit(1)

# Kiểm tra annotation.json có prefix images/
import json
ann = json.load(open(f"{MOCK}/annotation.json"))
print("annotation.json mẫu:", ann[0])
assert ann[0]["image_path"][0].startswith("images/"), "thiếu prefix images/"

# Load dataset + 1 batch qua model
sys.path.insert(0, ".")
from data.tokenizer import SimpleTokenizer
from data.iu_xray_dataset import IUXrayDataset
from torch.utils.data import DataLoader
import torch
from config import ModelConfig, TrainConfig
from model.full_model import MulticriteriaTransformer

tok = SimpleTokenizer.load(f"{MOCK}/vocab.json")
mcfg, tcfg = ModelConfig(), TrainConfig()
mcfg.vocab_size = len(tok); mcfg.use_paired_views = True; tcfg.batch_size = 2
ds = IUXrayDataset(f"{MOCK}/processed_train.json", MOCK, 60, 224)
loader = DataLoader(ds, batch_size=2)
imgs, rep, mlc = next(iter(loader))
print("Batch:", imgs.shape, rep.shape, mlc.shape)
tw = torch.load(f"{MOCK}/term_weights.pt")
model = MulticriteriaTransformer(mcfg, tcfg)
loss, ld = model(imgs, rep, mlc, tw, epoch=2)
loss.backward()
print("Loss:", {k: round(v, 4) for k, v in ld.items()})
print("\nOK — Bộ Kaggle (CSV + images/) cắm vào chạy được.")
