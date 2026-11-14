# plot_comparison.py
# Kịch bản trực quan hóa so sánh đường cong huấn luyện và hiệu năng đánh giá giữa các mô hình.
# Tự động quét toàn bộ thư mục artifacts/*/ để đọc:
#   - train_log_*.csv: Tiến trình giảm hàm mất mát (Loss) và tăng điểm CIDEr qua từng Epoch.
#   - test_results_*.txt: Bộ chỉ số đánh giá đầy đủ trên tập Test (BLEU-1..4, ROUGE-L, CIDEr, METEOR).
# Xuất biểu đồ chuẩn báo cáo khoa học độ phân giải 300 DPI vào thư mục outputs/charts/.

import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Cấu hình font và style cho matplotlib
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.9

MODEL_DISPLAY_NAMES = {
    "archive": "IU-Xray Archive (Full)",
    "timphoi": "IU-Xray TimPhoi",
    "mimic": "MIMIC-CXR",
}

PALETTE = {
    "archive": "#1f77b4",   # Xanh dương
    "timphoi": "#ff7f0e",   # Cam
    "mimic": "#2ca02c",     # Xanh lá
    "default": "#9467bd"    # Tím
}


def get_color(key: str, idx: int = 0) -> str:
    if key in PALETTE:
        return PALETTE[key]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    return colors[idx % len(colors)]


def find_artifact_models(artifacts_dir: str = "artifacts"):
    # Quét thư mục artifacts/ để tìm các mô hình có log và kết quả
    models_data = {}
    if not os.path.exists(artifacts_dir):
        print(f"[CẢNH BÁO] Không tìm thấy thư mục {artifacts_dir}")
        return models_data

    for item in os.listdir(artifacts_dir):
        sub_dir = os.path.join(artifacts_dir, item)
        if not os.path.isdir(sub_dir):
            continue

        model_key = item.lower()
        display_name = MODEL_DISPLAY_NAMES.get(model_key, f"Model {item.capitalize()}")

        # Tìm file train log
        log_files = glob.glob(os.path.join(sub_dir, "train_log*.csv"))
        log_path = log_files[0] if log_files else None

        # Tìm file test results
        test_files = glob.glob(os.path.join(sub_dir, "test_results*.txt"))
        test_path = test_files[0] if test_files else None

        models_data[model_key] = {
            "display_name": display_name,
            "dir": sub_dir,
            "train_log_path": log_path,
            "test_results_path": test_path,
        }

    return models_data


def parse_train_log(csv_path: str):
    # Đọc file train_log_*.csv và lọc lấy epoch, loss, val_cider
    if not csv_path or not os.path.exists(csv_path):
        return None

    try:
        df = pd.read_csv(csv_path)
        # Lọc các dòng tổng kết epoch (step == 'end') nếu có cột step
        if "step" in df.columns:
            epoch_rows = df[df["step"].astype(str) == "end"].copy()
            if epoch_rows.empty:
                epoch_rows = df.copy()
        else:
            epoch_rows = df.copy()

        # Ép kiểu số
        epoch_rows["epoch"] = pd.to_numeric(epoch_rows["epoch"], errors="coerce")
        epoch_rows["total"] = pd.to_numeric(epoch_rows["total"], errors="coerce")
        epoch_rows["val_cider"] = pd.to_numeric(epoch_rows["val_cider"], errors="coerce")

        epoch_rows = epoch_rows.dropna(subset=["epoch", "total"]).sort_values("epoch")
        return epoch_rows
    except Exception as e:
        print(f"Lỗi khi đọc file log {csv_path}: {e}")
        return None


def parse_test_results(txt_path: str):
    # Đọc file test_results_*.txt và trích xuất điểm số các metrics
    if not txt_path or not os.path.exists(txt_path):
        return {}

    metrics = {}
    metric_keys = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-L", "CIDEr", "METEOR"]

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    for m in metric_keys:
        # Regex tìm dòng như: BLEU-1           0.3110
        pattern = rf"\b{re.escape(m)}\b\s*[:=-]?\s*([0-9]+\.[0-9]+)"
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            metrics[m] = float(match.group(1))

    return metrics


def plot_training_curves(models_data: dict, output_dir: str):
    # Vẽ 2 biểu đồ đường so sánh Training Loss và Validation CIDEr
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    fig.patch.set_facecolor("#ffffff")

    ax_loss = axes[0]
    ax_cider = axes[1]

    has_data = False
    for i, (key, info) in enumerate(models_data.items()):
        df = parse_train_log(info["train_log_path"])
        if df is None or df.empty:
            continue

        has_data = True
        color = get_color(key, i)
        label = info["display_name"]

        epochs = df["epoch"].values
        loss = df["total"].values
        cider = df["val_cider"].values

        # Subplot 1: Training Loss
        ax_loss.plot(epochs, loss, marker="o", markersize=4, linewidth=2.0, color=color, label=label)

        # Subplot 2: Val CIDEr
        ax_cider.plot(epochs, cider, marker="s", markersize=4, linewidth=2.0, color=color, label=label)

        # Đánh dấu đỉnh CIDEr tốt nhất
        if len(cider) > 0 and not np.isnan(cider).all():
            best_idx = np.nanargmax(cider)
            best_epoch = epochs[best_idx]
            best_cider = cider[best_idx]
            ax_cider.scatter(
                [best_epoch], [best_cider],
                color=color, s=110, zorder=5, edgecolors="#ffffff", linewidth=1.5
            )
            ax_cider.annotate(
                f"{best_cider:.3f}",
                (best_epoch, best_cider),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=8.5,
                fontweight="bold",
                color=color
            )

    if not has_data:
        print("Không có dữ liệu train log hợp lệ để vẽ biểu đồ loss/CIDEr.")
        plt.close(fig)
        return

    # Định dạng trục Loss
    ax_loss.set_title("So sánh Training Loss theo Epoch", fontsize=13, fontweight="bold", pad=12)
    ax_loss.set_xlabel("Epoch", fontsize=11, labelpad=8)
    ax_loss.set_ylabel(r"Total Loss ($\mathcal{L}_{total}$)", fontsize=11, labelpad=8)
    ax_loss.grid(True, linestyle="--", alpha=0.5)
    ax_loss.legend(frameon=True, facecolor="#f8f9fa", edgecolor="#dddddd", fontsize=9.5)

    # Định dạng trục CIDEr
    ax_cider.set_title("So sánh Validation CIDEr theo Epoch", fontsize=13, fontweight="bold", pad=12)
    ax_cider.set_xlabel("Epoch", fontsize=11, labelpad=8)
    ax_cider.set_ylabel("CIDEr Score", fontsize=11, labelpad=8)
    ax_cider.grid(True, linestyle="--", alpha=0.5)
    ax_cider.legend(frameon=True, facecolor="#f8f9fa", edgecolor="#dddddd", fontsize=9.5)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "training_curves_comparison.png")
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"✅ Đã lưu biểu đồ Training Curves: {out_path}")


def plot_test_metrics_bar(models_data: dict, output_dir: str):
    # Vẽ biểu đồ cột so sánh điểm Test giữa các bộ dữ liệu
    metric_keys = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-L", "CIDEr", "METEOR"]
    models_scores = {}

    for key, info in models_data.items():
        res = parse_test_results(info["test_results_path"])
        if res:
            models_scores[key] = {
                "display_name": info["display_name"],
                "scores": res
            }

    if not models_scores:
        print("Không tìm thấy kết quả test hợp lệ để vẽ biểu đồ cột.")
        return

    # Lọc những metric có ít nhất 1 model có điểm
    active_metrics = [m for m in metric_keys if any(m in data["scores"] for data in models_scores.values())]
    if not active_metrics:
        print("Không có metric nào được tìm thấy.")
        return

    fig, ax = plt.subplots(figsize=(13, 6), dpi=300)
    fig.patch.set_facecolor("#ffffff")

    x = np.arange(len(active_metrics))
    num_models = len(models_scores)
    width = 0.75 / max(num_models, 1)

    for i, (key, data) in enumerate(models_scores.items()):
        color = get_color(key, i)
        offset = (i - (num_models - 1) / 2) * width
        scores = [data["scores"].get(m, 0.0) for m in active_metrics]

        bars = ax.bar(
            x + offset, scores, width,
            label=data["display_name"],
            color=color, edgecolor="#ffffff", linewidth=1.2, alpha=0.9
        )

        # Hiển thị giá trị trên đỉnh cột
        for bar in bars:
            height = bar.get_height()
            if height > 0.01:
                ax.annotate(
                    f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", color="#333333", rotation=0
                )

    ax.set_title("So sánh điểm số Test Metrics giữa các tập dữ liệu / mô hình", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Score", fontsize=11, labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(active_metrics, fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, max(0.55, ax.get_ylim()[1] * 1.12))
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend(frameon=True, facecolor="#f8f9fa", edgecolor="#dddddd", fontsize=10, loc="upper right")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "test_metrics_bar_chart.png")
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"✅ Đã lưu biểu đồ Test Metrics Bar Chart: {out_path}")


def plot_overall_dashboard(models_data: dict, output_dir: str):
    # Vẽ dashboard tổng hợp kết hợp training loss, val cider và bảng metrics
    fig = plt.figure(figsize=(16, 10), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0], hspace=0.3, wspace=0.22)

    ax_loss = fig.add_subplot(gs[0, 0])
    ax_cider = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, :])

    # 1. Training curves
    for i, (key, info) in enumerate(models_data.items()):
        df = parse_train_log(info["train_log_path"])
        if df is None or df.empty:
            continue
        color = get_color(key, i)
        label = info["display_name"]
        ax_loss.plot(df["epoch"], df["total"], marker="o", markersize=3.5, linewidth=1.8, color=color, label=label)
        ax_cider.plot(df["epoch"], df["val_cider"], marker="s", markersize=3.5, linewidth=1.8, color=color, label=label)

    ax_loss.set_title("(A) Tiến trình Training Loss qua các Epoch", fontsize=12, fontweight="bold", pad=10)
    ax_loss.set_xlabel("Epoch", fontsize=10)
    ax_loss.set_ylabel("Total Loss", fontsize=10)
    ax_loss.grid(True, linestyle="--", alpha=0.5)
    ax_loss.legend(frameon=True, fontsize=9)

    ax_cider.set_title("(B) Tiến trình Validation CIDEr qua các Epoch", fontsize=12, fontweight="bold", pad=10)
    ax_cider.set_xlabel("Epoch", fontsize=10)
    ax_cider.set_ylabel("CIDEr", fontsize=10)
    ax_cider.grid(True, linestyle="--", alpha=0.5)
    ax_cider.legend(frameon=True, fontsize=9)

    # 2. Test metrics bar chart
    metric_keys = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-L", "CIDEr", "METEOR"]
    models_scores = {k: {"display_name": v["display_name"], "scores": parse_test_results(v["test_results_path"])}
                     for k, v in models_data.items() if parse_test_results(v["test_results_path"])}

    if models_scores:
        x = np.arange(len(metric_keys))
        num_m = len(models_scores)
        width = 0.75 / max(num_m, 1)

        for i, (key, data) in enumerate(models_scores.items()):
            color = get_color(key, i)
            offset = (i - (num_m - 1) / 2) * width
            scores = [data["scores"].get(m, 0.0) for m in metric_keys]
            bars = ax_bar.bar(x + offset, scores, width, label=data["display_name"], color=color, alpha=0.9)
            for bar in bars:
                h = bar.get_height()
                if h > 0.01:
                    ax_bar.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                                    fontsize=8, fontweight="bold", rotation=0)

        ax_bar.set_title("(C) So sánh định lượng hiệu năng trên tập Test", fontsize=12, fontweight="bold", pad=10)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(metric_keys, fontsize=10, fontweight="bold")
        ax_bar.set_ylabel("Score", fontsize=10)
        ax_bar.set_ylim(0, max(0.55, ax_bar.get_ylim()[1] * 1.12))
        ax_bar.grid(axis="y", linestyle="--", alpha=0.5)
        ax_bar.legend(frameon=True, fontsize=9.5, loc="upper right")

    fig.suptitle("TỔNG HỢP SO SÁNH HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH (IEEE TMI 2022)",
                 fontsize=15, fontweight="bold", y=0.98)

    out_path = os.path.join(output_dir, "overall_benchmark_dashboard.png")
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"✅ Đã lưu Dashboard tổng hợp: {out_path}")


def main():
    charts_dir = os.path.join("outputs", "charts")
    os.makedirs(charts_dir, exist_ok=True)

    print("=" * 65)
    print("BẮT ĐẦU VẼ BIỂU ĐỒ SO SÁNH CÁC MÔ HÌNH TỪ ARTIFACTS/")
    print("=" * 65)

    models_data = find_artifact_models("artifacts")
    print(f"Đã phát hiện {len(models_data)} mô hình:")
    for k, v in models_data.items():
        print(f"  - [{k}] {v['display_name']}:")
        print(f"      + Train Log:    {v['train_log_path']}")
        print(f"      + Test Results: {v['test_results_path']}")

    plot_training_curves(models_data, charts_dir)
    plot_test_metrics_bar(models_data, charts_dir)
    plot_overall_dashboard(models_data, charts_dir)

    print("\n" + "=" * 65)
    print(f"HOÀN TẤT. Toàn bộ biểu đồ chất lượng cao 300 DPI đã lưu tại:\n  {os.path.abspath(charts_dir)}")
    print("=" * 65)


if __name__ == "__main__":
    main()
