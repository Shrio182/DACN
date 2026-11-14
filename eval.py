# eval.py
# Đánh giá hiệu năng mô hình trên tập kiểm thử (Test Set).
# Quy trình:
# 1. Nạp mô hình đã huấn luyện từ checkpoint (.pt) và từ vựng (vocab.json).
# 2. Chạy thuật toán sinh báo cáo tự động cho từng cặp ảnh X-quang.
# 3. Tính toán toàn bộ các thước đo ngôn ngữ y khoa tiêu chuẩn:
#    - BLEU-1, BLEU-2, BLEU-3, BLEU-4 (Độ trùng khớp cụm n-gram)
#    - ROUGE-L (Độ tương đồng chuỗi con chung dài nhất)
#    - CIDEr (Đánh giá mức độ đồng thuận với văn bản chuyên môn y khoa)
#    - METEOR (Đo lường có xét đến từ đồng nghĩa và biến thể từ)

import os
import argparse
import torch
from torch.utils.data import DataLoader

from config import ModelConfig, TrainConfig
from model.full_model import MulticriteriaTransformer
from data.tokenizer import SimpleTokenizer, BOS, EOS
from data.iu_xray_dataset import IUXrayDataset


def compute_metrics(gts, res):
    # Tính các chỉ số đánh giá bằng thư viện pycocoevalcap
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider

    results = {}

    # Đánh giá chỉ số BLEU từ 1 đến 4
    bleu_scorer = Bleu(4)
    bleu_scores, _ = bleu_scorer.compute_score(gts, res)
    for i, s in enumerate(bleu_scores):
        results[f"BLEU-{i+1}"] = s

    # Đánh giá chỉ số ROUGE-L
    rouge_scorer = Rouge()
    rouge_score, _ = rouge_scorer.compute_score(gts, res)
    results["ROUGE-L"] = rouge_score

    # Đánh giá chỉ số CIDEr
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(gts, res)
    results["CIDEr"] = cider_score

    # Đánh giá chỉ số METEOR (cần Java Runtime Environment)
    try:
        import threading
        from pycocoevalcap.meteor.meteor import Meteor

        meteor_result = [None]
        def _run_meteor():
            try:
                m = Meteor()
                score, _ = m.compute_score(gts, res)
                m.meteor_p.kill()
                meteor_result[0] = score
            except Exception:
                meteor_result[0] = None

        t = threading.Thread(target=_run_meteor, daemon=True)
        t.start()
        t.join(timeout=60)
        if meteor_result[0] is not None:
            results["METEOR"] = meteor_result[0]
        else:
            results["METEOR"] = float("nan")
    except Exception:
        results["METEOR"] = float("nan")

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Thư mục dữ liệu đã tiền xử lý")
    ap.add_argument("--checkpoint", required=True, help="Đường dẫn file trọng số checkpoint (.pt)")
    ap.add_argument("--batch_size", type=int, default=8, help="Batch size khi suy luận")
    ap.add_argument("--split", default="test", choices=["val", "test"], help="Tập dữ liệu đánh giá")
    args = ap.parse_args()

    mcfg, tcfg = ModelConfig(), TrainConfig()
    tokenizer = SimpleTokenizer.load(os.path.join(args.data_dir, "vocab.json"))
    mcfg.vocab_size = len(tokenizer)
    mcfg.use_paired_views = True

    mesh_labels_path = os.path.join(args.data_dir, "mesh_labels.json")
    if os.path.exists(mesh_labels_path):
        import json as _json
        with open(mesh_labels_path, "r", encoding="utf-8") as f:
            mesh_info = _json.load(f)
        mcfg.num_mlc_labels = mesh_info["num_labels"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị tính toán: {device}")

    # Nạp kiến trúc và trọng số mô hình
    model = MulticriteriaTransformer(mcfg, tcfg)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model = model.to(device)
    model.eval()

    # Chuẩn bị dữ liệu tập kiểm thử
    split_json = os.path.join(args.data_dir, f"processed_{args.split}.json")
    ds = IUXrayDataset(split_json, args.data_dir, mcfg.max_seq_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Đánh giá trên tập {args.split}: {len(ds)} mẫu")

    gts = {}
    res = {}
    bos_idx = tokenizer.vocab[BOS]
    eos_idx = tokenizer.vocab[EOS]

    idx = 0
    with torch.no_grad():
        for batch_i, (images, report_ids, _) in enumerate(loader):
            images = images.to(device)
            out_ids = model.generate(images, bos_idx, eos_idx)
            for i in range(out_ids.size(0)):
                hyp = tokenizer.decode(out_ids[i].tolist())
                ref = tokenizer.decode(report_ids[i].tolist())
                gts[idx] = [ref]
                res[idx] = [hyp]
                idx += 1
            if (batch_i + 1) % 10 == 0:
                print(f"  Đã hoàn thành sinh {idx}/{len(ds)} mẫu...")

    print(f"\nĐã sinh xong toàn bộ {idx} báo cáo. Bắt đầu tính toán metrics...\n")

    # In ra một số mẫu đối chiếu thực tế
    print("=" * 60)
    print("MẪU ĐỐI CHIẾU (Dự đoán vs Thực tế)")
    print("=" * 60)
    for i in range(min(3, idx)):
        print(f"\n[Ca bệnh số {i + 1}]")
        print(f"Mô hình sinh: {res[i][0]}")
        print(f"Bác sĩ ghi:   {gts[i][0]}")

    print("\n" + "=" * 60)
    print("BẢNG ĐIỂM ĐÁNH GIÁ CHẤT LƯỢNG")
    print("=" * 60)

    try:
        metrics = compute_metrics(gts, res)
        print(f"\n{'Chỉ số':<12} {'Điểm':>10}")
        print("-" * 24)
        for name, score in metrics.items():
            print(f"{name:<12} {score:>10.4f}")
        print("-" * 24)
    except Exception as e:
        print(f"[LỖI] Không tính được metrics: {e}")


if __name__ == "__main__":
    main()
