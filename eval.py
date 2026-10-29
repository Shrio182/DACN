"""
eval.py — Đánh giá trên tập test: sinh báo cáo -> tính BLEU/METEOR/ROUGE-L/CIDEr.
Yêu cầu: pip install pycocoevalcap
Chạy: python eval.py --data_dir ./iu_xray_data --checkpoint checkpoints/best.pt
"""
import os
import argparse
import torch
from torch.utils.data import DataLoader

from config import ModelConfig, TrainConfig
from model.full_model import MulticriteriaTransformer
from data.tokenizer import SimpleTokenizer, BOS, EOS
from data.iu_xray_dataset import IUXrayDataset


def compute_metrics(gts, res):
    """Tính BLEU-1→4, METEOR, ROUGE-L, CIDEr từ dict gts và res.
    gts: {id: [ref_str, ...]}
    res: {id: [hyp_str]}

    Lưu ý: METEOR cần Java subprocess — nếu bị treo, bỏ qua và dùng --no_meteor.
    """
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider

    results = {}

    # BLEU-1 → 4
    bleu_scorer = Bleu(4)
    bleu_scores, _ = bleu_scorer.compute_score(gts, res)
    for i, s in enumerate(bleu_scores):
        results[f"BLEU-{i+1}"] = s

    # ROUGE-L
    rouge_scorer = Rouge()
    rouge_score, _ = rouge_scorer.compute_score(gts, res)
    results["ROUGE-L"] = rouge_score

    # CIDEr
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(gts, res)
    results["CIDEr"] = cider_score

    # METEOR (tuỳ chọn — cần Java subprocess)
    try:
        import signal, threading
        from pycocoevalcap.meteor.meteor import Meteor

        meteor_result = [None]
        def _run_meteor():
            try:
                m = Meteor()
                score, _ = m.compute_score(gts, res)
                m.meteor_p.kill()   # tắt Java subprocess
                meteor_result[0] = score
            except Exception:
                meteor_result[0] = None

        t = threading.Thread(target=_run_meteor, daemon=True)
        t.start()
        t.join(timeout=60)   # tối đa 60 giây
        if meteor_result[0] is not None:
            results["METEOR"] = meteor_result[0]
        else:
            results["METEOR"] = float("nan")
            print("  [WARNING] METEOR timeout/error — bỏ qua.")
    except Exception as e:
        results["METEOR"] = float("nan")
        print(f"  [WARNING] METEOR không tính được: {e}")

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--split", default="test", choices=["val", "test"],
                    help="Tập dữ liệu để đánh giá")
    args = ap.parse_args()

    mcfg, tcfg = ModelConfig(), TrainConfig()
    tokenizer = SimpleTokenizer.load(os.path.join(args.data_dir, "vocab.json"))
    mcfg.vocab_size = len(tokenizer)
    mcfg.use_paired_views = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị: {device}")

    model = MulticriteriaTransformer(mcfg, tcfg)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model = model.to(device)
    model.eval()

    split_json = os.path.join(args.data_dir, f"processed_{args.split}.json")
    ds = IUXrayDataset(split_json, args.data_dir, mcfg.max_seq_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Đánh giá trên tập {args.split}: {len(ds)} mẫu")

    gts = {}   # ground truth: {idx: [ref_text]}
    res = {}   # generated:    {idx: [hyp_text]}
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
                print(f"  Đã xử lý {idx}/{len(ds)} mẫu...")

    print(f"\nĐã sinh xong {idx} báo cáo. Đang tính metrics...\n")

    # ---- In vài ví dụ ----
    print("=" * 60)
    print("VÍ DỤ (Generated vs Ground-truth)")
    print("=" * 60)
    for i in range(min(5, idx)):
        print(f"\n--- Mẫu {i} ---")
        print(f"GEN : {res[i][0]}")
        print(f"GT  : {gts[i][0]}")

    # ---- Tính metrics ----
    print("\n" + "=" * 60)
    print("KẾT QUẢ ĐÁNH GIÁ")
    print("=" * 60)

    try:
        metrics = compute_metrics(gts, res)
        print(f"\n{'Metric':<12} {'Score':>10}")
        print("-" * 24)
        for name, score in metrics.items():
            print(f"{name:<12} {score:>10.4f}")
        print("-" * 24)
        print(f"\nCheckpoint: {args.checkpoint}")
        print(f"Tập đánh giá: {args.split} ({idx} mẫu)")
    except Exception as e:
        print(f"\n[LỖI] Không tính được metrics: {e}")
        print("Kiểm tra: pip install pycocoevalcap")


if __name__ == "__main__":
    main()
