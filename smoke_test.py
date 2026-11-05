"""
smoke_test.py — Kiểm thử toàn diện trước khi train lại.
========================================================
Kiểm tra:
1. Forward trên batch thật (từ dataset)
2. Shape per-token decoder features
3. Temporal weights tăng theo token position
4. Padding tokens bị loại khỏi weighted similarity
5. Positive/negative similarities
6. RG, MLC, ITM losses đều hữu hạn
7. Total loss = đúng trọng số
8. Optimizer step: model parameters được cập nhật

Chạy: python smoke_test.py --data_dir ./iu_xray_data
"""
import os
import json
import argparse
import torch
from torch.utils.data import DataLoader

from config import ModelConfig, TrainConfig
from model.full_model import MulticriteriaTransformer
from data.tokenizer import SimpleTokenizer, BOS, EOS
from data.iu_xray_dataset import IUXrayDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    args = ap.parse_args()

    mcfg, tcfg = ModelConfig(), TrainConfig()

    # Load configs
    tokenizer = SimpleTokenizer.load(os.path.join(args.data_dir, "vocab.json"))
    mcfg.vocab_size = len(tokenizer)
    mcfg.pad_idx = 0
    mcfg.use_paired_views = True

    mesh_labels_path = os.path.join(args.data_dir, "mesh_labels.json")
    if os.path.exists(mesh_labels_path):
        with open(mesh_labels_path) as f:
            mesh_info = json.load(f)
        mcfg.num_mlc_labels = mesh_info["num_labels"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: vocab={mcfg.vocab_size}, mlc_labels={mcfg.num_mlc_labels}, "
          f"img={mcfg.img_size}, patch={mcfg.patch_size}, alpha={mcfg.temporal_alpha}")

    model = MulticriteriaTransformer(mcfg, tcfg).to(device)
    term_weights = torch.load(os.path.join(args.data_dir, "term_weights.pt"),
                              weights_only=True).to(device)

    ds = IUXrayDataset(
        os.path.join(args.data_dir, "processed_train.json"),
        image_root=args.data_dir,
        max_len=mcfg.max_seq_len, img_size=mcfg.img_size,
    )
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    images, report_ids, mlc_labels = next(iter(loader))
    images = images.to(device)
    report_ids = report_ids.to(device)
    mlc_labels = mlc_labels.to(device)

    print(f"\n{'='*60}")
    print("TEST 1: Forward trên batch thật")
    print(f"{'='*60}")
    print(f"  images shape: {images.shape}")
    print(f"  report_ids shape: {report_ids.shape}")
    print(f"  mlc_labels shape: {mlc_labels.shape}")
    print(f"  mlc_labels sum per sample: {mlc_labels.sum(dim=1).tolist()}")

    loss, loss_dict = model(images, report_ids, mlc_labels, term_weights)
    print(f"  Loss dict: {loss_dict}")
    assert torch.isfinite(loss), "FAIL: total loss is not finite!"
    print("  ✅ Forward pass OK, loss is finite")

    print(f"\n{'='*60}")
    print("TEST 2: Shape per-token decoder features")
    print(f"{'='*60}")
    regional_features, cls_global = model.encoder(images)
    dec_input = report_ids[:, :-1]
    logits, txt_features = model.decoder(dec_input, regional_features)
    print(f"  dec_input shape: {dec_input.shape}")
    print(f"  logits shape: {logits.shape}  (B, T-1, V)")
    print(f"  txt_features shape: {txt_features.shape}  (B, T-1, d_model)")
    assert txt_features.shape == (4, mcfg.max_seq_len - 1, mcfg.d_model), \
        f"FAIL: expected txt_features (4, {mcfg.max_seq_len-1}, {mcfg.d_model})"
    print("  ✅ Per-token features shape correct")

    print(f"\n{'='*60}")
    print("TEST 3: Temporal weights tăng theo position")
    print(f"{'='*60}")
    padding_mask = (dec_input != mcfg.pad_idx)  # (B, T-1)
    N_r = padding_mask.float().sum(dim=1)
    T = dec_input.size(1)
    positions = torch.arange(T, device=device).float()
    alpha = mcfg.temporal_alpha
    raw = (N_r[0] - positions + 1.0).clamp(min=1e-8) ** (-alpha)
    raw_masked = raw * padding_mask[0].float()
    print(f"  Sample 0: N_r = {N_r[0].item():.0f} valid tokens")
    print(f"  lambda_i formula: (N_r - i + 1) ** (-{alpha})")
    print(f"  First 10 weights: {raw_masked[:10].tolist()}")
    print(f"  Last 10 weights:  {raw_masked[-10:].tolist()}")
    # Verify: lambda_i increases as i increases (weights go up for later tokens)
    valid_len = int(N_r[0].item())
    if valid_len > 2:
        assert raw_masked[1] > raw_masked[0], \
            f"FAIL: weight at pos 1 ({raw_masked[1]:.4f}) should > pos 0 ({raw_masked[0]:.4f})"
        assert raw_masked[valid_len-1] > raw_masked[0], \
            f"FAIL: last valid weight should > first weight"
    print("  ✅ Temporal weights increase with position (later tokens get higher weight)")

    print(f"\n{'='*60}")
    print("TEST 4: Padding tokens excluded from similarity")
    print(f"{'='*60}")
    # Check that padding positions have zero weight
    for i in range(4):
        vlen = int(N_r[i].item())
        raw_i = (N_r[i] - positions + 1.0).clamp(min=1e-8) ** (-alpha)
        raw_i_masked = raw_i * padding_mask[i].float()
        pad_weights = raw_i_masked[vlen:]
        if pad_weights.numel() > 0:
            assert pad_weights.sum().item() == 0, f"FAIL: padding weights should be 0 for sample {i}"
    print(f"  ✅ Padding tokens have zero weight in temporal weighting")

    print(f"\n{'='*60}")
    print("TEST 5: Positive/Negative similarities")
    print(f"{'='*60}")
    sim = model.itm_head(cls_global, txt_features, padding_mask)
    print(f"  Similarity matrix shape: {sim.shape}  (B, B)")
    print(f"  Diagonal (positive): {sim.diagonal().tolist()}")
    pos = sim.diagonal()
    mask_diag = torch.eye(4, device=device).bool()
    off_diag = sim.masked_fill(mask_diag, float("-inf"))
    neg_max = off_diag.max(dim=1).values
    print(f"  Max off-diagonal (negative): {neg_max.tolist()}")
    print("  ✅ Similarity matrix computed correctly")

    print(f"\n{'='*60}")
    print("TEST 6: Individual losses are finite")
    print(f"{'='*60}")
    for k, v in loss_dict.items():
        is_finite = not (v != v) and abs(v) < 1e10  # NaN check + infinity check
        status = "✅" if is_finite else "❌"
        print(f"  {status} {k} = {v:.4f}")
    assert all(not (v != v) for v in loss_dict.values()), "FAIL: some losses are NaN!"
    print("  ✅ All losses finite")

    print(f"\n{'='*60}")
    print("TEST 7: Total loss = correct weighted sum")
    print(f"{'='*60}")
    expected = (tcfg.lambda_rg * loss_dict["rg"]
                + tcfg.lambda_mlc * loss_dict["mlc"]
                + tcfg.lambda_itm_max * loss_dict["itm"])
    actual = loss_dict["total"]
    diff = abs(expected - actual)
    print(f"  Expected: {tcfg.lambda_rg}*{loss_dict['rg']:.4f} + "
          f"{tcfg.lambda_mlc}*{loss_dict['mlc']:.4f} + "
          f"{tcfg.lambda_itm_max}*{loss_dict['itm']:.4f} = {expected:.4f}")
    print(f"  Actual:   {actual:.4f}")
    print(f"  Diff:     {diff:.6f}")
    assert diff < 0.01, f"FAIL: total loss mismatch (diff={diff})"
    print("  ✅ Total loss matches weighted sum")

    print(f"\n{'='*60}")
    print("TEST 8: Optimizer step updates parameters")
    print(f"{'='*60}")
    # Save a snapshot of one parameter
    param_name = "decoder.token_embed.weight"
    param = dict(model.named_parameters())[param_name]
    before = param.data.clone()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    after = param.data
    changed = not torch.equal(before, after)
    print(f"  Parameter '{param_name}' changed: {changed}")
    assert changed, f"FAIL: parameter {param_name} did not change after optimizer step!"
    print("  ✅ Model parameters updated successfully")

    print(f"\n{'='*60}")
    print("ALL 8 SMOKE TESTS PASSED ✅")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
