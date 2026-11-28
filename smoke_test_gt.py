"""
Smoke test toàn diện cho DACN:
1. Kiểm tra tra cứu Ground Truth trên cả 3 tập (iu_timphoi, iu_archive, mimic train/val/test)
   + kiểm tra ảnh ngoài tập dữ liệu (negative control).
2. Kiểm tra tải mô hình Pure Transformer, suy luận sinh báo cáo (FINDINGS / IMPRESSION / Raw)
   và sinh Attention Heatmap + ViT Attention Explanation.
Chạy: python smoke_test_gt.py
"""
import json
import sys
import time

import numpy as np
from PIL import Image

import demo
from clinical_utils import format_clinical_report, split_clinical_report

PASS = "[PASS]"
FAIL = "[FAIL]"


def run_tests():
    errors = []

    # ── 1. Ground Truth Lookup Tests ──────────────────────────────────────────
    t0 = time.time()
    # Pick real test sample IDs from each dataset
    with open(demo.GT_DIR / "iu_timphoi" / "annotation.json", encoding="utf-8") as f:
        ann = json.load(f)
    tp_train = ann["train"][0]
    tp_val = ann["val"][0]
    tp_test = ann["test"][0]

    with open(demo.GT_DIR / "mimic" / "test.txt", encoding="utf-8") as f:
        mimic_test_rows = json.load(f)
    mimic_sample = mimic_test_rows[0]
    mimic_multi = mimic_test_rows[4]  # multi-report row_101
    mimic_uuid = demo.Path(mimic_sample["frontal"]).name
    mimic_multi_uuid = demo.Path(mimic_multi["frontal"]).name

    cases = [
        ("iu_timphoi train id", f"{tp_train['id']}.png", "IU-Xray (TimPhoi)", "train"),
        ("iu_timphoi val path", tp_val["image_path"][0].replace("/", "_"), "IU-Xray (TimPhoi)", "val"),
        ("iu_timphoi test uid-prefix", f"hospitalA_{tp_test['id']}_frontal.png", "IU-Xray (TimPhoi)", "test"),
        ("iu_archive dcm.png", "1_IM-0001-4001.dcm.png", "IU X-Ray Archive", "Archive (Full)"),
        ("iu_archive no-dcm uid-prefix", "case_2_IM-0652-1001.png", "IU X-Ray Archive", "Archive (Full)"),
        ("mimic test uuid", mimic_uuid, "MIMIC-CXR", "test"),
        ("mimic multi-report", f"p10014729_{mimic_multi_uuid}", "MIMIC-CXR", "test"),
    ]

    for label, fname, exp_ds, exp_split in cases:
        rec = demo.lookup_ground_truth(fname)
        if not rec or rec["dataset"] != exp_ds or rec["split"] != exp_split:
            msg = f"{label} ({fname}) -> got {rec}"
            print(f"{FAIL} {msg}")
            errors.append(msg)
        elif "['findings:" in rec["findings"] or "', 'findings:" in rec["impression"]:
            msg = f"{label} leaked list syntax in findings/impression"
            print(f"{FAIL} {msg}")
            errors.append(msg)
        else:
            print(
                f"{PASS} GT [{label}]: ds={rec['dataset']} split={rec['split']} "
                f"| FINDINGS={rec['findings'][:55]!r} | IMPRESSION={rec['impression'][:45]!r}"
            )

    # Negative controls: arbitrary filenames outside datasets must return None
    for neg in ("1.png", "0.png", "custom_patient_xray.jpg", "chest_pa_2026.png"):
        rec = demo.lookup_ground_truth(neg)
        if rec is not None:
            msg = f"Negative control '{neg}' falsely matched {rec['dataset']} ({rec['uid']})"
            print(f"{FAIL} {msg}")
            errors.append(msg)
        else:
            print(f"{PASS} Negative control '{neg}' -> None")

    print(f"GT index & lookup elapsed: {time.time() - t0:.2f}s\n")

    # ── 2. Model Inference + Attention Heatmap + Explanation ──────────────────
    all_entries = demo.load_all_models.__wrapped__()
    ok_keys = [k for k, v in all_entries.items() if v and "model" in v]
    if not ok_keys:
        msg = "No models loaded successfully"
        print(f"{FAIL} {msg}")
        errors.append(msg)
        return errors
    print(f"{PASS} Loaded models: {ok_keys}")

    # Synthetic chest-like radiograph (non-square size to test upscaling to original size)
    arr = np.full((512, 640), 45, dtype=np.uint8)
    arr[90:420, 260:380] = 200
    arr[220:380, 240:410] = 220
    arr[280:400, 420:540] = 185
    pil_frontal = Image.fromarray(arr, mode="L").convert("RGB")
    frontal_t = demo.preprocess_image(pil_frontal)

    results = {}
    for key in ok_keys:
        report_en, probs_np = demo.run_model_inference(all_entries[key], frontal_t, None)
        assert isinstance(report_en, str) and len(report_en) > 0, f"Empty output for {key}"
        results[key] = (report_en, probs_np)
        print(f"{PASS} Inference [{key}]: {report_en[:85]}")

    mlc_probs = demo._compute_mlc_consensus(results, all_entries)
    best_key, best_raw = demo.select_best_report(results, all_entries, mlc_probs)
    clean_en = format_clinical_report(best_raw)
    structured = split_clinical_report(clean_en)
    assert structured["findings"] and structured["impression"], "Empty FINDINGS/IMPRESSION"
    print(f"{PASS} Selected [{best_key}] -> FINDINGS: {structured['findings'][:75]} | IMPRESSION: {structured['impression'][:60]}")

    hm_img, attn_info = demo.create_attention_overlay(
        pil_frontal,
        all_entries[best_key]["model"],
        all_entries[best_key]["device"],
        frontal_t,
        None,
    )
    assert hm_img.size == pil_frontal.size, f"Expected heatmap size {pil_frontal.size}, got {hm_img.size}"
    assert len(attn_info["peak_patches"]) == 3 and len(attn_info["top_zones"]) == 3
    html = demo.render_heatmap_explanation(attn_info)
    assert "ViT Attention Explanation" in html and "[CLS]" in html
    top_p = attn_info["peak_patches"][0]
    print(
        f"{PASS} Heatmap size={hm_img.size} | top_patch=({top_p['row']},{top_p['col']}) "
        f"-> {top_p['zone_vi']} ({top_p['rel_pct']}%)"
    )
    return errors


if __name__ == "__main__":
    errs = run_tests()
    print()
    if errs:
        print(f"=== {len(errs)} TEST(S) FAILED ===")
        sys.exit(1)
    print("=== ALL SMOKE TESTS PASSED ===")
    sys.exit(0)
