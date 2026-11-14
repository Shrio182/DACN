# dataset_filter_audit.py
# Kịch bản kiểm tra chi tiết quy trình lọc và phân tách dữ liệu của IU-Xray.
# Đối soát số lượng mẫu thực tế (2.937 mẫu hợp lệ có đủ cặp ảnh và 2 phần Findings/Impression)
# so với con số ~3.195 mẫu được nhóm tác giả đề cập trong bài báo gốc.
import csv
import os
import argparse

DEFAULT_DATA_DIR = "data/archive" if os.path.exists("data/archive/indiana_reports.csv") else "iu_xray_data"


def main():
    parser = argparse.ArgumentParser(description="Audit IU-Xray dataset filtering")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR, help="Thư mục chứa CSV và images")
    args = parser.parse_args()

    data_dir = args.data_dir
    reports_csv = os.path.join(data_dir, "indiana_reports.csv")
    proj_csv = os.path.join(data_dir, "indiana_projections.csv")
    images_dir = os.path.join(data_dir, "images")

    lines = []
    def p(s=""):
        print(s)
        lines.append(s)

    p("=" * 70)
    p(f"DATASET FILTER AUDIT — IU-Xray ({data_dir})")
    p("=" * 70)

    # --- 1. Reports CSV ---
    all_uids = set()
    uid_findings = {}
    uid_impression = {}
    uid_mesh = {}
    dup_uids_report = 0

    with open(reports_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            uid = str(row.get("uid", "")).strip()
            if uid in all_uids:
                dup_uids_report += 1
            all_uids.add(uid)
            uid_findings[uid] = (row.get("findings") or "").strip()
            uid_impression[uid] = (row.get("impression") or "").strip()
            uid_mesh[uid] = (row.get("MeSH") or "").strip()

    total_uids = len(all_uids)
    p(f"\n1. REPORTS CSV")
    p(f"   Total unique report UIDs: {total_uids}")
    p(f"   Duplicate UIDs in reports CSV: {dup_uids_report}")

    # --- Text filtering ---
    missing_findings = sum(1 for u in all_uids if not uid_findings.get(u))
    missing_impression = sum(1 for u in all_uids if not uid_impression.get(u))
    missing_both = sum(1 for u in all_uids if not uid_findings.get(u) and not uid_impression.get(u))
    has_both = sum(1 for u in all_uids if uid_findings.get(u) and uid_impression.get(u))

    p(f"\n2. TEXT FILTERING")
    p(f"   UIDs missing Findings:   {missing_findings}")
    p(f"   UIDs missing Impression: {missing_impression}")
    p(f"   UIDs missing BOTH:       {missing_both}")
    p(f"   UIDs with BOTH (text-valid): {has_both}")
    text_valid_uids = set(u for u in all_uids if uid_findings.get(u) and uid_impression.get(u))

    # --- 2. Projections CSV ---
    proj_uids = set()
    frontal_map = {}  # uid -> [filenames]
    lateral_map = {}  # uid -> [filenames]
    unknown_proj = {}  # projection -> count
    all_filenames = []
    dup_filenames = set()

    with open(proj_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            uid = str(row.get("uid", "")).strip()
            fname = (row.get("filename") or "").strip()
            proj = (row.get("projection") or "").strip().lower()
            proj_uids.add(uid)
            if fname:
                all_filenames.append(fname)
            if not fname:
                continue
            if proj in ("frontal", "pa", "ap"):
                frontal_map.setdefault(uid, []).append(fname)
            elif proj in ("lateral", "ll", "left lateral"):
                lateral_map.setdefault(uid, []).append(fname)
            else:
                unknown_proj[proj] = unknown_proj.get(proj, 0) + 1
                # Fallback: treat as frontal
                frontal_map.setdefault(uid, []).append(fname)

    # Check duplicate filenames
    fname_counts = {}
    for fn in all_filenames:
        fname_counts[fn] = fname_counts.get(fn, 0) + 1
    dup_filenames = {fn: c for fn, c in fname_counts.items() if c > 1}

    p(f"\n3. PROJECTIONS CSV")
    p(f"   Total unique UIDs in projections: {len(proj_uids)}")
    p(f"   Total image filenames: {len(all_filenames)}")
    p(f"   Duplicate image filenames: {len(dup_filenames)}")
    if dup_filenames:
        for fn, c in sorted(dup_filenames.items(), key=lambda x: -x[1])[:10]:
            p(f"     {fn} appears {c} times")
    p(f"   Unknown projection values:")
    if unknown_proj:
        for proj_val, cnt in sorted(unknown_proj.items(), key=lambda x: -x[1]):
            p(f"     '{proj_val}': {cnt} images")
    else:
        p(f"     (none)")

    # --- Image file existence check ---
    if os.path.isdir(images_dir):
        existing_images = set(os.listdir(images_dir))
        broken_paths = []
        for fn in set(all_filenames):
            if fn not in existing_images:
                broken_paths.append(fn)
        p(f"\n4. IMAGE FILE CHECK")
        p(f"   Images on disk: {len(existing_images)}")
        p(f"   Broken image paths (in CSV but not on disk): {len(broken_paths)}")
        if broken_paths:
            for bp in broken_paths[:10]:
                p(f"     MISSING: {bp}")
    else:
        p(f"\n4. IMAGE FILE CHECK")
        p(f"   [WARNING] images/ directory not found at {images_dir}")

    # --- Paired filtering ---
    p(f"\n5. PAIRED IMAGE FILTERING (text-valid UIDs only)")
    text_valid_with_frontal = sum(1 for u in text_valid_uids if u in frontal_map)
    text_valid_with_lateral = sum(1 for u in text_valid_uids if u in lateral_map)
    text_valid_with_both = sum(1 for u in text_valid_uids if u in frontal_map and u in lateral_map)
    text_valid_no_frontal = sum(1 for u in text_valid_uids if u not in frontal_map)
    text_valid_no_lateral = sum(1 for u in text_valid_uids if u not in lateral_map and u in frontal_map)

    p(f"   Text-valid UIDs:           {has_both}")
    p(f"   + has frontal:             {text_valid_with_frontal}")
    p(f"   + has lateral:             {text_valid_with_lateral}")
    p(f"   + has BOTH (final valid):  {text_valid_with_both}")
    p(f"   Lost: no frontal:          {text_valid_no_frontal}")
    p(f"   Lost: has frontal, no lat: {text_valid_no_lateral}")

    # --- Summary ---
    p(f"\n6. SUMMARY")
    p(f"   Total UIDs in reports:     {total_uids}")
    p(f"   After text filter:         {has_both}  (lost {total_uids - has_both})")
    p(f"   After paired image filter: {text_valid_with_both}  (lost {has_both - text_valid_with_both})")
    p(f"   Final valid samples:       {text_valid_with_both}")
    p(f"   Paper target:              3,195")
    p(f"   Difference:                {text_valid_with_both - 3195} ({(text_valid_with_both-3195)/3195*100:+.1f}%)")

    # --- UIDs that have text but no projection data ---
    text_valid_no_proj = text_valid_uids - proj_uids
    p(f"\n7. EDGE CASES")
    p(f"   Text-valid UIDs NOT in projections CSV: {len(text_valid_no_proj)}")
    if text_valid_no_proj:
        for u in sorted(text_valid_no_proj)[:10]:
            p(f"     uid={u}")

    # UIDs in projections but not in reports
    proj_only = proj_uids - all_uids
    p(f"   UIDs in projections but NOT in reports: {len(proj_only)}")

    p("\n" + "=" * 70)

    # Save to file
    out_path = os.path.join(DATA_DIR, "dataset_filter_audit.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nĐã lưu audit → {out_path}")


if __name__ == "__main__":
    main()

