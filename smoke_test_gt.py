"""
Smoke test: Tra cứu 1 mẫu từ mỗi tập ground_truth (mimic, iu_archive, iu_timphoi).
Chạy: python smoke_test_gt.py
"""
import sys, json, csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
GT_DIR   = BASE_DIR / "data" / "ground_truth"

PASS = "[PASS]"
FAIL = "[FAIL]"

errors = []

# ── 1. iu_timphoi annotation.json ──
ann_path = GT_DIR / "iu_timphoi" / "annotation.json"
try:
    with open(ann_path, encoding="utf-8") as f:
        ann = json.load(f)
    first_split = next(iter(ann))
    sample = ann[first_split][0]
    uid    = sample.get("id", "")
    report = sample.get("report", "")
    assert uid and report, "Missing id or report"
    print(f"{PASS} iu_timphoi  — split='{first_split}'  id='{uid}'  report[:60]='{report[:60]}'")
except Exception as e:
    print(f"{FAIL} iu_timphoi: {e}")
    errors.append(str(e))

# ── 2. iu_archive indiana_reports.csv ──
rep_path  = GT_DIR / "iu_archive" / "indiana_reports.csv"
proj_path = GT_DIR / "iu_archive" / "indiana_projections.csv"
try:
    with open(rep_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    sample = rows[1]
    uid    = sample.get("uid", "")
    findings = sample.get("findings", "")[:60]
    impression = sample.get("impression", "")[:60]
    assert uid, "Missing uid"
    print(f"{PASS} iu_archive  — uid='{uid}'  findings[:60]='{findings}'  impression[:60]='{impression}'")

    with open(proj_path, encoding="utf-8") as f:
        prows = list(csv.DictReader(f))
    prow = prows[0]
    print(f"{PASS} iu_projections — uid='{prow.get('uid','')}' filename='{prow.get('filename','')}' proj='{prow.get('projection','')}'")
except Exception as e:
    print(f"{FAIL} iu_archive: {e}")
    errors.append(str(e))

# ── 3. mimic train.txt (JSON array) ──
train_path = GT_DIR / "mimic" / "train.txt"
try:
    with open(train_path, encoding="utf-8") as f:
        raw = f.read().strip()
    import json as _json
    parsed = _json.loads(raw)
    samples = parsed if isinstance(parsed, list) else list(parsed.values())[0]
    sample = samples[0]
    sid    = sample.get("study_id", sample.get("id", ""))
    report = sample.get("report", "")
    if isinstance(report, list): report = report[0]
    report_str = str(report)[:80]
    assert sid, "Missing study_id"
    print(f"{PASS} mimic/train — study_id='{sid}'  report[:80]='{report_str}'")
except Exception as e:
    print(f"{FAIL} mimic train: {e}")
    errors.append(str(e))

# ── 4. mimic val/test ──
for split in ("val", "test"):
    p = GT_DIR / "mimic" / f"{split}.txt"
    try:
        with open(p, encoding="utf-8") as f:
            raw = f.read().strip()
        import json as _json
        parsed = _json.loads(raw)
        samples = parsed if isinstance(parsed, list) else list(parsed.values())[0]
        sample = samples[0]
        sid = sample.get("study_id", sample.get("id", ""))
        print(f"{PASS} mimic/{split}  — study_id='{sid}'")
    except Exception as e:
        print(f"{FAIL} mimic/{split}: {e}")
        errors.append(str(e))

print()
if errors:
    print(f"=== {len(errors)} TEST(S) FAILED ===")
    sys.exit(1)
else:
    print("=== ALL SMOKE TESTS PASSED ===")
    sys.exit(0)
