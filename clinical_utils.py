# clinical_utils.py
# Tiện ích chuẩn hóa ngắt câu và phân tách báo cáo X-quang thành 2 phần chuẩn bài báo:
# FINDINGS và IMPRESSION.

import re


def format_clinical_report(raw_text: str) -> str:
    """Normalize generated English radiology text into concise clinical sentences."""
    if not raw_text or not isinstance(raw_text, str):
        return ""

    text = raw_text.replace("xxxx", " ").replace("XXXX", " ")
    text = re.sub(r"_{2,}", " ", text)
    text = re.sub(r"<(?:bos|eos|pad|unk)>", " ", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\b(?:in comparison with|compared (?:to|with))\s+(?:the\s+)?(?:prior|previous\s+)?(?:study|radiograph|exam|examination)(?:\s+of\s+[^,.]*)?\s*[,.]?",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*\n+\s*", ". ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*,\s*\.", ",", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*", ", ", text)
    text = re.sub(r"\s*\.\s*", ". ", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\s+", " ", text).strip(" .")

    landmarks = (
        "there is", "there are", "bony structures", "no pneumothorax",
        "no pleural", "no focal", "no acute", "cardiomediastinal",
        "lungs are", "lungs", "heart", "mediastinum", "lung", "pleural",
        "osseous", "pneumothorax", "pneumonia", "cardiomegaly",
        "consolidation", "edema", "effusion", "acute", "impression",
    )
    connectors = {
        "no", "or", "nor", "without", "for", "of", "with", "and",
        "the", "a", "an", "is", "are", "demonstrate", "demonstrates",
        "show", "shows", "reveal", "reveals", "indicate", "suggest", "suggests",
    }
    compound_predecessors = {
        "heart": {"normal", "enlarged", "large"},
        "cardiomegaly": {"mild", "moderate", "borderline"},
        "consolidation": {"focal", "lobar", "airspace"},
        "edema": {"pulmonary"},
        "effusion": {"pleural"},
        "pneumothorax": {"small", "large", "left", "right"},
        "acute": {"no"},
    }
    insertion_points = set()
    for landmark in sorted(landmarks, key=len, reverse=True):
        pattern = re.compile(rf"\b{re.escape(landmark)}\b", re.IGNORECASE)
        for match in pattern.finditer(text):
            start = match.start()
            if landmark in {"heart", "lung", "lungs", "mediastinum", "pleural"}:
                article = re.search(r"\b(the|a|an)\s+$", text[:start], re.IGNORECASE)
                if article:
                    start -= len(article.group(0))
            prefix = text[:start]
            if prefix.endswith(". ") or not prefix:
                continue
            previous = prefix.rstrip()
            if previous.endswith(","):
                continue
            words = re.findall(r"[a-z']+", previous.lower())
            if words and words[-1] in connectors:
                continue
            if words and words[-1] in compound_predecessors.get(landmark, set()):
                continue
            if landmark == "consolidation" and len(words) >= 2:
                if words[-2:] == ["no", "focal"]:
                    continue
            # Preserve negated pathology lists (e.g., "no pneumonia effusions edema pneumothorax ...")
            if landmark in {"pneumothorax", "pneumonia", "consolidation", "edema", "effusion", "cardiomegaly"}:
                last_clause = previous.split(".")[-1].lower()
                if re.search(r"\b(?:no|without|nor)\b", last_clause) and not re.search(
                    r"\b(?:is|are|was|were|seen|identified|present|noted)\b",
                    last_clause.split("no")[-1] if "no" in last_clause else last_clause,
                ):
                    continue

            if start > 0 and not text[:start].endswith(". "):
                insertion_points.add(start)

    for point in sorted(insertion_points, reverse=True):
        text = text[:point].rstrip() + ". " + text[point:].lstrip()

    text = re.sub(r"\s*,\s*\.", ",", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r"\s*\.\s*", ". ", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\s+", " ", text).strip(" .")

    dangling = {
        "and", "or", "nor", "with", "of", "in", "to", "from", "the", "a", "an",
        "has", "have", "had", "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did", "without", "such", "as", "yet", "although", "however",
    }
    sentences = []
    for sentence in (part.strip(" .") for part in text.split(".")):
        sentence = re.sub(
            r"^(?:findings|impression)\s*[:=-]?\s*",
            "",
            sentence,
            flags=re.IGNORECASE,
        ).strip(" ,")
        words = sentence.split()
        was_truncated = False
        ended_with_auxiliary = False
        while words and words[-1].lower() in dangling:
            removed = words.pop().lower()
            was_truncated = True
            ended_with_auxiliary = ended_with_auxiliary or removed in {
                "has", "have", "had", "is", "are", "was", "were", "be", "been",
                "being", "do", "does", "did",
            }
        if was_truncated and "and" in [word.lower() for word in words]:
            last_conjunction = max(
                index for index, word in enumerate(words) if word.lower() == "and"
            )
            words = words[:last_conjunction]
        elif ended_with_auxiliary or len(words) < 3:
            continue
        if len(words) >= 2:
            normalized = " ".join(words)
            normalized = normalized[0].upper() + normalized[1:]
            sentences.append(normalized)

    if not sentences:
        return "Heart size and mediastinal contours are within normal limits. The lungs are clear."
    return ". ".join(sentences) + "."


def split_clinical_report(raw_text: str) -> dict:
    """Split normalized text into FINDINGS and IMPRESSION sections."""
    if not raw_text or not isinstance(raw_text, str):
        return {
            "findings": "No focal abnormality is described.",
            "impression": "No acute cardiopulmonary abnormality.",
            "raw": "",
        }

    # Respect explicit FINDINGS: / IMPRESSION: headers when present
    m_imp = re.search(r"\bimpression\s*[:=-]\s*", raw_text, flags=re.IGNORECASE)
    if m_imp:
        raw_findings = raw_text[: m_imp.start()].strip()
        raw_impression = raw_text[m_imp.end() :].strip()
        findings_norm = format_clinical_report(raw_findings) if raw_findings else ""
        impression_norm = format_clinical_report(raw_impression) if raw_impression else ""
        if findings_norm or impression_norm:
            return {
                "findings": findings_norm or "No focal abnormality is described.",
                "impression": impression_norm or "No acute cardiopulmonary abnormality.",
                "raw": format_clinical_report(raw_text),
            }

    text = format_clinical_report(raw_text)
    sentences = [sentence.strip() for sentence in text.split(".") if sentence.strip()]

    findings = []
    impression = []
    for sentence in sentences:
        low = sentence.lower()
        if any(marker in low for marker in ("no acute", "conclusion", "impression")):
            impression.append(sentence)
        else:
            findings.append(sentence)

    if not impression:
        negative = [s for s in findings if s.lower().startswith("no ")]
        impression = negative[-1:] if negative else findings[-1:]
        if impression:
            findings = [s for s in findings if s not in impression]

    if not findings:
        findings = ["No focal abnormality is described."]
    if not impression:
        impression = ["No acute cardiopulmonary abnormality."]

    return {
        "findings": ". ".join(findings) + ".",
        "impression": ". ".join(impression) + ".",
        "raw": text,
    }
