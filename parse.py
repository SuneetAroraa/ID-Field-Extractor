import re
from typing import Optional, List
from pydantic import BaseModel, Field
from trace_logging import log
from datetime import datetime

class OCR_Output(BaseModel):
    name: Optional[str] = None
    id_number: Optional[str] = None
    date_of_birth: Optional[str] = None
    date_of_issue: Optional[str] = None
    expiry_date: Optional[str] = None
    confidence_scores: dict[str, float] = Field(default_factory=dict)
    raw_text: str = ""

def parse_date_str(text):
    try:
        return datetime.strptime(text, "%d-%m-%Y")
    except ValueError:
        return None

def extract_date_from_row(text):
    m = re.search(r"\d{2}-\d{2}-\d{4}", text)
    if m:
        return m.group()

    m2 = re.search(r"(\d{2})-(\d{2})-?\D*(\d{4})", text)
    if m2:
        return f"{m2.group(1)}-{m2.group(2)}-{m2.group(3)}"

    return None

def group_into_rows(results, y_tolerance=15):
    items = sorted(results, key=lambda r: r[0][0][1])

    rows = []
    current_row = [items[0]]
    current_y = items[0][0][0][1]

    for item in items[1:]:
        y = item[0][0][1]
        if abs(y - current_y) <= y_tolerance:
            current_row.append(item)
            current_y = y
        else:
            rows.append(current_row)
            current_row = [item]
            current_y = y
    rows.append(current_row)

    merged = []
    for row in rows:
        row.sort(key=lambda item: item[0][0][0])
        merged_text = " ".join(t for (_, t, _) in row)
        row_conf = max(c for (_, _, c) in row)
        top_y = row[0][0][0][1]
        merged.append(([[0, top_y]], merged_text, row_conf))
    return merged


def extract(results, image_height, trace_path="trace.jsonl"):
    id_pattern = r"[A-Z][A-Z0-9]{7,10}"
    name_pattern = r"^[A-Z][a-z]+$"

    merged_results = group_into_rows(results)

    dates_found = []
    id_number = None
    id_conf = None
    names_found = []
    raw_text_parts = []

    fields = {
        "name": None,
        "id_number": None,
        "date_of_birth": None,
        "date_of_issue": None,
        "expiry_date": None,
        "confidence_scores": {},
        "raw_text": ""
    }

    for (bbox, text, conf) in merged_results:
        text = text.strip()
        raw_text_parts.append(text)
        if conf < 0.4:
            continue

        top_y = bbox[0][1]

        date_str = extract_date_from_row(text)
        if date_str:
            dates_found.append((date_str, conf))
        elif re.fullmatch(id_pattern, text):
            id_number = text
            id_conf = conf
        elif top_y <= image_height * 0.5 and re.fullmatch(name_pattern, text):
            names_found.append((text, conf))

    fields["raw_text"] = " ".join(raw_text_parts)

    if id_number:
        fields["id_number"] = id_number
        fields["confidence_scores"]["id_number"] = id_conf

    parsed_dates = []
    for text, conf in dates_found:
        dt = parse_date_str(text)
        if dt is not None:
            parsed_dates.append((text, conf, dt))
    parsed_dates.sort(key=lambda x: x[2])

    if len(parsed_dates) >= 1:
        fields["date_of_birth"] = parsed_dates[0][0]
        fields["confidence_scores"]["date_of_birth"] = parsed_dates[0][1]

    if len(parsed_dates) >= 2:
        fields["expiry_date"] = parsed_dates[-1][0]
        fields["confidence_scores"]["expiry_date"] = parsed_dates[-1][1]

    if len(parsed_dates) >= 3:
        fields["date_of_issue"] = parsed_dates[1][0]
        fields["confidence_scores"]["date_of_issue"] = parsed_dates[1][1]

    if names_found:
        fields["name"] = " ".join(n for n, c in names_found)
        avg_conf = sum(c for n, c in names_found) / len(names_found)
        fields["confidence_scores"]["name"] = avg_conf

    log(trace_path, "parse", {
        "fields_found": [k for k, v in fields.items() if k not in ("confidence_scores", "raw_text") and v is not None],
        "confidence_scores": fields["confidence_scores"]
    })

    return OCR_Output(**fields)