import re
from typing import Optional, List
from pydantic import BaseModel

def extract(results):
    date_pattern = r"\d{2}-\d{2}-\d{4}"
    id_pattern = r"[A-Z]\d{9}"
    name_pattern = r"^[A-Z][a-z]+$"

    dates_found = []
    id_number = None
    id_conf = None
    names_found = []
    raw_text_parts = []

    fields = {
        "name": None,
        "id_number": None,
        "date_of_birth": None,
        "expiry_date": None,
        "confidence_scores": {},
        "raw_text": ""
    }

    for i, (bbox, text, conf) in enumerate(results):
        text = text.strip()
        raw_text_parts.append(text)

        if conf > 0.4:
            continue

        if re.fullmatch(date_pattern,text):
            dates_found.append((text,conf))
        elif re.fullmatch(id_pattern,text):
            id_number = text
            id_conf = conf
        elif re.fullmatch(name_pattern,text):
            names_found.append((text,conf)) 

        fields["raw_text"] = " ".join(raw_text_parts)     

        if id_number:
            fields["id_number"] = id_number
            fields["confidence_scores"]["id_number"] = id_conf
        
        if len(dates_found) >= 1:
            dates_found.sort(key=lambda x: x[0])
            fields["date_of_birth"] = dates_found[0][0]
            fields["confidence_scores"]["date_of_birth"] = dates_found[0][1]
            if len(dates_found) >= 2:
                fields["expiry_date"] = dates_found[-1][0]
                fields["confidence_scores"]["expiry_date"] = dates_found[-1][1]

        if names_found:
            fields["name"] = " ".join(n for n, c in names_found)
            avg_conf = sum(c for n, c in names_found) / len(names_found)
            fields["confidence_scores"]["name"] = avg_conf

    return fields
