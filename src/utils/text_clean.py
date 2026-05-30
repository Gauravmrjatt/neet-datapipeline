from __future__ import annotations

import re
import unicodedata
from typing import Optional

STATE_ABBREVIATIONS: dict[str, str] = {
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CG": "Chhattisgarh",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HR": "Haryana",
    "HP": "Himachal Pradesh",
    "JH": "Jharkhand",
    "KA": "Karnataka",
    "KL": "Kerala",
    "MP": "Madhya Pradesh",
    "MH": "Maharashtra",
    "MN": "Manipur",
    "ML": "Meghalaya",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OD": "Odisha",
    "OR": "Odisha",
    "PB": "Punjab",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TN": "Tamil Nadu",
    "TS": "Telangana",
    "TR": "Tripura",
    "UP": "Uttar Pradesh",
    "UK": "Uttarakhand",
    "UN": "Uttarakhand",
    "WB": "West Bengal",
    "AN": "Andaman and Nicobar Islands",
    "CH": "Chandigarh",
    "DD": "Dadra and Nagar Haveli and Daman and Diu",
    "DL": "Delhi",
    "JK": "Jammu and Kashmir",
    "LA": "Ladakh",
    "LD": "Lakshadweep",
    "PY": "Puducherry",
    "PU": "Puducherry",
    "TG": "Telangana",
}

QUOTA_MAPPINGS: dict[str, str] = {
    "AIQ": "All India Quota",
    "ALL INDIA": "All India Quota",
    "ALLINDIA": "All India Quota",
    "SQ": "State Quota",
    "STATE": "State Quota",
    "MQ": "Management Quota",
    "MGMT": "Management Quota",
    "MANAGEMENT": "Management Quota",
    "NRI": "NRI Quota",
    "FN": "Foreign National",
    "OCI": "OCI",
    "EWS": "Economically Weaker Section",
    "SC": "Scheduled Caste",
    "ST": "Scheduled Tribe",
    "OBC": "Other Backward Class",
    "GEN": "General",
    "UR": "Unreserved",
    "PH": "Physically Handicapped",
    "PwD": "Persons with Disability",
    "PWD": "Persons with Disability",
}

CATEGORY_MAPPINGS: dict[str, str] = {
    "GEN": "General",
    "UR": "Unreserved",
    "GN": "General",
    "OPEN": "Open",
    "SC": "Scheduled Caste",
    "ST": "Scheduled Tribe",
    "OBC": "Other Backward Class",
    "OBC-NCL": "Other Backward Class",
    "EWS": "Economically Weaker Section",
    "EWS-UR": "Economically Weaker Section",
    "PH": "Physically Handicapped",
    "PwD": "Persons with Disability",
    "PWD": "Persons with Disability",
    "ORS": "Open Round Seat",
    "JP": "Jurisdiction Person",
}

_CLEAN_TEXT_PATTERN = re.compile(r"[^\w\s\-,./()&+]")
_MULTI_SPACE_PATTERN = re.compile(r"\s+")
_RANK_PATTERN = re.compile(r"[^\d]")


def normalize_college_name(name: str) -> str:
    if not name:
        return ""
    name = unicodedata.normalize("NFKC", name)
    name = name.strip()
    name = _MULTI_SPACE_PATTERN.sub(" ", name)
    name = name.title()
    suffixes = [" College", " Institute", " University", " Hospital", " Medical"]
    for suffix in suffixes:
        if name.endswith(suffix.lower()):
            name = name[: -len(suffix.lower())] + suffix
    return name.strip()


def normalize_state(name: str) -> str:
    if not name:
        return ""
    name = name.strip()
    upper = name.upper().strip()
    if upper in STATE_ABBREVIATIONS:
        return STATE_ABBREVIATIONS[upper]
    name = unicodedata.normalize("NFKC", name)
    name = _MULTI_SPACE_PATTERN.sub(" ", name).strip()
    return name.title()


def normalize_quota(code: str) -> str:
    if not code:
        return ""
    code = code.strip()
    upper = code.upper().strip()
    if upper in QUOTA_MAPPINGS:
        return QUOTA_MAPPINGS[upper]
    code = unicodedata.normalize("NFKC", code)
    code = _MULTI_SPACE_PATTERN.sub(" ", code).strip()
    return code.title()


def normalize_category(code: str) -> str:
    if not code:
        return ""
    code = code.strip()
    upper = code.upper().strip()
    if upper in CATEGORY_MAPPINGS:
        return CATEGORY_MAPPINGS[upper]
    code = unicodedata.normalize("NFKC", code)
    code = _MULTI_SPACE_PATTERN.sub(" ", code).strip()
    return code.title()


def clean_rank(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    s = str(value).strip()
    if not s or s.lower() in ("", "na", "n/a", "-", "--", "nil", "null"):
        return None
    s = _RANK_PATTERN.sub("", s)
    if not s:
        return None
    try:
        rank = int(s)
        if rank < 0:
            return None
        return rank
    except (ValueError, OverflowError):
        return None


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.strip()
    text = _MULTI_SPACE_PATTERN.sub(" ", text)
    text = _CLEAN_TEXT_PATTERN.sub("", text)
    return text.strip()
