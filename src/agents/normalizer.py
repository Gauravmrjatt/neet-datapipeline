"""Data Normalizer Agent - normalizes extracted data into consistent formats."""

import asyncio
import csv
import json
import re
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

MAX_WORKERS = 4

STATE_ABBREVIATIONS: dict[str, str] = {
    "U.P.": "Uttar Pradesh", "UP": "Uttar Pradesh",
    "M.P.": "Madhya Pradesh", "MP": "Madhya Pradesh",
    "T.N.": "Tamil Nadu", "TN": "Tamil Nadu",
    "A.P.": "Andhra Pradesh", "AP": "Andhra Pradesh",
    "W.B.": "West Bengal", "WB": "West Bengal",
    "N.C.T.": "Delhi", "NCT": "Delhi",
    "J&K": "Jammu & Kashmir", "HP": "Himachal Pradesh",
    "H.P.": "Himachal Pradesh", "RJ": "Rajasthan",
    "GJ": "Gujarat", "PB": "Punjab", "HR": "Haryana",
    "KA": "Karnataka", "KL": "Kerala", "TS": "Telangana",
    "MH": "Maharashtra", "OR": "Odisha", "CG": "Chhattisgarh",
}

STATE_FULL_NAMES: dict[str, str] = {
    "uttar pradesh": "Uttar Pradesh", "madhya pradesh": "Madhya Pradesh",
    "tamil nadu": "Tamil Nadu", "andhra pradesh": "Andhra Pradesh",
    "west bengal": "West Bengal", "himachal pradesh": "Himachal Pradesh",
    "jammu & kashmir": "Jammu & Kashmir", "delhi": "Delhi",
    "maharashtra": "Maharashtra", "rajasthan": "Rajasthan",
    "bihar": "Bihar", "karnataka": "Karnataka", "kerala": "Kerala",
    "telangana": "Telangana", "gujarat": "Gujarat", "punjab": "Punjab",
    "haryana": "Haryana", "odisha": "Odisha", "goa": "Goa",
    "chhattisgarh": "Chhattisgarh", "jharkhand": "Jharkhand",
    "uttarakhand": "Uttarakhand", "assam": "Assam",
    "andaman and nicobar islands": "Andaman and Nicobar Islands",
    "puducherry": "Puducherry", "chandigarh": "Chandigarh",
    "all india": "All India",
}

QUOTA_CODES: dict[str, str] = {
    "AI": "All India", "AM": "AMU Quota", "DU": "Delhi University",
    "ES": "ESI", "IP": "IP University", "PS": "Deemed/Paid Seats",
    "NR": "NRI", "SO": "Open Seat",
    "all india": "All India", "deemed/paid seats quota": "Deemed/Paid Seats",
    "deemed/paid seats": "Deemed/Paid Seats", "delhi ncr": "Delhi NCR",
}

CATEGORY_CODES: dict[str, str] = {
    "GN": "General", "GEN": "General", "General": "General", "Open": "General",
    "BC": "OBC-NCL", "OBC": "OBC-NCL", "OBC-NCL": "OBC-NCL",
    "SC": "Scheduled Caste", "ST": "Scheduled Tribe",
    "EW": "EWS", "EWS": "EWS",
    "GN PwD": "General PwD", "GEN PwD": "General PwD",
    "OBC PwD": "OBC-NCL PwD", "BC PwD": "OBC-NCL PwD",
    "SC PwD": "Scheduled Caste PwD", "ST PwD": "Scheduled Tribe PwD",
    "EWS PwD": "EWS PwD", "EW PwD": "EWS PwD",
    "UR": "General", "Unreserved": "General",
}

COLLEGE_ALIASES: dict[str, str] = {
    "KGMU": "King George Medical University",
    "KGMC": "King George Medical University",
    "MAMC": "Maulana Azad Medical College",
    "VMMC": "Vardhman Mahavir Medical College",
    "UCMS": "University College of Medical Sciences",
    "JIPMER": "Jawaharlal Institute of Postgraduate Medical Education and Research",
    "AIIMS": "All India Institute of Medical Sciences",
    "LHMC": "Lady Hardinge Medical College",
    "BJMC": "Byramjee Jeejeebhoy Government Medical College",
    "GSVM": "GSVM Medical College",
    "LLRM": "LLRM Medical College",
}


def _normalize_state(raw_state: str) -> str:
    if not raw_state or not raw_state.strip():
        return "Unknown"
    cleaned = raw_state.strip()
    if cleaned in STATE_ABBREVIATIONS:
        return STATE_ABBREVIATIONS[cleaned]
    lower = cleaned.lower()
    if lower in STATE_FULL_NAMES:
        return STATE_FULL_NAMES[lower]
    for abbr, full in STATE_ABBREVIATIONS.items():
        if abbr.lower() in lower:
            return full
    return cleaned.title()


def _normalize_quota(raw_quota: str) -> str:
    if not raw_quota or not raw_quota.strip():
        return "General"
    cleaned = raw_quota.strip()
    lower = cleaned.lower()
    if lower in QUOTA_CODES:
        return QUOTA_CODES[lower]
    upper = cleaned.upper()
    if upper in QUOTA_CODES:
        return QUOTA_CODES[upper]
    for code, full in QUOTA_CODES.items():
        if code.lower() in lower:
            return full
    return cleaned


def _normalize_category(raw_category: str) -> str:
    if not raw_category or not raw_category.strip():
        return "General"
    cleaned = raw_category.strip()
    if cleaned in CATEGORY_CODES:
        return CATEGORY_CODES[cleaned]
    lower = cleaned.lower()
    for code, full in CATEGORY_CODES.items():
        if code.lower() == lower or full.lower() == lower:
            return full
    if "obc" in lower or "bc" in lower:
        return "OBC-NCL"
    if "sc" in lower:
        return "Scheduled Caste"
    if "st" in lower:
        return "Scheduled Tribe"
    if "ews" in lower or "ew" in lower:
        return "EWS"
    if "pwd" in lower:
        base = "General"
        if "obc" in lower or "bc" in lower:
            base = "OBC-NCL"
        elif "sc" in lower:
            base = "Scheduled Caste"
        elif "st" in lower:
            base = "Scheduled Tribe"
        elif "ews" in lower:
            base = "EWS"
        return f"{base} PwD"
    return cleaned


def _normalize_college_name(raw_name: str) -> str:
    if not raw_name or not raw_name.strip():
        return "Unknown"
    cleaned = raw_name.strip()
    if cleaned in COLLEGE_ALIASES:
        return COLLEGE_ALIASES[cleaned]
    upper = cleaned.upper().replace(".", "").replace(" ", "")
    for alias, full in COLLEGE_ALIASES.items():
        if alias.upper().replace(".", "").replace(" ", "") == upper:
            return full
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _clean_rank(raw_rank: str) -> str:
    if not raw_rank or not raw_rank.strip():
        return ""
    cleaned = raw_rank.strip().replace(",", "").replace(" ", "")
    cleaned = re.sub(r"[^\d]", "", cleaned)
    if cleaned:
        try:
            rank_int = int(cleaned)
            if 0 < rank_int <= 2_000_000:
                return str(rank_int)
        except ValueError:
            pass
    return ""


def _extract_state_from_institute(institute_text: str) -> str:
    """Try to extract state from institute address text."""
    if not institute_text:
        return ""
    text = str(institute_text)
    for state_name in STATE_FULL_NAMES.values():
        if state_name.lower() in text.lower():
            return state_name
    for abbr, full in STATE_ABBREVIATIONS.items():
        if abbr.lower() in text.lower():
            return full
    return ""


def _is_title_row(headers: list[str]) -> bool:
    """Check if headers look like a title row rather than column headers."""
    if not headers:
        return True
    non_empty = [h for h in headers if h and h.strip()]
    if len(non_empty) <= 1 and len(non_empty[0]) > 30:
        return True
    if all(len(h) > 40 for h in non_empty):
        return True
    return False


def _detect_table_type(headers: list[str], rows: list[list]) -> str:
    """Detect if table is allotment, seat_matrix, or other."""
    header_text = " ".join(h.lower() for h in headers if h)
    if any(w in header_text for w in ["opening rank", "closing rank", "allotted", "quota"]):
        return "allotment"
    if any(w in header_text for w in ["seat matrix", "vacant", "vacancy", "total seats"]):
        return "seat_matrix"
    if any(w in header_text for w in ["roll", "admitted", "candidate"]):
        return "admitted"
    return "unknown"


def _normalize_allotment_row(values: list[str], source_file: str, metadata: dict) -> dict[str, Any] | None:
    """Normalize a row from an allotment table."""
    if len(values) < 3:
        return None

    row: dict[str, Any] = {}

    # Try to detect column mapping from header patterns
    # Common allotment formats:
    # [opening_rank, closing_rank, quota, institute, course, category]
    # [sno, rank, quota, institute, course, category, candidate_cat]
    # [rank, round1_quota, round1_institute, ..., round3_allotted_cat]

    opening_rank = ""
    closing_rank = ""
    quota = ""
    institute = ""
    course = ""
    category = ""

    for val in values:
        val_clean = str(val).strip()
        if not val_clean or val_clean == "-":
            continue

        # Check if it's a rank (pure number)
        rank = _clean_rank(val_clean)
        if rank:
            if not opening_rank:
                opening_rank = rank
            elif not closing_rank:
                closing_rank = rank
            continue

        # Check if it's a category
        cat_lower = val_clean.lower()
        if any(c.lower() == cat_lower or cat_lower in c.lower() for c in CATEGORY_CODES):
            category = _normalize_category(val_clean)
            continue

        # Check if it's a quota
        if any(q.lower() in cat_lower or cat_lower in q.lower() for q in QUOTA_CODES):
            quota = _normalize_quota(val_clean)
            continue

        # Check if it's a course
        if any(c in cat_lower for c in ["mbbs", "bds", "b.sc", "nursing", "bams", "bhms"]):
            course = val_clean
            continue

        # Check if it looks like an institute (long text with address)
        if len(val_clean) > 20 and ("," in val_clean or "college" in cat_lower or "medical" in cat_lower or "hospital" in cat_lower):
            institute = val_clean
            continue

    if not opening_rank and not closing_rank:
        return None

    state = _extract_state_from_institute(institute)

    return {
        "college_name": _normalize_college_name(institute.split(",")[0] if institute else ""),
        "state": state or "Unknown",
        "quota": quota or "All India",
        "category": category or "General",
        "rank": opening_rank or closing_rank or "",
        "opening_rank": opening_rank,
        "closing_rank": closing_rank,
        "course": course or "MBBS",
        "year": metadata.get("year", 0),
        "round": metadata.get("round", 0),
        "round_number": metadata.get("round", 0),
        "source_file": source_file,
    }


def _normalize_seat_matrix_row(values: list[str], source_file: str, metadata: dict) -> dict[str, Any] | None:
    """Normalize a row from a seat matrix table."""
    if len(values) < 3:
        return None

    row: dict[str, Any] = {}
    college = ""
    state = ""
    quota = ""
    course = ""
    category = ""
    seats = ""

    for val in values:
        val_clean = str(val).strip()
        if not val_clean or val_clean == "-":
            continue

        # Check if it's a seat count
        try:
            num = int(val_clean.replace(",", ""))
            if 0 < num < 10000:
                seats = str(num)
                continue
        except ValueError:
            pass

        # Check category
        cat_lower = val_clean.lower()
        if any(c.lower() == cat_lower for c in CATEGORY_CODES):
            category = _normalize_category(val_clean)
            continue

        # Check quota
        if any(q.lower() in cat_lower for q in QUOTA_CODES):
            quota = _normalize_quota(val_clean)
            continue

        # Check course
        if any(c in cat_lower for c in ["mbbs", "bds", "b.sc", "nursing"]):
            course = val_clean
            continue

        # Check state
        state_name = _normalize_state(val_clean)
        if state_name != val_clean.title():
            state = state_name
            continue

        # Long text = institute name
        if len(val_clean) > 15 and not val_clean.replace(" ", "").isdigit():
            college = val_clean
            continue

    if not college and not seats:
        return None

    return {
        "college_name": _normalize_college_name(college.split(",")[0] if college else ""),
        "state": state or _extract_state_from_institute(college) or "Unknown",
        "quota": quota or "All India",
        "category": category or "General",
        "rank": "",
        "opening_rank": "",
        "closing_rank": "",
        "course": course or "MBBS",
        "seat_count": seats,
        "year": metadata.get("year", 0),
        "round": metadata.get("round", 0),
        "round_number": metadata.get("round", 0),
        "source_file": source_file,
    }


def _extract_tables_from_json(json_path: Path) -> list[dict[str, Any]]:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning("json_load_failed", file=str(json_path), error=str(e))
        return []

    rows: list[dict[str, Any]] = []
    metadata = data.get("metadata", {})
    source_file = data.get("source_file", json_path.stem)

    # Infer year and round from filename
    year_match = re.search(r"(20[12]\d)", source_file)
    if year_match and not metadata.get("year"):
        metadata["year"] = int(year_match.group(1))

    round_match = re.search(r"round\s*(\d)", source_file.lower())
    if round_match and not metadata.get("round"):
        metadata["round"] = int(round_match.group(1))
    elif "stray" in source_file.lower() and not metadata.get("round"):
        metadata["round"] = 4
    elif "mop" in source_file.lower() and not metadata.get("round"):
        metadata["round"] = 3

    for table in data.get("tables", []):
        headers = table.get("headers", [])
        table_rows = table.get("rows", [])

        if _is_title_row(headers):
            # Use first data row as headers if available
            if table_rows:
                headers = [str(c).strip() for c in table_rows[0]]
                table_rows = table_rows[1:]
            else:
                continue

        if len(table_rows) < 1:
            continue

        table_type = _detect_table_type(headers, table_rows)

        for row_data in table_rows:
            if isinstance(row_data, list):
                values = [str(c).strip() if c is not None else "" for c in row_data]
            elif isinstance(row_data, dict):
                values = list(row_data.values())
            else:
                continue

            if table_type == "allotment":
                normalized = _normalize_allotment_row(values, source_file, metadata)
            elif table_type == "seat_matrix":
                normalized = _normalize_seat_matrix_row(values, source_file, metadata)
            else:
                normalized = _normalize_allotment_row(values, source_file, metadata)

            if normalized:
                rows.append(normalized)

    return rows


class DataNormalizer:
    """Normalizes extracted JSON data into a consistent, deduplicated dataset."""

    CLEANED_DIR = "cleaned"
    NORMALIZED_FILE = "normalized_dataset.csv"
    LOG_FILE = "normalization_log.csv"

    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir or Path("data")
        self.cleaned_dir = self.output_dir / self.CLEANED_DIR
        self.cleaned_dir.mkdir(parents=True, exist_ok=True)
        self.normalized_path = self.cleaned_dir / self.NORMALIZED_FILE
        self.log_path = self.output_dir / self.LOG_FILE
        self.semaphore = asyncio.Semaphore(MAX_WORKERS)

    async def _normalize_file(self, json_path: Path) -> tuple[list[dict[str, Any]], int]:
        async with self.semaphore:
            return await asyncio.to_thread(self._normalize_file_sync, json_path)

    def _normalize_file_sync(self, json_path: Path) -> tuple[list[dict[str, Any]], int]:
        rows = _extract_tables_from_json(json_path)
        return rows, len(rows)

    def _deduplicate(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        unique_rows: list[dict[str, Any]] = []
        for row in rows:
            key_parts = [
                str(row.get("college_name", "")).lower().strip(),
                str(row.get("year", "")).strip(),
                str(row.get("round", row.get("round_number", ""))).strip(),
                str(row.get("quota", "")).lower().strip(),
                str(row.get("category", "")).lower().strip(),
                str(row.get("rank", row.get("closing_rank", ""))).strip(),
            ]
            composite_key = "|".join(key_parts)
            if composite_key not in seen:
                seen.add(composite_key)
                unique_rows.append(row)
        return unique_rows

    async def normalize(self, extracted_dir: Path) -> Path:
        """Normalize all extracted JSON files and produce a cleaned dataset."""
        json_files = [f for f in extracted_dir.glob("*.json") if not f.name.startswith("_")]
        logger.info("normalization_started", total_files=len(json_files))

        tasks = [self._normalize_file(jf) for jf in json_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_rows: list[dict[str, Any]] = []
        total_raw = 0
        for result in results:
            if isinstance(result, Exception):
                logger.error("normalization_task_error", error=str(result))
                continue
            if isinstance(result, tuple) and len(result) == 2:
                rows, count = result
                all_rows.extend(rows)
                total_raw += count

        logger.info("normalization_raw", total_rows=total_raw)

        deduped = self._deduplicate(all_rows)
        logger.info(
            "deduplication_complete",
            before=len(all_rows),
            after=len(deduped),
            removed=len(all_rows) - len(deduped),
        )

        output_fields = [
            "source_file", "college_name", "state", "quota", "category",
            "rank", "opening_rank", "closing_rank", "course",
            "year", "round", "round_number", "seat_count",
        ]

        with open(self.normalized_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=output_fields, extrasaction="ignore")
            writer.writeheader()
            for row in deduped:
                clean_row = {k: row.get(k, "") for k in output_fields}
                writer.writerow(clean_row)

        logger.info(
            "normalization_complete",
            output_path=str(self.normalized_path),
            total_rows=len(deduped),
        )
        return self.normalized_path
