from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

COL_MAP_2019_2021 = {
    "sno": "serial_number",
    "s. no.": "serial_number",
    "s.no": "serial_number",
    "rank": "rank",
    "air": "rank",
    "allotted quota": "quota",
    "allotted institute": "college_name",
    "institute name": "college_name",
    "course": "course",
    "subject name": "course",
    "allotted category": "allotted_category",
    "candidate category": "category",
    "remarks": "remarks",
}

COL_MAP_2022_2023 = {
    "sno": "serial_number",
    "s. no.": "serial_number",
    "s.no": "serial_number",
    "rank": "rank",
    "air": "rank",
    "allotted quota": "quota",
    "allotted institute": "college_name",
    "institute name": "college_name",
    "course": "course",
    "subject name": "course",
    "allotted category": "allotted_category",
    "candidate category": "category",
    "option no": "option_no",
    "option no.": "option_no",
    "remarks": "remarks",
}

COL_MAP_2024_2025 = {
    "sno": "serial_number",
    "s. no.": "serial_number",
    "s.no": "serial_number",
    "air": "rank",
    "rank": "rank",
    "allotted quota": "quota",
    "allotted institute": "college_name",
    "institute name": "college_name",
    "course": "course",
    "subject name": "course",
    "allotted category": "allotted_category",
    "candidate category": "category",
    "remarks": "remarks",
}

SEAT_MATRIX_COL_MAP = {
    "sr no": "serial_number",
    "sr. no.": "serial_number",
    "sr. no": "serial_number",
    "state": "state",
    "quotaname": "quota",
    "quota name": "quota",
    "college name": "college_name",
    "subject": "course",
    "category": "category",
    "pwd": "pwd",
    "totalseats": "total_seats",
    "total seats": "total_seats",
    "total seats ": "total_seats",
}


def _normalize_col_name(col: str) -> str:
    col = col.strip().lower()
    col = re.sub(r"\s+", " ", col)
    return col


def _get_col_map(year: int) -> dict[str, str]:
    if 2019 <= year <= 2021:
        return COL_MAP_2019_2021
    elif 2022 <= year <= 2023:
        return COL_MAP_2022_2023
    elif 2024 <= year <= 2025:
        return COL_MAP_2024_2025
    else:
        logger.warning("Year %d not in known ranges, using 2024-2025 map", year)
        return COL_MAP_2024_2025


def _parse_int(value: str) -> Optional[int]:
    if not value:
        return None
    cleaned = re.sub(r"[^\d]", "", str(value).strip())
    if cleaned:
        try:
            return int(cleaned)
        except ValueError:
            return None
    return None


class MCCParser:
    """Parser for MCC (Medical Counselling Committee) allotment and seat matrix PDFs."""

    def parse_allotment_table(
        self, headers: list[str], rows: list[list], year: int
    ) -> list[dict]:
        """Parse MCC allotment table into normalized records."""
        col_map = _get_col_map(year)
        normalized_headers = self._map_headers(headers, col_map)
        records = []

        for row_idx, row in enumerate(rows):
            try:
                record = self._extract_allotment_record(
                    normalized_headers, row, year, row_idx
                )
                if record:
                    records.append(record)
            except Exception as e:
                logger.warning(
                    "Failed to parse row %d in year %d: %s", row_idx, year, str(e)
                )

        logger.info(
            "Parsed %d allotment records from %d rows (year=%d)",
            len(records),
            len(rows),
            year,
        )
        return records

    def parse_seat_matrix_table(
        self, headers: list[str], rows: list[list], year: int
    ) -> list[dict]:
        """Parse seat matrix table into records."""
        normalized_headers = self._map_headers(headers, SEAT_MATRIX_COL_MAP)
        records = []

        for row_idx, row in enumerate(rows):
            try:
                record = self._extract_seat_record(
                    normalized_headers, row, year, row_idx
                )
                if record:
                    records.append(record)
            except Exception as e:
                logger.warning(
                    "Failed to parse seat row %d in year %d: %s",
                    row_idx,
                    year,
                    str(e),
                )

        logger.info(
            "Parsed %d seat matrix records from %d rows (year=%d)",
            len(records),
            len(rows),
            year,
        )
        return records

    def _map_headers(
        self, headers: list[str], col_map: dict[str, str]
    ) -> list[str]:
        """Map raw column headers to normalized names."""
        result = []
        for h in headers:
            normalized = _normalize_col_name(h)
            mapped = col_map.get(normalized, normalized)
            result.append(mapped)
        return result

    def _extract_allotment_record(
        self,
        headers: list[str],
        row: list,
        year: int,
        row_idx: int,
    ) -> Optional[dict]:
        """Extract a single allotment record from a row."""
        data = dict(zip(headers, row))

        rank = self._resolve_int_field(data, ["rank", "air"])
        if rank is None:
            return None

        college_name = self._resolve_str_field(
            data, ["college_name", "institute_name"]
        )
        if not college_name:
            return None

        quota = self._resolve_str_field(data, ["quota", "allotted_quota"])
        if not quota:
            return None

        category = self._resolve_str_field(data, ["category", "candidate_category"])
        if not category:
            return None

        course = self._resolve_str_field(data, ["course", "subject_name"]) or "MBBS"

        return {
            "college_name": college_name.strip(),
            "state": self._infer_state(college_name),
            "year": year,
            "round": 0,
            "quota": self._normalize_quota(quota),
            "category": self._normalize_category(category),
            "course": course.strip(),
            "opening_rank": rank,
            "closing_rank": rank,
            "seat_type": "government",
            "seats": None,
            "source_file": f"mcc_allotment_{year}",
            "confidence": 1.0,
        }

    def _extract_seat_record(
        self,
        headers: list[str],
        row: list,
        year: int,
        row_idx: int,
    ) -> Optional[dict]:
        """Extract a single seat matrix record from a row."""
        data = dict(zip(headers, row))

        college_name = self._resolve_str_field(data, ["college_name"])
        if not college_name:
            return None

        total_seats = self._resolve_int_field(data, ["total_seats", "totalseats"])
        if total_seats is None:
            return None

        category = self._resolve_str_field(data, ["category"]) or "General"
        quota = self._resolve_str_field(data, ["quota", "quotaname"]) or "State Quota"
        course = self._resolve_str_field(data, ["course", "subject"]) or "MBBS"
        state = self._resolve_str_field(data, ["state"]) or self._infer_state(
            college_name
        )

        return {
            "college_name": college_name.strip(),
            "state": state,
            "year": year,
            "round": 0,
            "quota": self._normalize_quota(quota),
            "category": self._normalize_category(category),
            "course": course.strip(),
            "opening_rank": None,
            "closing_rank": None,
            "seat_type": "government",
            "seats": total_seats,
            "source_file": f"mcc_seat_matrix_{year}",
            "confidence": 1.0,
        }

    def _resolve_str_field(self, data: dict, field_names: list[str]) -> Optional[str]:
        for name in field_names:
            val = data.get(name)
            if val and str(val).strip():
                return str(val).strip()
        return None

    def _resolve_int_field(self, data: dict, field_names: list[str]) -> Optional[int]:
        for name in field_names:
            val = data.get(name)
            if val is not None and str(val).strip():
                result = _parse_int(str(val))
                if result is not None:
                    return result
        return None

    def _normalize_quota(self, quota: str) -> str:
        quota_lower = quota.lower().strip()
        mapping = {
            "all india": "All India",
            "amu": "AMU Quota",
            "delhi": "Delhi University",
            "esi": "ESI",
            "ip": "IP University",
            "deemed": "Deemed/Paid Seats",
            "nri": "NRI",
            "open": "Open Seat",
            "state": "State Quota",
        }
        for key, val in mapping.items():
            if key in quota_lower:
                return val
        return quota.strip()

    def _normalize_category(self, category: str) -> str:
        cat = category.strip()
        mapping = {
            "general": "General",
            "obc": "OBC-NCL",
            "obc-ncl": "OBC-NCL",
            "sc": "SC",
            "st": "ST",
            "ews": "EWS",
            "general pwd": "General PwD",
            "obc-ncl pwd": "OBC-NCL PwD",
            "obc pwd": "OBC-NCL PwD",
            "sc pwd": "SC PwD",
            "st pwd": "ST PwD",
            "ews pwd": "EWS PwD",
        }
        cat_lower = cat.lower()
        for key, val in mapping.items():
            if key == cat_lower:
                return val
        return cat

    def _infer_state(self, college_name: str) -> str:
        name_lower = college_name.lower()
        state_keywords = {
            "Delhi": ["delhi", "new delhi"],
            "Maharashtra": ["mumbai", "pune", "maharashtra", "nagpur"],
            "Karnataka": ["bangalore", "bengaluru", "karnataka", "mysore"],
            "Tamil Nadu": ["chennai", "tamil nadu", "coimbatore"],
            "Uttar Pradesh": ["lucknow", "uttar pradesh", "kanpur", "noida"],
            "West Bengal": ["kolkata", "west bengal", "calcutta"],
            "Gujarat": ["ahmedabad", "gujarat", "gandhinagar"],
            "Rajasthan": ["jaipur", "rajasthan", "jodhpur"],
            "Telangana": ["hyderabad", "telangana"],
            "Andhra Pradesh": ["andhra pradesh", "vijayawada", "visakhapatnam"],
            "Kerala": ["kerala", "kochi", "trivandrum"],
            "Madhya Pradesh": ["bhopal", "madhya pradesh", "indore"],
            "Bihar": ["patna", "bihar"],
            "Punjab": ["chandigarh", "punjab", "ludhiana"],
            "Haryana": ["haryana", "gurgaon", "faridabad"],
            "Odisha": ["bhubaneswar", "odisha", "orissa"],
            "Assam": ["guwahati", "assam"],
            "Jharkhand": ["ranchi", "jharkhand"],
            "Chhattisgarh": ["raipur", "chhattisgarh"],
            "Uttarakhand": ["dehradun", "uttarakhand"],
            "Himachal Pradesh": ["shimla", "himachal"],
            "Goa": ["goa"],
            "Jammu and Kashmir": ["jammu", "kashmir", "srinagar"],
            "Puducherry": ["puducherry", "pondicherry"],
        }
        for state, keywords in state_keywords.items():
            for kw in keywords:
                if kw in name_lower:
                    return state
        return "Unknown"
