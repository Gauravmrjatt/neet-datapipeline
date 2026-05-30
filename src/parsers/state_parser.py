from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

RANK_KEYWORDS = {"rank", "air", "all india rank", "merit no", "merit number", "position"}
COLLEGE_KEYWORDS = {
    "college",
    "institute",
    "institution",
    "medical college",
    "hospital",
}
CATEGORY_KEYWORDS = {
    "category",
    "cat",
    "class",
    "reservation",
}
QUOTA_KEYWORDS = {"quota", "seat type", "seat_type"}
COURSE_KEYWORDS = {"course", "subject", "program", "branch"}


def _normalize_col(col: str) -> str:
    return re.sub(r"\s+", " ", col.strip().lower())


def _is_rank_col(col: str) -> bool:
    normalized = _normalize_col(col)
    return any(kw in normalized for kw in RANK_KEYWORDS)


def _is_college_col(col: str) -> bool:
    normalized = _normalize_col(col)
    return any(kw in normalized for kw in COLLEGE_KEYWORDS)


def _is_category_col(col: str) -> bool:
    normalized = _normalize_col(col)
    return any(kw == normalized or normalized.startswith(kw) for kw in CATEGORY_KEYWORDS)


def _is_quota_col(col: str) -> bool:
    normalized = _normalize_col(col)
    return any(kw in normalized for kw in QUOTA_KEYWORDS)


def _is_course_col(col: str) -> bool:
    normalized = _normalize_col(col)
    return any(kw in normalized for kw in COURSE_KEYWORDS)


def _detect_column_roles(headers: list[str]) -> dict[str, Optional[int]]:
    """Auto-detect column roles from header names."""
    roles: dict[str, Optional[int]] = {
        "rank": None,
        "college_name": None,
        "category": None,
        "quota": None,
        "course": None,
    }

    for i, h in enumerate(headers):
        if roles["rank"] is None and _is_rank_col(h):
            roles["rank"] = i
        elif roles["college_name"] is None and _is_college_col(h):
            roles["college_name"] = i
        elif roles["category"] is None and _is_category_col(h):
            roles["category"] = i
        elif roles["quota"] is None and _is_quota_col(h):
            roles["quota"] = i
        elif roles["course"] is None and _is_course_col(h):
            roles["course"] = i

    return roles


def _parse_int(value: str) -> Optional[int]:
    if not value:
        return None
    cleaned = re.sub(r"[^\d]", "", str(value).strip())
    try:
        return int(cleaned) if cleaned else None
    except ValueError:
        return None


STATE_COLLEGE_HINTS: dict[str, list[str]] = {
    "Maharashtra": ["mumbai", "pune", "maharashtra", "nagpur", "thane"],
    "Karnataka": ["bangalore", "bengaluru", "karnataka", "mysore", "hubli"],
    "Tamil Nadu": ["chennai", "tamil nadu", "coimbatore", "madurai"],
    "Uttar Pradesh": ["lucknow", "uttar pradesh", "kanpur", "noida", "agra"],
    "West Bengal": ["kolkata", "west bengal", "calcutta", "durgapur"],
    "Gujarat": ["ahmedabad", "gujarat", "gandhinagar", "surat"],
    "Rajasthan": ["jaipur", "rajasthan", "jodhpur", "udaipur"],
    "Telangana": ["hyderabad", "telangana", "warangal"],
    "Andhra Pradesh": ["andhra pradesh", "vijayawada", "visakhapatnam", "tirupati"],
    "Kerala": ["kerala", "kochi", "trivandrum", "calicut", "thrissur"],
    "Madhya Pradesh": ["bhopal", "madhya pradesh", "indore", "gwalior"],
    "Bihar": ["patna", "bihar", "muzaffarpur"],
    "Punjab": ["chandigarh", "punjab", "ludhiana", "amritsar"],
    "Haryana": ["haryana", "gurgaon", "faridabad", "rohtak"],
    "Odisha": ["bhubaneswar", "odisha", "orissa", "cuttack"],
    "Assam": ["guwahati", "assam", "silchar"],
    "Jharkhand": ["ranchi", "jharkhand", "jamshedpur"],
    "Chhattisgarh": ["raipur", "chhattisgarh", "bilaspur"],
    "Uttarakhand": ["dehradun", "uttarakhand", "haldwani"],
    "Himachal Pradesh": ["shimla", "himachal", "hamirpur"],
    "Goa": ["goa", "panaji"],
    "Jammu and Kashmir": ["jammu", "kashmir", "srinagar"],
    "Puducherry": ["puducherry", "pondicherry"],
    "Delhi": ["delhi", "new delhi"],
}


class StateParser:
    """Generic parser for state-specific NEET counselling PDF tables with auto-detection."""

    def parse_state_table(
        self, headers: list[str], rows: list[list], state: str, year: int
    ) -> list[dict]:
        """Parse a state-specific table with auto-detected column mapping."""
        roles = _detect_column_roles(headers)
        logger.info(
            "State=%s Year=%d Detected roles: %s", state, year, roles
        )

        records = []
        for row_idx, row in enumerate(rows):
            try:
                record = self._extract_record(headers, row, roles, state, year, row_idx)
                if record:
                    records.append(record)
            except Exception as e:
                logger.warning(
                    "Failed to parse state row %d (state=%s): %s",
                    row_idx,
                    state,
                    str(e),
                )

        logger.info(
            "Parsed %d records from state=%s year=%d",
            len(records),
            state,
            year,
        )
        return records

    def _extract_record(
        self,
        headers: list[str],
        row: list,
        roles: dict[str, Optional[int]],
        state: str,
        year: int,
        row_idx: int,
    ) -> Optional[dict]:
        """Extract a single record using detected roles."""
        data = dict(zip(headers, row))

        rank = self._get_rank_value(data, roles, headers)
        college_name = self._get_college_value(data, roles, headers)
        category = self._get_category_value(data, roles, headers)
        quota = self._get_quota_value(data, roles, headers)
        course = self._get_course_value(data, roles, headers)

        if not college_name:
            return None

        inferred_state = state if state != "Unknown" else self._infer_state(college_name)

        return {
            "college_name": college_name.strip(),
            "state": inferred_state,
            "year": year,
            "round": 0,
            "quota": self._normalize_quota(quota or "State Quota"),
            "category": self._normalize_category(category or "General"),
            "course": (course or "MBBS").strip(),
            "opening_rank": rank,
            "closing_rank": rank,
            "seat_type": "government",
            "seats": None,
            "source_file": f"state_{state.lower().replace(' ', '_')}_{year}",
            "confidence": 0.8,
        }

    def _get_rank_value(
        self, data: dict, roles: dict[str, Optional[int]], headers: list[str]
    ) -> Optional[int]:
        if roles["rank"] is not None and roles["rank"] < len(headers):
            val = data.get(headers[roles["rank"]])
            return _parse_int(str(val)) if val else None

        for h in headers:
            val = data.get(h)
            if val and _is_rank_col(h):
                parsed = _parse_int(str(val))
                if parsed is not None:
                    return parsed
        return None

    def _get_college_value(
        self, data: dict, roles: dict[str, Optional[int]], headers: list[str]
    ) -> Optional[str]:
        if roles["college_name"] is not None and roles["college_name"] < len(headers):
            val = data.get(headers[roles["college_name"]])
            return str(val).strip() if val else None

        for h in headers:
            val = data.get(h)
            if val and _is_college_col(h):
                return str(val).strip()
        return None

    def _get_category_value(
        self, data: dict, roles: dict[str, Optional[int]], headers: list[str]
    ) -> Optional[str]:
        if roles["category"] is not None and roles["category"] < len(headers):
            val = data.get(headers[roles["category"]])
            return str(val).strip() if val else None
        return None

    def _get_quota_value(
        self, data: dict, roles: dict[str, Optional[int]], headers: list[str]
    ) -> Optional[str]:
        if roles["quota"] is not None and roles["quota"] < len(headers):
            val = data.get(headers[roles["quota"]])
            return str(val).strip() if val else None
        return None

    def _get_course_value(
        self, data: dict, roles: dict[str, Optional[int]], headers: list[str]
    ) -> Optional[str]:
        if roles["course"] is not None and roles["course"] < len(headers):
            val = data.get(headers[roles["course"]])
            return str(val).strip() if val else None
        return None

    def _normalize_quota(self, quota: str) -> str:
        q = quota.lower().strip()
        mapping = {
            "all india": "All India",
            "state": "State Quota",
            "management": "Deemed/Paid Seats",
            "nri": "NRI",
            "ews": "EWS",
            "open": "Open Seat",
        }
        for key, val in mapping.items():
            if key in q:
                return val
        return quota.strip()

    def _normalize_category(self, category: str) -> str:
        c = category.strip()
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
        c_lower = c.lower()
        for key, val in mapping.items():
            if key == c_lower:
                return val
        return c

    def _infer_state(self, college_name: str) -> str:
        name_lower = college_name.lower()
        for state, hints in STATE_COLLEGE_HINTS.items():
            for hint in hints:
                if hint in name_lower:
                    return state
        return "Unknown"
