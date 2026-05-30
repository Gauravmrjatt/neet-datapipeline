from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

ALLOTMENT_KEYWORDS = {
    "rank",
    "air",
    "all india rank",
    "allotted",
    "quota",
    "institute",
    "college",
    "candidate",
    "category",
    "remarks",
    "option",
}

SEAT_MATRIX_KEYWORDS = {
    "seat",
    "total",
    "totalseats",
    "total seats",
    "sr no",
    "pwd",
    "state",
}

MERIT_LIST_KEYWORDS = {
    "merit",
    "position",
    "percentile",
    "score",
    "marks",
    "percentile score",
}

EXPECTED_ALLOTMENT_COLS = {
    "rank",
    "air",
    "quota",
    "college_name",
    "institute",
    "category",
    "course",
}

EXPECTED_SEAT_MATRIX_COLS = {
    "college_name",
    "total_seats",
    "category",
    "quota",
}

EXPECTED_MERIT_COLS = {
    "rank",
    "score",
    "marks",
}


def _normalize(col: str) -> str:
    return re.sub(r"\s+", " ", col.strip().lower())


def _count_keyword_hits(headers: list[str], keywords: set[str]) -> int:
    count = 0
    for h in headers:
        normalized = _normalize(h)
        for kw in keywords:
            if kw in normalized:
                count += 1
                break
    return count


def _header_quality_score(headers: list[str]) -> float:
    if not headers:
        return 0.0

    non_empty = sum(1 for h in headers if h and h.strip())
    unique = len(set(_normalize(h) for h in headers if h and h.strip()))

    completeness = non_empty / len(headers) if headers else 0.0
    uniqueness = unique / non_empty if non_empty > 0 else 0.0

    return (completeness + uniqueness) / 2.0


def _data_completeness_score(rows: list[list], num_cols: int) -> float:
    if not rows or num_cols == 0:
        return 0.0

    total_cells = len(rows) * num_cols
    filled_cells = 0
    for row in rows:
        for cell in row:
            if cell is not None and str(cell).strip():
                filled_cells += 1

    return filled_cells / total_cells if total_cells > 0 else 0.0


def _column_consistency_score(rows: list[list], expected_cols: int) -> float:
    if not rows:
        return 0.0

    matching = sum(1 for row in rows if len(row) == expected_cols)
    return matching / len(rows)


class TableDetector:
    """Analyzes extracted table data to determine type and quality."""

    def detect(self, headers: list[str], rows: list[list]) -> dict:
        """Detect table type and compute quality metadata.

        Returns dict with keys:
            table_type: "allotment" | "seat_matrix" | "merit_list" | "unknown"
            quality_score: float [0, 1]
            expected_columns: set[str]
            actual_columns: list[str]
            header_score: float
            data_completeness: float
            column_consistency: float
        """
        table_type = self._detect_type(headers)
        expected_cols = self._get_expected_columns(table_type)

        header_score = _header_quality_score(headers)
        data_completeness = _data_completeness_score(rows, len(headers))
        column_consistency = _column_consistency_score(rows, len(headers))

        quality_score = (header_score * 0.3) + (data_completeness * 0.4) + (column_consistency * 0.3)

        result = {
            "table_type": table_type,
            "quality_score": round(quality_score, 4),
            "expected_columns": expected_cols,
            "actual_columns": headers,
            "header_score": round(header_score, 4),
            "data_completeness": round(data_completeness, 4),
            "column_consistency": round(column_consistency, 4),
        }

        logger.info(
            "Detected table type=%s quality=%.2f cols=%d rows=%d",
            table_type,
            quality_score,
            len(headers),
            len(rows),
        )
        return result

    def _detect_type(self, headers: list[str]) -> str:
        allotment_hits = _count_keyword_hits(headers, ALLOTMENT_KEYWORDS)
        seat_matrix_hits = _count_keyword_hits(headers, SEAT_MATRIX_KEYWORDS)
        merit_hits = _count_keyword_hits(headers, MERIT_LIST_KEYWORDS)

        scores = {
            "allotment": allotment_hits,
            "seat_matrix": seat_matrix_hits,
            "merit_list": merit_hits,
        }

        best_type = max(scores, key=scores.get)  # type: ignore
        best_score = scores[best_type]

        if best_score == 0:
            return "unknown"

        second_best = sorted(scores.values(), reverse=True)[1]
        if best_score == second_best:
            logger.warning(
                "Ambiguous table type: allotment=%d seat_matrix=%d merit=%d",
                allotment_hits,
                seat_matrix_hits,
                merit_hits,
            )
            return "unknown"

        return best_type

    def _get_expected_columns(self, table_type: str) -> set[str]:
        if table_type == "allotment":
            return EXPECTED_ALLOTMENT_COLS
        elif table_type == "seat_matrix":
            return EXPECTED_SEAT_MATRIX_COLS
        elif table_type == "merit_list":
            return EXPECTED_MERIT_COLS
        return set()
