from __future__ import annotations

import asyncio
import csv
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

ALLOWED_CATEGORIES = {
    "General",
    "OBC-NCL",
    "SC",
    "ST",
    "EWS",
    "General PwD",
    "OBC-NCL PwD",
    "SC PwD",
    "ST PwD",
    "EWS PwD",
}

ALLOWED_QUOTAS = {
    "All India",
    "AMU Quota",
    "Delhi University",
    "ESI",
    "IP University",
    "Deemed/Paid Seats",
    "NRI",
    "Open Seat",
    "State Quota",
}

VALID_ROUNDS = set(range(1, 11)) | {100, 101}

INDIAN_STATES = {
    "Andhra Pradesh",
    "Arunachal Pradesh",
    "Assam",
    "Bihar",
    "Chhattisgarh",
    "Goa",
    "Gujarat",
    "Haryana",
    "Himachal Pradesh",
    "Jharkhand",
    "Karnataka",
    "Kerala",
    "Madhya Pradesh",
    "Maharashtra",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Nagaland",
    "Odisha",
    "Punjab",
    "Rajasthan",
    "Sikkim",
    "Tamil Nadu",
    "Telangana",
    "Tripura",
    "Uttar Pradesh",
    "Uttarakhand",
    "West Bengal",
    "Delhi",
    "Jammu and Kashmir",
    "Ladakh",
    "Chandigarh",
    "Puducherry",
    "Andaman and Nicobar Islands",
    "Dadra and Nagar Haveli",
    "Daman and Diu",
    "Lakshadweep",
}


class ValidationAgent:
    """Validates normalized NEET counselling dataset against business rules."""

    def __init__(self, max_concurrency: int = 3) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._rules = [
            self._check_rank_order,
            self._check_category,
            self._check_quota,
            self._check_college_name,
            self._check_duplicates,
            self._check_rank_sanity,
            self._check_year_range,
            self._check_round_range,
            self._check_state_name,
        ]

    async def validate(
        self, normalized_path: Path
    ) -> tuple[Path, dict]:
        """Validate the normalized dataset and return (report_path, summary_stats)."""
        logger.info("Starting validation of %s", normalized_path)
        df = pd.read_csv(normalized_path)
        logger.info("Loaded %d rows for validation", len(df))

        all_results: list[dict] = []

        async def _run_rule(rule_fn, rule_name: str) -> list[dict]:
            async with self._semaphore:
                logger.info("Running rule: %s", rule_name)
                return rule_fn(df)

        tasks = [
            _run_rule(rule_fn, rule_fn.__name__)
            for rule_fn in self._rules
        ]

        rule_results = await asyncio.gather(*tasks)
        for results in rule_results:
            all_results.extend(results)

        report_df = pd.DataFrame(all_results)
        report_path = normalized_path.parent.parent / "validated" / "validation_report.csv"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(report_path, index=False)

        summary = self._build_summary(df, report_df)
        logger.info(
            "Validation complete. Pass rate: %.2f%%", summary["pass_rate"] * 100
        )
        return report_path, summary

    def _check_rank_order(self, df: pd.DataFrame) -> list[dict]:
        results = []
        for idx, row in df.iterrows():
            opening = row.get("opening_rank")
            closing = row.get("closing_rank")
            passed = True
            detail = ""
            if pd.isna(opening) or pd.isna(closing):
                passed = False
                detail = "opening_rank or closing_rank is null"
            elif opening > closing:
                passed = False
                detail = f"opening_rank ({opening}) > closing_rank ({closing})"
            results.append(
                {"row_id": idx, "rule_name": "rank_order", "passed": passed, "details": detail}
            )
        return results

    def _check_category(self, df: pd.DataFrame) -> list[dict]:
        results = []
        for idx, row in df.iterrows():
            cat = row.get("category", "")
            passed = cat in ALLOWED_CATEGORIES
            detail = "" if passed else f"Invalid category: '{cat}'"
            results.append(
                {"row_id": idx, "rule_name": "valid_category", "passed": passed, "details": detail}
            )
        return results

    def _check_quota(self, df: pd.DataFrame) -> list[dict]:
        results = []
        for idx, row in df.iterrows():
            q = row.get("quota", "")
            passed = q in ALLOWED_QUOTAS
            detail = "" if passed else f"Invalid quota: '{q}'"
            results.append(
                {"row_id": idx, "rule_name": "valid_quota", "passed": passed, "details": detail}
            )
        return results

    def _check_college_name(self, df: pd.DataFrame) -> list[dict]:
        results = []
        for idx, row in df.iterrows():
            name = row.get("college_name")
            passed = bool(name and str(name).strip())
            detail = "" if passed else "college_name is null or empty"
            results.append(
                {"row_id": idx, "rule_name": "college_name_valid", "passed": passed, "details": detail}
            )
        return results

    def _check_duplicates(self, df: pd.DataFrame) -> list[dict]:
        results = []
        dup_cols = [
            "college_name",
            "year",
            "round",
            "quota",
            "category",
            "closing_rank",
        ]
        existing = set()
        for idx, row in df.iterrows():
            key = tuple(row[c] for c in dup_cols)
            passed = key not in existing
            if passed:
                existing.add(key)
            detail = "" if passed else f"Duplicate composite key: {key}"
            results.append(
                {"row_id": idx, "rule_name": "no_duplicates", "passed": passed, "details": detail}
            )
        return results

    def _check_rank_sanity(self, df: pd.DataFrame) -> list[dict]:
        results = []
        for idx, row in df.iterrows():
            passed = True
            detail = ""
            for col in ("opening_rank", "closing_rank"):
                val = row.get(col)
                if pd.notna(val):
                    try:
                        rank = int(val)
                        if not (1 <= rank <= 2_000_000):
                            passed = False
                            detail = f"{col}={rank} out of range [1, 2000000]"
                            break
                    except (ValueError, TypeError):
                        passed = False
                        detail = f"{col} is not a valid integer: {val}"
                        break
            results.append(
                {"row_id": idx, "rule_name": "rank_sanity", "passed": passed, "details": detail}
            )
        return results

    def _check_year_range(self, df: pd.DataFrame) -> list[dict]:
        results = []
        for idx, row in df.iterrows():
            year = row.get("year")
            passed = pd.notna(year) and 2019 <= int(year) <= 2025
            detail = "" if passed else f"Invalid year: {year}"
            results.append(
                {"row_id": idx, "rule_name": "year_range", "passed": passed, "details": detail}
            )
        return results

    def _check_round_range(self, df: pd.DataFrame) -> list[dict]:
        results = []
        for idx, row in df.iterrows():
            rnd = row.get("round")
            try:
                passed = int(rnd) in VALID_ROUNDS
            except (ValueError, TypeError):
                passed = False
            detail = "" if passed else f"Invalid round: {rnd}"
            results.append(
                {"row_id": idx, "rule_name": "round_range", "passed": passed, "details": detail}
            )
        return results

    def _check_state_name(self, df: pd.DataFrame) -> list[dict]:
        results = []
        for idx, row in df.iterrows():
            state = row.get("state", "")
            passed = state in INDIAN_STATES
            detail = "" if passed else f"Invalid state: '{state}'"
            results.append(
                {"row_id": idx, "rule_name": "valid_state", "passed": passed, "details": detail}
            )
        return results

    def _build_summary(self, df: pd.DataFrame, report_df: pd.DataFrame) -> dict:
        total_rows = len(df)
        if report_df.empty:
            return {
                "total_rows": total_rows,
                "passed_rows": 0,
                "failed_rows": total_rows,
                "pass_rate": 0.0,
                "per_rule_stats": {},
            }

        failed_row_ids = set(report_df.loc[~report_df["passed"], "row_id"].unique())
        failed_rows = len(failed_row_ids)
        passed_rows = total_rows - failed_rows
        pass_rate = passed_rows / total_rows if total_rows > 0 else 0.0

        per_rule: dict[str, dict] = {}
        for rule_name, group in report_df.groupby("rule_name"):
            total = len(group)
            passed = int(group["passed"].sum())
            per_rule[rule_name] = {
                "total": total,
                "passed": passed,
                "failed": total - passed,
                "pass_rate": passed / total if total > 0 else 0.0,
            }

        return {
            "total_rows": total_rows,
            "passed_rows": passed_rows,
            "failed_rows": failed_rows,
            "pass_rate": pass_rate,
            "per_rule_stats": per_rule,
        }
