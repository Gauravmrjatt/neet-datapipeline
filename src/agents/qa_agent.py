from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

logger = logging.getLogger(__name__)

DB_URL = "postgresql+asyncpg://neet_user:neet_pass_2024@localhost:5432/neet_counselling"

QUERY = text(
    """
    SELECT
        c.name AS college_name,
        c.state,
        co.year,
        co.round,
        co.quota,
        co.category,
        co.course,
        co.opening_rank,
        co.closing_rank,
        co.seat_type,
        co.source_file,
        co.confidence,
        sm.seat_count
    FROM cutoffs co
    JOIN colleges c ON c.college_id = co.college_id
    LEFT JOIN seat_matrix sm
        ON sm.college_id = co.college_id
        AND sm.year = co.year
        AND sm.course = co.course
        AND sm.category = co.category
    ORDER BY c.name, co.year, co.round, co.quota, co.category
    """
)


class QADatasetBuilder:
    """Builds ML-ready dataset from PostgreSQL cutoffs, colleges, and seat_matrix."""

    def __init__(self, db_url: str = DB_URL) -> None:
        self._db_url = db_url

    async def build_ml_dataset(self, normalized_path: Path) -> dict:
        """Join tables, add features, split train/test, write CSVs."""
        engine = create_async_engine(self._db_url, echo=False)
        async with engine.connect() as conn:
            result = await conn.execute(QUERY)
            rows = result.fetchall()
            columns = result.keys()

        df = pd.DataFrame(rows, columns=columns)
        logger.info("Fetched %d rows from database", len(df))

        if df.empty:
            logger.warning("No data found in database")
            output_dir = normalized_path.parent.parent / "ml_ready"
            output_dir.mkdir(parents=True, exist_ok=True)
            empty = pd.DataFrame()
            empty.to_csv(output_dir / "neet_cutoff_dataset.csv", index=False)
            empty.to_csv(output_dir / "train.csv", index=False)
            empty.to_csv(output_dir / "test.csv", index=False)
            return {
                "total_rows": 0,
                "train_rows": 0,
                "test_rows": 0,
                "output_paths": {
                    "dataset": str(output_dir / "neet_cutoff_dataset.csv"),
                    "train": str(output_dir / "train.csv"),
                    "test": str(output_dir / "test.csv"),
                },
            }

        df = self._add_derived_features(df)

        output_dir = normalized_path.parent.parent / "ml_ready"
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset_path = output_dir / "neet_cutoff_dataset.csv"
        df.to_csv(dataset_path, index=False)

        train_df, test_df = self._stratified_split(df)
        train_path = output_dir / "train.csv"
        test_path = output_dir / "test.csv"
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        stats = {
            "total_rows": len(df),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "output_paths": {
                "dataset": str(dataset_path),
                "train": str(train_path),
                "test": str(test_path),
            },
        }

        self._log_dataset_stats(df, train_df, test_df)
        await engine.dispose()
        return stats

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(
            ["college_name", "year", "round", "quota", "category"]
        ).reset_index(drop=True)

        df["historical_cutoff_gap"] = self._compute_historical_gap(df)

        df["category_encoded"] = self._label_encode(df, "category")
        df["quota_encoded"] = self._label_encode(df, "quota")
        df["state_encoded"] = self._label_encode(df, "state")

        df["seat_ratio"] = self._compute_seat_ratio(df)

        df["admission_possible"] = self._compute_admission_target(df)

        return df

    def _compute_historical_gap(self, df: pd.DataFrame) -> pd.Series:
        gaps = pd.Series(np.nan, index=df.index)
        for (college, quota, category), group in df.groupby(
            ["college_name", "quota", "category"]
        ):
            sorted_g = group.sort_values("year")
            prev_closing = sorted_g["closing_rank"].shift(1)
            gap = prev_closing - sorted_g["closing_rank"]
            gaps.loc[sorted_g.index] = gap
        return gaps

    def _label_encode(self, df: pd.DataFrame, column: str) -> pd.Series:
        unique_vals = sorted(df[column].dropna().unique())
        mapping = {v: i for i, v in enumerate(unique_vals)}
        return df[column].map(mapping).fillna(-1).astype(int)

    def _compute_seat_ratio(self, df: pd.DataFrame) -> pd.Series:
        ratios = pd.Series(np.nan, index=df.index)
        for (college, category), group in df.groupby(["college_name", "category"]):
            total_seats = group["seat_count"].sum()
            if total_seats > 0:
                ratios.loc[group.index] = group["seat_count"] / total_seats
            else:
                ratios.loc[group.index] = 0.0
        return ratios

    def _compute_admission_target(self, df: pd.DataFrame) -> pd.Series:
        median_rank = df["closing_rank"].median()
        if pd.isna(median_rank) or median_rank == 0:
            return pd.Series(0, index=df.index, dtype=int)

        target = (df["closing_rank"] >= median_rank).astype(int)
        return target

    def _stratified_split(
        self, df: pd.DataFrame, train_ratio: float = 0.8
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = df.copy()
        df["_split_key"] = df["college_name"] + "_" + df["year"].astype(str)

        split_keys = df["_split_key"].unique()
        np.random.seed(42)
        np.random.shuffle(split_keys)
        split_idx = int(len(split_keys) * train_ratio)
        train_keys = set(split_keys[:split_idx])
        test_keys = set(split_keys[split_idx:])

        train_df = df[df["_split_key"].isin(train_keys)].drop(columns=["_split_key"])
        test_df = df[df["_split_key"].isin(test_keys)].drop(columns=["_split_key"])

        return train_df, test_df

    def _log_dataset_stats(
        self,
        df: pd.DataFrame,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        logger.info("Dataset stats:")
        logger.info("  Total rows: %d", len(df))
        logger.info("  Train rows: %d", len(train_df))
        logger.info("  Test rows: %d", len(test_df))
        logger.info("  Unique colleges: %d", df["college_name"].nunique())
        logger.info("  Years covered: %s", sorted(df["year"].unique()))
        logger.info("  Categories: %s", sorted(df["category"].unique()))
        logger.info("  Mean closing_rank: %.0f", df["closing_rank"].mean())
        logger.info("  Median closing_rank: %.0f", df["closing_rank"].median())
        logger.info(
            "  Admission possible distribution: %s",
            df["admission_possible"].value_counts().to_dict(),
        )
