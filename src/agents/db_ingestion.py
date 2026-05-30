from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

logger = logging.getLogger(__name__)

DB_URL = "postgresql+asyncpg://neet_user:neet_pass_2024@localhost:5432/neet_counselling"

COLLEGE_UPSERT = text(
    """
    INSERT INTO colleges (name, state, ownership, type)
    VALUES (:name, :state, :ownership, :type)
    ON CONFLICT (name, state) DO UPDATE SET
        ownership = COALESCE(EXCLUDED.ownership, colleges.ownership),
        type = COALESCE(EXCLUDED.type, colleges.type)
    RETURNING college_id
    """
)

CUTOFF_UPSERT = text(
    """
    INSERT INTO cutoffs (
        college_id, year, round, quota, category, course,
        opening_rank, closing_rank, seat_type, source_file, confidence
    ) VALUES (
        :college_id, :year, :round, :quota, :category, :course,
        :opening_rank, :closing_rank, :seat_type, :source_file, :confidence
    )
    """
)

CHECKPOINT_FILE = "db_ingestion_checkpoint.json"


class DBIngestionAgent:
    """Ingests normalized NEET data into PostgreSQL with async batch processing."""

    def __init__(
        self,
        db_url: str = DB_URL,
        batch_size: int = 1000,
        max_workers: int = 3,
    ) -> None:
        self._db_url = db_url
        self._batch_size = batch_size
        self._max_workers = max_workers
        self._engine: Optional[AsyncEngine] = None

    async def ingest(self, normalized_path: Path) -> dict:
        """Ingest normalized CSV into PostgreSQL. Returns stats dict."""
        self._engine = create_async_engine(self._db_url, echo=False)
        df = pd.read_csv(normalized_path)
        logger.info("Loaded %d rows for ingestion", len(df))

        checkpoint = self._load_checkpoint(normalized_path)
        start_idx = checkpoint.get("last_processed_row", 0)
        if start_idx > 0:
            logger.info("Resuming from row %d", start_idx)
            df = df.iloc[start_idx:]

        states = df["state"].unique().tolist()
        state_groups = [df[df["state"] == s] for s in states]

        stats = {"colleges_inserted": 0, "cutoffs_inserted": 0, "errors": 0}
        total_rows = len(df)

        sem = asyncio.Semaphore(self._max_workers)

        async def _process_group(group: pd.DataFrame, state: str) -> dict:
            async with sem:
                return await self._ingest_group(group, state, normalized_path.name)

        tasks = [
            _process_group(group, state)
            for group, state in zip(state_groups, states)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logger.error("Worker failed: %s", result)
                stats["errors"] += 1
            else:
                stats["colleges_inserted"] += result["colleges"]
                stats["cutoffs_inserted"] += result["cutoffs"]

        self._save_checkpoint(normalized_path, total_rows)
        logger.info("Ingestion complete: %s", stats)
        await self._engine.dispose()
        return stats

    async def _ingest_group(
        self, group: pd.DataFrame, state: str, source_file: str
    ) -> dict:
        """Ingest a state-partitioned group of rows."""
        colleges_map: dict[str, int] = {}
        cutoffs_count = 0
        errors = 0

        async with self._engine.connect() as conn:
            for start in range(0, len(group), self._batch_size):
                batch = group.iloc[start : start + self._batch_size]

                for _, row in batch.iterrows():
                    try:
                        college_name = str(row.get("college_name", "")).strip()
                        if not college_name or college_name == "nan" or college_name == "Unknown":
                            continue

                        year = row.get("year")
                        if pd.isna(year):
                            continue
                        year = int(year)

                        rnd = row.get("round", row.get("round_number"))
                        if pd.isna(rnd):
                            rnd = 0
                        else:
                            rnd = int(rnd)

                        college_key = f"{college_name}|{state}"
                        if college_key not in colleges_map:
                            result = await conn.execute(
                                COLLEGE_UPSERT,
                                {
                                    "name": college_name,
                                    "state": state,
                                    "ownership": "unknown",
                                    "type": "unknown",
                                },
                            )
                            college_id = result.scalar_one()
                            colleges_map[college_key] = college_id

                        college_id = colleges_map[college_key]

                        opening = row.get("opening_rank")
                        closing = row.get("closing_rank")
                        opening_rank = int(float(opening)) if pd.notna(opening) else None
                        closing_rank = int(float(closing)) if pd.notna(closing) else None

                        await conn.execute(
                            CUTOFF_UPSERT,
                            {
                                "college_id": college_id,
                                "year": year,
                                "round": rnd,
                                "quota": str(row.get("quota", "All India")),
                                "category": str(row.get("category", "General")),
                                "course": str(row.get("course", "MBBS")),
                                "opening_rank": opening_rank,
                                "closing_rank": closing_rank,
                                "seat_type": "government",
                                "source_file": str(row.get("source_file", "")),
                                "confidence": 1.0,
                            },
                        )
                        cutoffs_count += 1
                    except Exception as e:
                        errors += 1
                        if errors <= 5:
                            logger.warning("Error ingesting row: %s", str(e)[:200])
                        try:
                            await conn.rollback()
                        except Exception:
                            pass

                await conn.commit()
                logger.info(
                    "State=%s committed batch rows %d-%d",
                    state,
                    start,
                    min(start + self._batch_size, len(group)),
                )

        return {"colleges": len(colleges_map), "cutoffs": cutoffs_count, "errors": errors}

    def _load_checkpoint(self, normalized_path: Path) -> dict:
        cp_file = normalized_path.parent.parent / CHECKPOINT_FILE
        if cp_file.exists():
            try:
                with open(cp_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_checkpoint(self, normalized_path: Path, last_row: int) -> None:
        cp_file = normalized_path.parent.parent / CHECKPOINT_FILE
        cp_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cp_file, "w") as f:
            json.dump({"last_processed_row": last_row}, f)
