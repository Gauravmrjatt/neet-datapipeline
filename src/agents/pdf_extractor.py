"""PDF Extraction Agent - extracts tabular data from PDFs using multiple extraction methods."""

import asyncio
import csv
import json
import re
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

MAX_WORKERS = 6
CONFIDENCE_THRESHOLD = 0.8


def _import_pdfplumber():
    try:
        import pdfplumber

        return pdfplumber
    except ImportError:
        return None


def _import_camelot():
    try:
        import camelot

        return camelot
    except ImportError:
        return None


def _import_tabula():
    try:
        import tabula

        return tabula
    except ImportError:
        return None


def _compute_confidence(tables: list[dict[str, Any]]) -> float:
    if not tables:
        return 0.0

    scores = []
    for table in tables:
        rows = table.get("rows", [])
        headers = table.get("headers", [])

        table_detection_score = 1.0 if headers and rows else 0.0

        row_count_score = min(len(rows) / 10.0, 1.0)

        total_cells = 0
        filled_cells = 0
        for row in rows:
            total_cells += len(row)
            filled_cells += sum(1 for cell in row if cell and str(cell).strip())

        cell_completeness = filled_cells / total_cells if total_cells > 0 else 0.0

        col_counts = [len(row) for row in rows]
        if col_counts:
            mode = max(set(col_counts), key=col_counts.count)
            consistent = sum(1 for c in col_counts if c == mode)
            column_consistency = consistent / len(col_counts)
        else:
            column_consistency = 0.0

        table_score = (
            table_detection_score * 0.3
            + row_count_score * 0.2
            + cell_completeness * 0.3
            + column_consistency * 0.2
        )
        scores.append(table_score)

    return sum(scores) / len(scores)


def _extract_with_pdfplumber(pdf_path: Path, mode: str = "lattice") -> list[dict[str, Any]]:
    pdfplumber = _import_pdfplumber()
    if not pdfplumber:
        return []

    tables_data = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                extracted = page.extract_tables(
                    table_settings={
                        "vertical_strategy": "text",
                        "horizontal_strategy": "text",
                        "snap_tolerance": 5,
                        "join_tolerance": 5,
                        "edge_min_length": 10,
                        "min_words_vertical": 3,
                        "min_words_horizontal": 3,
                    }
                    if mode == "stream"
                    else {}
                )

                if not extracted:
                    continue

                for table_idx, table in enumerate(extracted):
                    if not table or len(table) < 2:
                        continue

                    cleaned = []
                    for row in table:
                        cleaned_row = [
                            str(cell).strip() if cell is not None else "" for cell in row
                        ]
                        if any(cell for cell in cleaned_row):
                            cleaned.append(cleaned_row)

                    if len(cleaned) < 2:
                        continue

                    headers = cleaned[0]
                    rows = cleaned[1:]

                    tables_data.append(
                        {
                            "headers": headers,
                            "rows": rows,
                            "metadata": {
                                "page": page_num + 1,
                                "table_index": table_idx,
                                "row_count": len(rows),
                                "col_count": len(headers),
                            },
                        }
                    )
    except Exception as e:
        logger.warning(
            "pdfplumber_extraction_error",
            file=str(pdf_path),
            mode=mode,
            error=str(e),
        )

    return tables_data


def _extract_with_camelot(pdf_path: Path, mode: str = "lattice") -> list[dict[str, Any]]:
    camelot = _import_camelot()
    if not camelot:
        return []

    tables_data = []
    try:
        tables = camelot.read_pdf(
            str(pdf_path),
            pages="all",
            flavor=mode,
        )

        for table_idx, table in enumerate(tables):
            if table.df.empty or len(table.df) < 2:
                continue

            df = table.df
            headers = [str(h).strip() for h in df.columns.tolist()]
            rows = []
            for _, row in df.iterrows():
                cleaned_row = [str(cell).strip() for cell in row.tolist()]
                if any(cell for cell in cleaned_row):
                    rows.append(cleaned_row)

            if len(rows) < 1:
                continue

            tables_data.append(
                {
                    "headers": headers,
                    "rows": rows,
                    "metadata": {
                        "page": table.page,
                        "table_index": table_idx,
                        "accuracy": table.accuracy,
                        "row_count": len(rows),
                        "col_count": len(headers),
                    },
                }
            )
    except Exception as e:
        logger.warning(
            "camelot_extraction_error",
            file=str(pdf_path),
            mode=mode,
            error=str(e),
        )

    return tables_data


def _extract_with_tabula(pdf_path: Path) -> list[dict[str, Any]]:
    tabula = _import_tabula()
    if not tabula:
        return []

    tables_data = []
    try:
        dfs = tabula.read_pdf(
            str(pdf_path),
            pages="all",
            multiple_tables=True,
            lattice=True,
        )

        for table_idx, df in enumerate(dfs):
            if df.empty or len(df) < 2:
                continue

            headers = [str(h).strip() for h in df.columns.tolist()]
            rows = []
            for _, row in df.iterrows():
                cleaned_row = [str(cell).strip() for cell in row.tolist()]
                if any(cell for cell in cleaned_row):
                    rows.append(cleaned_row)

            if len(rows) < 1:
                continue

            tables_data.append(
                {
                    "headers": headers,
                    "rows": rows,
                    "metadata": {
                        "table_index": table_idx,
                        "row_count": len(rows),
                        "col_count": len(headers),
                    },
                }
            )
    except Exception as e:
        logger.warning(
            "tabula_extraction_error",
            file=str(pdf_path),
            error=str(e),
        )

    return tables_data


def _parse_metadata_from_filename(filename: str) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "year": None,
        "round": None,
        "source": "mcc",
        "type": "unknown",
    }

    year_match = re.search(r"(20[12]\d)", filename)
    if year_match:
        meta["year"] = int(year_match.group(1))

    title_lower = filename.lower()
    if "round 1" in title_lower or "r1" in title_lower or "first" in title_lower:
        meta["round"] = 1
    elif "round 2" in title_lower or "r2" in title_lower or "second" in title_lower:
        meta["round"] = 2
    elif "round 3" in title_lower or "r3" in title_lower or "mop" in title_lower:
        meta["round"] = 3
    elif "stray" in title_lower:
        meta["round"] = 4

    if "seat matrix" in title_lower:
        meta["type"] = "seat_matrix"
    elif "allotment" in title_lower:
        meta["type"] = "allotment"
    elif "result" in title_lower:
        meta["type"] = "result"
    elif "admitted" in title_lower:
        meta["type"] = "admitted"
    elif "cutoff" in title_lower or "cut off" in title_lower:
        meta["type"] = "cutoff"

    return meta


class PDFExtractionAgent:
    """Extracts tabular data from PDFs using a cascade of extraction methods."""

    EXTRACTED_DIR = "extracted"
    LOG_FILE = "extraction_log.csv"

    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir or Path("data")
        self.extracted_dir = self.output_dir / self.EXTRACTED_DIR
        self.extracted_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / self.LOG_FILE
        self.semaphore = asyncio.Semaphore(MAX_WORKERS)
        self._extraction_log: list[dict[str, Any]] = []
        self._completed_files: set[str] = set()
        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        checkpoint_path = self.extracted_dir / "_checkpoint.json"
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, "r") as f:
                    data = json.load(f)
                self._completed_files = set(data.get("completed_files", []))
                logger.info(
                    "extraction_checkpoint_loaded",
                    completed=len(self._completed_files),
                )
            except (json.JSONDecodeError, KeyError):
                self._completed_files = set()

    def _save_checkpoint(self) -> None:
        checkpoint_path = self.extracted_dir / "_checkpoint.json"
        data = {
            "completed_files": list(self._completed_files),
            "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
        }
        with open(checkpoint_path, "w") as f:
            json.dump(data, f, indent=2)

    def _read_manifest(self, manifest_path: Path) -> list[dict[str, str]]:
        entries = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (
                    row.get("download_status") == "completed"
                    and row.get("local_path")
                    and Path(row["local_path"]).exists()
                ):
                    entries.append(row)
        return entries

    def _extract_single_pdf(self, pdf_path: Path) -> dict[str, Any]:
        source_name = pdf_path.stem
        metadata = _parse_metadata_from_filename(pdf_path.name)

        methods = [
            ("pdfplumber_lattice", lambda: _extract_with_pdfplumber(pdf_path, "lattice")),
            ("pdfplumber_stream", lambda: _extract_with_pdfplumber(pdf_path, "stream")),
            ("camelot_lattice", lambda: _extract_with_camelot(pdf_path, "lattice")),
            ("camelot_stream", lambda: _extract_with_camelot(pdf_path, "stream")),
            ("tabula", lambda: _extract_with_tabula(pdf_path)),
        ]

        best_result: dict[str, Any] = {
            "source_file": pdf_path.name,
            "extraction_method": "none",
            "confidence": 0.0,
            "tables": [],
            "metadata": metadata,
            "needs_ocr": True,
        }

        for method_name, method_fn in methods:
            try:
                tables = method_fn()
                if not tables:
                    continue

                confidence = _compute_confidence(tables)

                logger.info(
                    "extraction_method_result",
                    file=pdf_path.name,
                    method=method_name,
                    confidence=round(confidence, 4),
                    table_count=len(tables),
                )

                if confidence > best_result["confidence"]:
                    best_result = {
                        "source_file": pdf_path.name,
                        "extraction_method": method_name,
                        "confidence": round(confidence, 4),
                        "tables": tables,
                        "metadata": metadata,
                        "needs_ocr": confidence < CONFIDENCE_THRESHOLD,
                    }

                if confidence >= CONFIDENCE_THRESHOLD:
                    break

            except Exception as e:
                logger.warning(
                    "extraction_method_failed",
                    file=pdf_path.name,
                    method=method_name,
                    error=str(e),
                )

        return best_result

    async def _extract_with_semaphore(
        self, pdf_path: Path
    ) -> dict[str, Any]:
        async with self.semaphore:
            return await asyncio.to_thread(self._extract_single_pdf, pdf_path)

    async def extract_all(
        self, manifest_path: Path
    ) -> tuple[Path, Path]:
        """Extract tables from all PDFs in the download manifest."""
        entries = self._read_manifest(manifest_path)
        logger.info("extraction_started", total_pdfs=len(entries))

        tasks = []
        for entry in entries:
            pdf_path = Path(entry["local_path"])
            if pdf_path.name not in self._completed_files:
                tasks.append(pdf_path)

        logger.info(
            "extraction_plan",
            to_extract=len(tasks),
            skipping=len(entries) - len(tasks),
        )

        pending = [
            asyncio.ensure_future(self._extract_with_semaphore(p))
            for p in tasks
        ]

        for future in asyncio.as_completed(pending):
            try:
                result = await future
            except Exception as e:
                logger.error("extraction_task_error", error=str(e))
                continue

            if not isinstance(result, dict):
                continue

            source_file = result.get("source_file", "unknown")
            json_path = self.extracted_dir / f"{source_file}.json"

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            self._completed_files.add(source_file)

            log_entry = {
                "source_file": source_file,
                "extraction_method": result.get("extraction_method", "none"),
                "confidence": result.get("confidence", 0.0),
                "table_count": len(result.get("tables", [])),
                "needs_ocr": result.get("needs_ocr", True),
                "output_path": str(json_path),
            }
            self._extraction_log.append(log_entry)

            logger.info(
                "file_extracted",
                file=source_file,
                method=result.get("extraction_method"),
                confidence=result.get("confidence"),
                tables=len(result.get("tables", [])),
                needs_ocr=result.get("needs_ocr"),
            )

            if len(self._completed_files) % 10 == 0:
                self._save_checkpoint()

        self._save_checkpoint()
        self._write_extraction_log()

        logger.info(
            "extraction_complete",
            total_extracted=len(self._completed_files),
            log_path=str(self.log_path),
        )
        return self.extracted_dir, self.log_path

    def _write_extraction_log(self) -> None:
        fieldnames = [
            "source_file",
            "extraction_method",
            "confidence",
            "table_count",
            "needs_ocr",
            "output_path",
        ]
        with open(self.log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for entry in self._extraction_log:
                writer.writerow(entry)
