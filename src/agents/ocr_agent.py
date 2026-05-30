"""OCR Agent - extracts tabular data from PDFs using OCR when standard extraction fails."""

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

MAX_OCR_WORKERS = 3
CONFIDENCE_THRESHOLD = 0.8

RANK_PATTERN = re.compile(
    r"(\d{1,6})\s+([A-Z][A-Za-z\s\.\-\']+?)(?:\s+([A-Z]{1,5}))?(?:\s+([A-Z][\w\s]+?))?\s*$",
    re.MULTILINE,
)

RANK_COLUMN_PATTERNS = [
    re.compile(r"(?:rank|sl\.?\s*no|s\.?no|roll\s*no)[\s:]*", re.IGNORECASE),
    re.compile(r"(?:college|institute|medical college)[\s:]*", re.IGNORECASE),
    re.compile(r"(?:category|cat|quota)[\s:]*", re.IGNORECASE),
    re.compile(r"(?:state|branch|course)[\s:]*", re.IGNORECASE),
]


def _import_pdf2image():
    try:
        from pdf2image import convert_from_path

        return convert_from_path
    except ImportError:
        return None


def _import_pytesseract():
    try:
        import pytesseract

        return pytesseract
    except ImportError:
        return None


def _import_pil():
    try:
        from PIL import Image, ImageFilter, ImageOps

        return Image, ImageFilter, ImageOps
    except ImportError:
        return None


def _preprocess_image(img: Any) -> Any:
    Image, ImageFilter, ImageOps = _import_pil()
    if not Image:
        return img

    if img.mode != "L":
        img = img.convert("L")

    img = ImageOps.autocontrast(img, cutoff=2)

    img = img.filter(ImageFilter.SHARPEN)

    threshold = 140
    img = img.point(lambda x: 255 if x > threshold else 0, "1")

    img = img.convert("L")

    width, height = img.size
    if width < 2000:
        scale = 2000 / width
        img = img.resize(
            (int(width * scale), int(height * scale)), Image.LANCZOS
        )

    return img


def _deskew_image(img: Any) -> Any:
    try:
        import numpy as np

        angles = []
        for angle in range(-5, 6, 1):
            rotated = img.rotate(angle, fillcolor=255)
            arr_rot = np.array(rotated)
            h_proj = np.sum(arr_rot < 128, axis=1)
            var = np.var(h_proj)
            angles.append((angle, var))

        best_angle = max(angles, key=lambda x: x[1])[0]
        if abs(best_angle) > 0.5:
            img = img.rotate(best_angle, fillcolor=255, expand=False)
    except Exception:
        pass

    return img


def _extract_text_from_pdf(pdf_path: Path) -> list[dict[str, Any]]:
    convert_from_path = _import_pdf2image()
    pytesseract = _import_pytesseract()

    if not convert_from_path or not pytesseract:
        logger.error("ocr_dependencies_missing", pdf=str(pdf_path))
        return []

    pages_data = []
    try:
        images = convert_from_path(
            str(pdf_path), dpi=300, fmt="jpeg", thread_count=2
        )

        for page_num, img in enumerate(images):
            processed = _preprocess_image(img)
            processed = _deskew_image(processed)

            custom_config = r"--oem 3 --psm 6"
            text = pytesseract.image_to_string(processed, config=custom_config)

            pages_data.append(
                {
                    "page_num": page_num + 1,
                    "text": text,
                    "width": img.size[0],
                    "height": img.size[1],
                }
            )

    except Exception as e:
        logger.error(
            "ocr_pdf_conversion_error",
            file=str(pdf_path),
            error=str(e),
        )

    return pages_data


def _parse_tables_from_text(pages_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    all_tables = []

    for page_data in pages_data:
        text = page_data["text"]
        lines = text.split("\n")

        table_rows = []
        current_headers: list[str] = []

        for line in lines:
            line = line.strip()
            if not line:
                if table_rows and len(table_rows) > 1:
                    all_tables.append(
                        {
                            "headers": current_headers if current_headers else _infer_headers(table_rows),
                            "rows": table_rows,
                            "metadata": {
                                "page": page_data["page_num"],
                                "row_count": len(table_rows),
                            },
                        }
                    )
                    table_rows = []
                continue

            if any(pat.search(line) for pat in RANK_COLUMN_PATTERNS):
                parts = re.split(r"\s{2,}|\t", line)
                current_headers = [p.strip() for p in parts if p.strip()]
                continue

            rank_match = re.match(r"^\s*(\d{1,6})\b", line)
            if rank_match:
                parts = re.split(r"\s{2,}|\t", line)
                cleaned = [p.strip() for p in parts if p.strip()]
                if cleaned:
                    table_rows.append(cleaned)

        if table_rows and len(table_rows) > 1:
            all_tables.append(
                {
                    "headers": current_headers if current_headers else _infer_headers(table_rows),
                    "rows": table_rows,
                    "metadata": {
                        "page": page_data["page_num"],
                        "row_count": len(table_rows),
                    },
                }
            )

    return all_tables


def _infer_headers(rows: list[list[str]]) -> list[str]:
    if not rows:
        return []

    max_cols = max(len(row) for row in rows)
    headers = []
    for col_idx in range(max_cols):
        col_values = [
            row[col_idx] for row in rows if col_idx < len(row) and row[col_idx]
        ]
        if not col_values:
            headers.append(f"col_{col_idx}")
            continue

        all_numeric = all(re.match(r"^\d+$", v) for v in col_values[:10])
        if all_numeric:
            if col_idx == 0:
                headers.append("rank")
            else:
                headers.append(f"col_{col_idx}")
        else:
            avg_len = sum(len(v) for v in col_values) / len(col_values)
            if avg_len > 15:
                headers.append("college_name" if col_idx > 0 else "col_0")
            else:
                headers.append(f"col_{col_idx}")

    return headers


def _compute_ocr_confidence(tables: list[dict[str, Any]]) -> float:
    if not tables:
        return 0.0

    scores = []
    for table in tables:
        rows = table.get("rows", [])
        headers = table.get("headers", [])

        table_detected = 1.0 if headers and rows else 0.0
        row_score = min(len(rows) / 10.0, 1.0)

        total_cells = sum(len(row) for row in rows)
        filled_cells = sum(
            1 for row in rows for cell in row if cell and str(cell).strip()
        )
        cell_score = filled_cells / total_cells if total_cells > 0 else 0.0

        col_counts = [len(row) for row in rows]
        if col_counts:
            mode = max(set(col_counts), key=col_counts.count)
            consistent = sum(1 for c in col_counts if c == mode)
            col_score = consistent / len(col_counts)
        else:
            col_score = 0.0

        table_score = (
            table_detected * 0.2 + row_score * 0.3 + cell_score * 0.3 + col_score * 0.2
        )
        scores.append(table_score)

    return sum(scores) / len(scores)


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


class OCRAgent:
    """Performs OCR extraction on PDFs that failed standard extraction."""

    EXTRACTED_DIR = "extracted"

    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir or Path("data")
        self.extracted_dir = self.output_dir / self.EXTRACTED_DIR
        self.extracted_dir.mkdir(parents=True, exist_ok=True)
        self.semaphore = asyncio.Semaphore(MAX_OCR_WORKERS)

    async def _ocr_single_pdf(self, pdf_path: Path) -> dict[str, Any] | None:
        async with self.semaphore:
            start_time = time.time()
            source_name = pdf_path.stem
            metadata = _parse_metadata_from_filename(pdf_path.name)

            logger.info("ocr_started", file=pdf_path.name)

            try:
                pages_data = await asyncio.to_thread(
                    _extract_text_from_pdf, pdf_path
                )

                if not pages_data:
                    logger.warning("ocr_no_pages", file=pdf_path.name)
                    return None

                tables = await asyncio.to_thread(
                    _parse_tables_from_text, pages_data
                )

                confidence = _compute_ocr_confidence(tables)
                elapsed = time.time() - start_time

                all_text = "\n".join(p["text"] for p in pages_data)
                char_count = len(all_text)
                word_count = len(all_text.split())

                result = {
                    "source_file": pdf_path.name,
                    "extraction_method": "ocr_tesseract",
                    "confidence": round(confidence, 4),
                    "tables": tables,
                    "metadata": {
                        **metadata,
                        "ocr_pages": len(pages_data),
                        "ocr_characters": char_count,
                        "ocr_words": word_count,
                        "processing_time_seconds": round(elapsed, 2),
                    },
                    "needs_ocr": False,
                }

                json_path = self.extracted_dir / f"{source_name}_ocr.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

                logger.info(
                    "ocr_completed",
                    file=pdf_path.name,
                    confidence=round(confidence, 4),
                    tables=len(tables),
                    pages=len(pages_data),
                    elapsed=round(elapsed, 2),
                )

                return result

            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(
                    "ocr_failed",
                    file=pdf_path.name,
                    error=str(e),
                    elapsed=round(elapsed, 2),
                )
                return None

    async def ocr_extract(self, pdf_paths: list[Path]) -> list[Path]:
        """Extract data from PDFs using OCR processing."""
        logger.info("ocr_batch_started", total_files=len(pdf_paths))

        tasks = [self._ocr_single_pdf(pdf_path) for pdf_path in pdf_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output_paths: list[Path] = []
        for result in results:
            if isinstance(result, Exception):
                logger.error("ocr_task_exception", error=str(result))
                continue
            if result and isinstance(result, dict):
                source_name = result.get("source_file", "unknown")
                json_path = self.extracted_dir / f"{source_name}_ocr.json"
                if json_path.exists():
                    output_paths.append(json_path)

        logger.info(
            "ocr_batch_complete",
            requested=len(pdf_paths),
            successful=len(output_paths),
        )
        return output_paths
