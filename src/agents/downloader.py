"""Download Agent - downloads PDFs from discovered links using aiohttp."""

import asyncio
import csv
import hashlib
import re
from pathlib import Path
from typing import Any

import aiohttp
import structlog

logger = structlog.get_logger(__name__)

CHUNK_SIZE = 8192
DEFAULT_CONCURRENCY = 8
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2


class DownloadAgent:
    """Downloads PDF files from link inventory using aiohttp with resumable support."""

    MANIFEST_FILE = "download_manifest.csv"
    MANIFEST_FIELDS = [
        "url",
        "local_path",
        "filename",
        "sha256",
        "size_bytes",
        "download_status",
        "retry_count",
    ]

    def __init__(
        self, output_dir: Path | None = None, concurrency: int = DEFAULT_CONCURRENCY
    ) -> None:
        self.output_dir = output_dir or Path("data")
        self.raw_dir = self.output_dir / "raw" / "mcc"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / self.MANIFEST_FILE
        self.semaphore = asyncio.Semaphore(concurrency)
        self._manifest_entries: dict[str, dict[str, Any]] = {}
        self._load_manifest()

    def _load_manifest(self) -> None:
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get("url"):
                            self._manifest_entries[row["url"]] = row
                logger.info(
                    "manifest_loaded",
                    entries=len(self._manifest_entries),
                    path=str(self.manifest_path),
                )
            except Exception as e:
                logger.warning("manifest_load_failed", error=str(e))

    def _save_manifest(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.MANIFEST_FIELDS)
            writer.writeheader()
            for entry in self._manifest_entries.values():
                writer.writerow(entry)

    def _read_inventory(self, inventory_path: Path) -> list[dict[str, str]]:
        entries = []
        with open(inventory_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("url") and row.get("url").strip():
                    entries.append(row)
        return entries

    def _sanitize_filename(self, url: str, title: str) -> str:
        name = re.sub(r"[^\w\s\-]", "", title)
        name = re.sub(r"\s+", "_", name).strip("_")
        if not name:
            name = Path(url.split("?")[0]).stem
            name = re.sub(r"[^\w\-]", "_", name)[:100]
        return name[:150]

    def _compute_sha256(self, file_path: Path) -> str:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    async def _download_file(
        self,
        session: aiohttp.ClientSession,
        url: str,
        dest_path: Path,
    ) -> dict[str, Any]:
        async with self.semaphore:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

            if dest_path.exists() and dest_path.stat().st_size > 0:
                sha256 = self._compute_sha256(dest_path)
                logger.info("file_already_exists", path=str(dest_path))
                return {
                    "url": url,
                    "local_path": str(dest_path),
                    "filename": dest_path.name,
                    "sha256": sha256,
                    "size_bytes": str(dest_path.stat().st_size),
                    "download_status": "completed",
                    "retry_count": "0",
                }

            last_error = None
            for attempt in range(MAX_RETRIES):
                try:
                    async with session.get(
                        url,
                        timeout=aiohttp.ClientTimeout(total=120),
                        allow_redirects=True,
                    ) as response:
                        if response.status != 200:
                            last_error = f"HTTP {response.status}"
                            logger.warning(
                                "download_http_error",
                                url=url,
                                status=response.status,
                                attempt=attempt + 1,
                            )
                            delay = RETRY_BASE_DELAY * (2**attempt)
                            await asyncio.sleep(delay)
                            continue

                        total_size = 0
                        with open(temp_path, "wb") as f:
                            async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                                f.write(chunk)
                                total_size += len(chunk)

                    if temp_path.exists():
                        sha256 = self._compute_sha256(temp_path)
                        temp_path.rename(dest_path)

                        logger.info(
                            "download_completed",
                            url=url,
                            filename=dest_path.name,
                            size=total_size,
                            sha256=sha256[:16],
                        )

                        return {
                            "url": url,
                            "local_path": str(dest_path),
                            "filename": dest_path.name,
                            "sha256": sha256,
                            "size_bytes": str(total_size),
                            "download_status": "completed",
                            "retry_count": str(attempt),
                        }

                except asyncio.TimeoutError:
                    last_error = "timeout"
                    logger.warning(
                        "download_timeout", url=url, attempt=attempt + 1
                    )
                except aiohttp.ClientError as e:
                    last_error = str(e)
                    logger.warning(
                        "download_client_error",
                        url=url,
                        attempt=attempt + 1,
                        error=str(e),
                    )
                except Exception as e:
                    last_error = str(e)
                    logger.warning(
                        "download_error",
                        url=url,
                        attempt=attempt + 1,
                        error=str(e),
                    )

                delay = RETRY_BASE_DELAY * (2**attempt)
                await asyncio.sleep(delay)

            if temp_path.exists():
                temp_path.unlink()

            logger.error("download_failed", url=url, error=last_error)
            return {
                "url": url,
                "local_path": "",
                "filename": "",
                "sha256": "",
                "size_bytes": "0",
                "download_status": "failed",
                "retry_count": str(MAX_RETRIES),
            }

    async def download_all(self, inventory_path: Path) -> Path:
        """Download all PDFs from inventory with resumable support."""
        entries = self._read_inventory(inventory_path)
        logger.info("download_started", total_entries=len(entries))

        skip_count = 0
        download_tasks = []
        for entry in entries:
            url = entry["url"]
            year = entry.get("year", "unknown")
            title = entry.get("title", "untitled")

            if url in self._manifest_entries:
                existing = self._manifest_entries[url]
                if existing.get("download_status") == "completed":
                    skip_count += 1
                    continue

            filename = self._sanitize_filename(url, title) + ".pdf"
            dest_path = self.raw_dir / str(year) / filename
            download_tasks.append((url, dest_path))

        logger.info(
            "download_plan",
            to_download=len(download_tasks),
            skipping=skip_count,
        )

        async with aiohttp.ClientSession(
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
                "Accept": "application/pdf,*/*",
            }
        ) as session:
            tasks = [
                self._download_file(session, url, dest) for url, dest in download_tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, dict):
                self._manifest_entries[result["url"]] = result
            elif isinstance(result, Exception):
                logger.error("download_task_exception", error=str(result))

        self._save_manifest()
        logger.info(
            "download_complete",
            total=len(self._manifest_entries),
            manifest_path=str(self.manifest_path),
        )
        return self.manifest_path
