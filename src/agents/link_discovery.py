"""MCC Link Discovery Agent - crawls mcc.nic.in archive pages to discover PDF links."""

import asyncio
import csv
import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp
from bs4 import BeautifulSoup
import structlog

logger = structlog.get_logger(__name__)

TITLE_KEYWORDS = [
    "final result",
    "seat matrix",
    "allotment",
    "admitted",
    "seat allotment",
    "first round",
    "second round",
    "mop up",
    "stray vacancy",
    "round 1",
    "round 2",
    "round 3",
    "all india",
]

DOC_TYPE_PATTERNS = {
    "seat_matrix": ["seat matrix", "seat allotment matrix"],
    "allotment": ["allotment", "seat allotment list"],
    "result": ["final result", "result"],
    "admitted": ["admitted", "admitted list"],
    "cutoff": ["cutoff", "cut off", "cutoff marks"],
}

ROUND_PATTERNS = [
    (r"round\s*[\-:\s]*1|first round|r1", 1),
    (r"round\s*[\-:\s]*2|second round|r2", 2),
    (r"round\s*[\-:\s]*3|third round|r3|mop\s*up", 3),
    (r"stray|vacancy round|stray round", 4),
    (r"special stray|special round", 5),
]

STATE_PATTERNS = {
    "Uttar Pradesh": ["u.p.", "uttar pradesh", "up"],
    "Madhya Pradesh": ["m.p.", "madhya pradesh", "mp"],
    "Tamil Nadu": ["t.n.", "tamil nadu", "tn"],
    "Andhra Pradesh": ["a.p.", "andhra pradesh", "ap"],
    "West Bengal": ["west bengal", "w.b.", "wb"],
    "Maharashtra": ["maharashtra", "maha"],
    "Rajasthan": ["rajasthan", "raj"],
    "Bihar": ["bihar"],
    "Karnataka": ["karnataka", "ka"],
    "Kerala": ["kerala", "kl"],
    "Telangana": ["telangana", "ts"],
    "Gujarat": ["gujarat", "gj"],
    "Punjab": ["punjab", "pb"],
    "Haryana": ["haryana", "hr"],
    "Delhi": ["delhi", "nct of delhi"],
    "Jammu & Kashmir": ["jammu", "kashmir", "j&k"],
    "Odisha": ["odisha", "orissa"],
    "Chhattisgarh": ["chhattisgarh", "cg"],
    "Jharkhand": ["jharkhand", "jh"],
    "Uttarakhand": ["uttarakhand", "uk"],
    "Assam": ["assam"],
    "Goa": ["goa"],
    "Himachal Pradesh": ["himachal", "hp"],
    "All India": ["all india", "ai"],
}


class MCCLinkDiscoveryAgent:
    """Crawls MCC archive pages to discover and inventory PDF document links."""

    BASE_URL = "https://mcc.nic.in"
    ARCHIVE_URL = "https://mcc.nic.in/archive-ug/"
    CDN_BASE = "https://cdnbbsr.s3waas.gov.in/s3e0f7a4d0ef9b84b83b693bbf3feb8e6e/uploads/"
    CHECKPOINT_FILE = "link_discovery_checkpoint.json"

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir or Path("data")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.inventory_path = self.output_dir / "links_inventory.csv"
        self.checkpoint_path = self.output_dir / self.CHECKPOINT_FILE
        self.semaphore = asyncio.Semaphore(3)
        self._discovered_urls: dict[str, dict[str, Any]] = {}
        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, "r") as f:
                    data = json.load(f)
                self._discovered_urls = data.get("discovered_urls", {})
                self._last_page = data.get("last_page", 1)
                logger.info(
                    "checkpoint_loaded",
                    pages_completed=self._last_page,
                    urls_found=len(self._discovered_urls),
                )
            except (json.JSONDecodeError, KeyError):
                self._last_page = 1
                self._discovered_urls = {}
        else:
            self._last_page = 1
            self._discovered_urls = {}

    def _save_checkpoint(self, last_page: int) -> None:
        data = {
            "last_page": last_page,
            "discovered_urls": self._discovered_urls,
            "timestamp": datetime.utcnow().isoformat(),
        }
        with open(self.checkpoint_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.debug("checkpoint_saved", page=last_page)

    def _classify_document(self, title: str, url: str) -> dict[str, Any]:
        title_lower = title.lower()

        doc_type = "unknown"
        for dtype, patterns in DOC_TYPE_PATTERNS.items():
            for pattern in patterns:
                if pattern in title_lower:
                    doc_type = dtype
                    break
        if doc_type == "unknown":
            for keyword in TITLE_KEYWORDS:
                if keyword in title_lower:
                    doc_type = keyword.replace(" ", "_")
                    break

        year = 2024
        year_match = re.search(r"(20[12]\d)", title)
        if year_match:
            year = int(year_match.group(1))
        else:
            year_match = re.search(r"(20[12]\d)", url)
            if year_match:
                year = int(year_match.group(1))

        round_number = 0
        for pattern, rnd in ROUND_PATTERNS:
            if re.search(pattern, title_lower):
                round_number = rnd
                break

        state = "All India"
        for state_name, patterns in STATE_PATTERNS.items():
            for pat in patterns:
                if pat in title_lower:
                    state = state_name
                    break

        return {
            "url": url,
            "year": year,
            "state": state,
            "source_type": "mcc",
            "file_type": "pdf",
            "document_type": doc_type,
            "round_number": round_number,
            "title": title.strip(),
            "crawl_status": "discovered",
        }

    async def _fetch_page(
        self,
        session: aiohttp.ClientSession,
        url: str,
        retries: int = 5,
    ) -> str | None:
        delay = 3 + random.random() * 2
        await asyncio.sleep(delay)

        for attempt in range(retries):
            try:
                async with self.semaphore:
                    timeout = aiohttp.ClientTimeout(total=60)
                    async with session.get(
                        url,
                        headers=self.HEADERS,
                        timeout=timeout,
                        allow_redirects=True,
                        ssl=False,
                    ) as response:
                        if response.status == 200:
                            html = await response.text()
                            logger.info("page_fetched", url=url, attempt=attempt + 1, length=len(html))
                            return html
                        elif response.status == 404:
                            logger.info("page_not_found", url=url, status=response.status)
                            return None
                        else:
                            logger.warning(
                                "page_fetch_bad_status",
                                url=url,
                                status=response.status,
                                attempt=attempt + 1,
                            )
            except asyncio.TimeoutError:
                logger.warning("page_fetch_timeout", url=url, attempt=attempt + 1)
            except aiohttp.ClientError as e:
                logger.warning("page_fetch_error", url=url, attempt=attempt + 1, error=str(e))
            except Exception as e:
                logger.warning("page_fetch_error", url=url, attempt=attempt + 1, error=str(e))

            backoff = min(2**attempt * 2, 60)
            jitter = random.random() * 2
            await asyncio.sleep(backoff + jitter)

        logger.error("page_fetch_exhausted", url=url, retries=retries)
        return None

    def _parse_links_from_html(self, html: str) -> list[dict[str, str]]:
        links = []
        soup = BeautifulSoup(html, "lxml")

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"].strip()
            if not href.lower().endswith(".pdf"):
                continue

            title_text = a_tag.get_text(strip=True)
            if not title_text:
                title_text = Path(href.split("?")[0]).stem.replace("_", " ").replace("-", " ")

            if not href.startswith("http"):
                href = self.BASE_URL + href

            links.append({"title": title_text, "url": href})

        return links

    async def _discover_single_page(
        self,
        session: aiohttp.ClientSession,
        page_num: int,
    ) -> list[dict[str, Any]]:
        if page_num == 1:
            url = self.ARCHIVE_URL
        else:
            url = f"{self.ARCHIVE_URL}page/{page_num}/"

        html = await self._fetch_page(session, url)
        if not html:
            return []

        raw_links = self._parse_links_from_html(html)
        classified = []
        for link in raw_links:
            if link["url"] not in self._discovered_urls:
                info = self._classify_document(link["title"], link["url"])
                self._discovered_urls[link["url"]] = info
                classified.append(info)
            else:
                classified.append(self._discovered_urls[link["url"]])

        logger.info(
            "page_links_extracted",
            page=page_num,
            links_found=len(classified),
            url=url,
        )
        return classified

    def _write_inventory(self) -> Path:
        fieldnames = [
            "url",
            "year",
            "state",
            "source_type",
            "file_type",
            "document_type",
            "round_number",
            "title",
            "crawl_status",
        ]
        self.inventory_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.inventory_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for entry in self._discovered_urls.values():
                writer.writerow(entry)

        logger.info(
            "inventory_written",
            path=str(self.inventory_path),
            count=len(self._discovered_urls),
        )
        return self.inventory_path

    async def discover(self, config_path: str) -> Path:
        """Discover all PDF links from MCC archive pages."""
        max_pages = 50
        start_page = self._last_page

        logger.info("discovery_started", start_page=start_page, max_pages=max_pages)

        connector = aiohttp.TCPConnector(limit=5, ssl=False)
        timeout = aiohttp.ClientTimeout(total=120)

        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=self.HEADERS,
        ) as session:
            prev_url_count = len(self._discovered_urls)
            consecutive_no_new = 0

            for page_num in range(start_page, max_pages + 1):
                try:
                    entries = await self._discover_single_page(session, page_num)

                    new_count = len(self._discovered_urls) - prev_url_count
                    if new_count == 0:
                        consecutive_no_new += 1
                        if consecutive_no_new >= 2:
                            logger.info(
                                "no_new_links_found",
                                stopped_at=page_num,
                                total_unique=len(self._discovered_urls),
                            )
                            break
                    else:
                        consecutive_no_new = 0
                        prev_url_count = len(self._discovered_urls)

                    if not entries:
                        logger.info("no_more_pages", stopped_at=page_num)
                        break

                    self._save_checkpoint(page_num)

                except Exception as e:
                    logger.error("page_discovery_error", page=page_num, error=str(e))
                    self._save_checkpoint(page_num)
                    continue

        inventory_path = self._write_inventory()
        logger.info(
            "discovery_complete",
            total_links=len(self._discovered_urls),
            inventory_path=str(inventory_path),
        )
        return inventory_path
