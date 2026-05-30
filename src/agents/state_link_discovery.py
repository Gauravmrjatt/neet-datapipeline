"""
State Portal Link Discovery Agent — crawls state counselling websites to discover PDF links.
Supports: Maharashtra, Tamil Nadu, West Bengal, Gujarat, Karnataka.
"""

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
import yaml

logger = structlog.get_logger(__name__)

# Patterns to classify state counselling documents
DOC_TYPE_PATTERNS = {
    "seat_matrix": ["seat matrix", "seat allotment matrix", "seat position"],
    "allotment": ["allotment", "allotment list", "seat allotment list", "provisional allotment"],
    "result": ["final result", "result list", "merit list", "select list"],
    "cutoff": ["cutoff", "cut off", "cutoff marks", "cut-off"],
    "admitted": ["admitted", "admitted list", "joined"],
    "vacancy": ["vacancy", "vacancy position", "vacancy list"],
    "counselling_schedule": ["schedule", "counselling schedule", "counselling program"],
    "prospectus": ["prospectus", "information brochure"],
}

ROUND_PATTERNS = [
    (r"round\s*[\-:\s]*1|first round|r1|cap\s*round\s*1|cap\s*1", 1),
    (r"round\s*[\-:\s]*2|second round|r2|cap\s*round\s*2|cap\s*2", 2),
    (r"round\s*[\-:\s]*3|third round|r3|mop\s*up|cap\s*round\s*3|cap\s*3", 3),
    (r"stray|vacancy round|stray round|stray vacancy", 4),
    (r"special stray|special round|mop\s*up\s*round", 5),
]

# Course detection
COURSE_PATTERNS = [
    (r"\bmbbs\b", "MBBS"),
    (r"\bbds\b", "BDS"),
    (r"\bmd\b|\bms\b|\bdiploma\b|\bpg\b", "PG"),
    (r"\bsuper\s*speciality\b|\bss\b", "Super Speciality"),
    (r"\bnursing\b", "Nursing"),
    (r"\bphysiotherapy\b|\bpt\b", "Physiotherapy"),
]


def _detect_course(title: str) -> str:
    title_lower = title.lower()
    for pattern, course in COURSE_PATTERNS:
        if re.search(pattern, title_lower):
            return course
    return "UG"


def _detect_round(title: str) -> int:
    title_lower = title.lower()
    for pattern, rnd in ROUND_PATTERNS:
        if re.search(pattern, title_lower):
            return rnd
    return 0


def _detect_doc_type(title: str) -> str:
    title_lower = title.lower()
    for dtype, patterns in DOC_TYPE_PATTERNS.items():
        for pattern in patterns:
            if pattern in title_lower:
                return dtype
    return "unknown"


def _detect_year(title: str, url: str) -> int:
    year_match = re.search(r"(20[12]\d)", title)
    if year_match:
        return int(year_match.group(1))
    year_match = re.search(r"(20[12]\d)", url)
    if year_match:
        return int(year_match.group(1))
    return 0


class StatePortalDiscoveryAgent:
    """Crawls state counselling portals to discover and inventory PDF document links."""

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    CHECKPOINT_FILE = "state_portal_checkpoint.json"

    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir or Path("data")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.inventory_path = self.output_dir / "state_links_inventory.csv"
        self.checkpoint_path = self.output_dir / self.CHECKPOINT_FILE
        self.semaphore = asyncio.Semaphore(3)
        self._discovered_urls: dict[str, dict[str, Any]] = {}
        self._load_config()
        self._load_checkpoint()

    def _load_config(self) -> None:
        config_path = Path("config/state_sources.yaml")
        if config_path.exists():
            with open(config_path) as f:
                self.config = yaml.safe_load(f).get("state_portals", {})
        else:
            self.config = {}

    def _load_checkpoint(self) -> None:
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path) as f:
                    data = json.load(f)
                self._discovered_urls = data.get("discovered_urls", {})
                self._last_state = data.get("last_state", "")
                self._last_page = data.get("last_page", 1)
                logger.info(
                    "checkpoint_loaded",
                    last_state=self._last_state,
                    urls_found=len(self._discovered_urls),
                )
            except (json.JSONDecodeError, KeyError):
                self._last_state = ""
                self._last_page = 1
                self._discovered_urls = {}
        else:
            self._last_state = ""
            self._last_page = 1
            self._discovered_urls = {}

    def _save_checkpoint(self, state: str, page: int) -> None:
        data = {
            "last_state": state,
            "last_page": page,
            "discovered_urls": self._discovered_urls,
            "timestamp": datetime.utcnow().isoformat(),
        }
        with open(self.checkpoint_path, "w") as f:
            json.dump(data, f, indent=2)

    async def _fetch_page(
        self, session: aiohttp.ClientSession, url: str, retries: int = 5
    ) -> str | None:
        delay = 3 + random.random() * 2
        await asyncio.sleep(delay)

        for attempt in range(retries):
            try:
                async with self.semaphore:
                    timeout = aiohttp.ClientTimeout(total=60)
                    async with session.get(
                        url, headers=self.HEADERS, timeout=timeout,
                        allow_redirects=True, ssl=False,
                    ) as response:
                        if response.status == 200:
                            html = await response.text()
                            logger.info("page_fetched", url=url, length=len(html))
                            return html
                        elif response.status == 404:
                            return None
                        else:
                            logger.warning("page_bad_status", url=url, status=response.status)
            except asyncio.TimeoutError:
                logger.warning("page_timeout", url=url, attempt=attempt + 1)
            except aiohttp.ClientError as e:
                logger.warning("page_error", url=url, error=str(e)[:100])
            except Exception as e:
                logger.warning("page_error", url=url, error=str(e)[:100])

            backoff = min(2**attempt * 2, 60)
            await asyncio.sleep(backoff + random.random() * 2)

        logger.error("page_exhausted", url=url)
        return None

    def _parse_pdf_links(self, html: str, base_url: str, state_name: str) -> list[dict[str, str]]:
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
                href = base_url.rstrip("/") + "/" + href.lstrip("/")

            links.append({"title": title_text, "url": href, "state": state_name})

        return links

    def _classify_link(self, link: dict[str, str], state_config: dict) -> dict[str, Any]:
        title = link["title"]
        url = link["url"]

        return {
            "url": url,
            "year": _detect_year(title, url),
            "state": state_config.get("state_name", link["state"]),
            "source_type": "state_portal",
            "file_type": "pdf",
            "document_type": _detect_doc_type(title),
            "round_number": _detect_round(title),
            "course": _detect_course(title),
            "title": title.strip(),
            "portal": state_config.get("name", link["state"]),
            "crawl_status": "discovered",
        }

    async def _crawl_maharashtra(self, session: aiohttp.ClientSession) -> list[dict[str, Any]]:
        """Crawl Maharashtra CET Cell — WordPress, direct PDF links."""
        config = self.config.get("maharashtra", {})
        base_url = config.get("base_url", "https://cetcell.mahacet.org")
        results = []

        # Crawl notifications page and paginated archives
        urls_to_try = [
            f"{base_url}/notifications/",
            f"{base_url}/cet-2/",
            f"{base_url}/cap/",
        ]

        for page_url in urls_to_try:
            html = await self._fetch_page(session, page_url)
            if not html:
                continue

            links = self._parse_pdf_links(html, base_url, "Maharashtra")
            for link in links:
                if link["url"] not in self._discovered_urls:
                    info = self._classify_link(link, config)
                    self._discovered_urls[link["url"]] = info
                    results.append(info)

        # Try paginated notifications (up to page 10)
        for page_num in range(2, 11):
            html = await self._fetch_page(session, f"{base_url}/notifications/page/{page_num}/")
            if not html:
                break
            links = self._parse_pdf_links(html, base_url, "Maharashtra")
            new_count = 0
            for link in links:
                if link["url"] not in self._discovered_urls:
                    info = self._classify_link(link, config)
                    self._discovered_urls[link["url"]] = info
                    results.append(info)
                    new_count += 1
            if new_count == 0:
                break

        logger.info("maharashtra_complete", pdfs_found=len(results))
        return results

    async def _crawl_tamil_nadu(self, session: aiohttp.ClientSession) -> list[dict[str, Any]]:
        """Crawl TN Medical Selection — ASP.NET, PDFs in notifications."""
        config = self.config.get("tamil_nadu", {})
        base_url = config.get("base_url", "https://tnmedicalselection.org")
        results = []

        html = await self._fetch_page(session, base_url)
        if not html:
            return results

        links = self._parse_pdf_links(html, base_url, "Tamil Nadu")
        for link in links:
            if link["url"] not in self._discovered_urls:
                info = self._classify_link(link, config)
                self._discovered_urls[link["url"]] = info
                results.append(info)

        # Also try course-specific pages
        for course_page in ["ug-mbbs-bds", "counselling-schedule"]:
            html = await self._fetch_page(session, f"{base_url}/{course_page}")
            if not html:
                continue
            links = self._parse_pdf_links(html, base_url, "Tamil Nadu")
            for link in links:
                if link["url"] not in self._discovered_urls:
                    info = self._classify_link(link, config)
                    self._discovered_urls[link["url"]] = info
                    results.append(info)

        logger.info("tamil_nadu_complete", pdfs_found=len(results))
        return results

    async def _crawl_west_bengal(self, session: aiohttp.ClientSession) -> list[dict[str, Any]]:
        """Crawl WBMCC — WordPress GOI theme."""
        config = self.config.get("west_bengal", {})
        base_url = config.get("base_url", "https://wbmcc.nic.in")
        results = []

        urls_to_try = [
            base_url,
            f"{base_url}/notifications/",
            f"{base_url}/downloads/",
        ]

        for page_url in urls_to_try:
            html = await self._fetch_page(session, page_url)
            if not html:
                continue
            links = self._parse_pdf_links(html, base_url, "West Bengal")
            for link in links:
                if link["url"] not in self._discovered_urls:
                    info = self._classify_link(link, config)
                    self._discovered_urls[link["url"]] = info
                    results.append(info)

        logger.info("west_bengal_complete", pdfs_found=len(results))
        return results

    async def _crawl_gujarat(self, session: aiohttp.ClientSession) -> list[dict[str, Any]]:
        """Crawl ACPMEC Gujarat — ASP.NET, UG home page."""
        config = self.config.get("gujarat", {})
        base_url = config.get("base_url", "https://medadmgujarat.org")
        results = []

        urls_to_try = [
            f"{base_url}/ug/home.aspx",
            f"{base_url}/ug/UGAIQ_Home.aspx",
        ]

        for page_url in urls_to_try:
            html = await self._fetch_page(session, page_url)
            if not html:
                continue
            links = self._parse_pdf_links(html, base_url, "Gujarat")
            for link in links:
                if link["url"] not in self._discovered_urls:
                    info = self._classify_link(link, config)
                    self._discovered_urls[link["url"]] = info
                    results.append(info)

        logger.info("gujarat_complete", pdfs_found=len(results))
        return results

    async def _crawl_karnataka(self, session: aiohttp.ClientSession) -> list[dict[str, Any]]:
        """Crawl KEA Karnataka — ASP.NET WebForms."""
        config = self.config.get("karnataka", {})
        base_url = config.get("base_url", "https://cetonline.karnataka.gov.in/kea")
        results = []

        # KEA has year-specific pages
        for year in range(2020, 2026):
            page_url = f"{base_url}/ugneet{year}.aspx"
            html = await self._fetch_page(session, page_url)
            if not html:
                continue
            links = self._parse_pdf_links(html, base_url, "Karnataka")
            for link in links:
                if link["url"] not in self._discovered_urls:
                    info = self._classify_link(link, config)
                    self._discovered_urls[link["url"]] = info
                    results.append(info)

        logger.info("karnataka_complete", pdfs_found=len(results))
        return results

    def _write_inventory(self) -> Path:
        fieldnames = [
            "url", "year", "state", "source_type", "file_type",
            "document_type", "round_number", "course", "title",
            "portal", "crawl_status",
        ]
        self.inventory_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.inventory_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for entry in self._discovered_urls.values():
                writer.writerow(entry)

        logger.info("inventory_written", path=str(self.inventory_path), count=len(self._discovered_urls))
        return self.inventory_path

    async def discover(self) -> Path:
        """Discover PDF links from all state portals."""
        state_order = ["maharashtra", "tamil_nadu", "west_bengal", "gujarat", "karnataka"]
        crawlers = {
            "maharashtra": self._crawl_maharashtra,
            "tamil_nadu": self._crawl_tamil_nadu,
            "west_bengal": self._crawl_west_bengal,
            "gujarat": self._crawl_gujarat,
            "karnataka": self._crawl_karnataka,
        }

        connector = aiohttp.TCPConnector(limit=5, ssl=False)
        timeout = aiohttp.ClientTimeout(total=120)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            for state in state_order:
                if state in crawlers:
                    logger.info("crawling_state", state=state)
                    try:
                        await crawlers[state](session)
                    except Exception as e:
                        logger.error("crawl_failed", state=state, error=str(e)[:200])
                    self._save_checkpoint(state, 0)

        inventory_path = self._write_inventory()
        logger.info(
            "discovery_complete",
            total_links=len(self._discovered_urls),
            inventory_path=str(inventory_path),
        )
        return inventory_path
