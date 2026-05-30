from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, Optional

import aiohttp

logger = logging.getLogger("http_client")

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
]


def _random_user_agent() -> str:
    return random.choice(USER_AGENTS)


class AsyncHTTPClient:
    def __init__(
        self,
        max_concurrent: int = 8,
        default_timeout: int = 120,
        max_retries: int = 5,
    ) -> None:
        self._max_concurrent = max_concurrent
        self._default_timeout = default_timeout
        self._max_retries = max_retries
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._session: Optional[aiohttp.ClientSession] = None
        self._domain_last_hit: dict[str, float] = {}
        self._domain_locks: dict[str, asyncio.Lock] = {}

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=self._max_concurrent * 2,
                ttl_dns_cache=300,
                enable_cleanup_closed=True,
            )
            timeout = aiohttp.ClientTimeout(total=self._default_timeout)
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={"User-Agent": _random_user_agent()},
            )
        return self._session

    def _get_domain_lock(self, domain: str) -> asyncio.Lock:
        if domain not in self._domain_locks:
            self._domain_locks[domain] = asyncio.Lock()
        return self._domain_locks[domain]

    def _extract_domain(self, url: str) -> str:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc or parsed.hostname or url

    async def _rate_limit(self, domain: str) -> None:
        import time
        lock = self._get_domain_lock(domain)
        async with lock:
            now = time.monotonic()
            last_hit = self._domain_last_hit.get(domain, 0.0)
            delay = random.uniform(0.5, 1.5)
            elapsed = now - last_hit
            if elapsed < delay:
                await asyncio.sleep(delay - elapsed)
            self._domain_last_hit[domain] = time.monotonic()

    async def get(
        self,
        url: str,
        headers: Optional[dict[str, str]] = None,
        timeout: Optional[int] = None,
    ) -> dict[str, Any]:
        domain = self._extract_domain(url)
        last_error: Optional[Exception] = None

        for attempt in range(1, self._max_retries + 1):
            async with self._semaphore:
                await self._rate_limit(domain)
                try:
                    session = await self._get_session()
                    effective_headers = {"User-Agent": _random_user_agent()}
                    if headers:
                        effective_headers.update(headers)

                    request_timeout = (
                        aiohttp.ClientTimeout(total=timeout)
                        if timeout
                        else None
                    )

                    async with session.get(
                        url,
                        headers=effective_headers,
                        timeout=request_timeout,
                        ssl=False,
                    ) as response:
                        if response.status == 429 or response.status == 503:
                            backoff = min(2 ** attempt * random.uniform(1.0, 2.0), 60.0)
                            logger.warning(
                                "Rate limited on %s (status=%d), backing off %.1fs (attempt %d/%d)",
                                url, response.status, backoff, attempt, self._max_retries,
                            )
                            await asyncio.sleep(backoff)
                            continue

                        response.raise_for_status()
                        content_type = response.headers.get("Content-Type", "")
                        if "json" in content_type:
                            data = await response.json()
                        else:
                            text = await response.text()
                            data = {"text": text, "status": response.status}

                        return {
                            "status": response.status,
                            "headers": dict(response.headers),
                            "data": data,
                            "url": str(response.url),
                        }

                except aiohttp.ClientError as e:
                    last_error = e
                    backoff = min(2 ** attempt * random.uniform(0.5, 1.5), 30.0)
                    logger.warning(
                        "Request failed for %s: %s, retrying in %.1fs (attempt %d/%d)",
                        url, e, backoff, attempt, self._max_retries,
                    )
                    await asyncio.sleep(backoff)

                except asyncio.TimeoutError as e:
                    last_error = e
                    backoff = min(2 ** attempt * random.uniform(1.0, 2.0), 60.0)
                    logger.warning(
                        "Timeout for %s, retrying in %.1fs (attempt %d/%d)",
                        url, backoff, attempt, self._max_retries,
                    )
                    await asyncio.sleep(backoff)

        raise last_error or Exception(f"All {self._max_retries} attempts failed for {url}")

    async def get_pdf(
        self,
        url: str,
        save_path: str,
        chunk_size: int = 8192,
    ) -> dict[str, Any]:
        domain = self._extract_domain(url)
        last_error: Optional[Exception] = None

        for attempt in range(1, self._max_retries + 1):
            async with self._semaphore:
                await self._rate_limit(domain)
                try:
                    session = await self._get_session()
                    headers = {"User-Agent": _random_user_agent()}

                    async with session.get(url, headers=headers, ssl=False) as response:
                        if response.status == 429 or response.status == 503:
                            backoff = min(2 ** attempt * random.uniform(1.0, 2.0), 60.0)
                            logger.warning(
                                "Rate limited on PDF %s (status=%d), backing off %.1fs",
                                url, response.status, backoff,
                            )
                            await asyncio.sleep(backoff)
                            continue

                        response.raise_for_status()

                        total_size = 0
                        sha256_hash = __import__("hashlib").sha256()

                        with open(save_path, "wb") as f:
                            async for chunk in response.content.iter_chunked(chunk_size):
                                f.write(chunk)
                                sha256_hash.update(chunk)
                                total_size += len(chunk)

                        return {
                            "status": response.status,
                            "local_path": save_path,
                            "size_bytes": total_size,
                            "sha256": sha256_hash.hexdigest(),
                            "content_type": response.headers.get("Content-Type", ""),
                        }

                except aiohttp.ClientError as e:
                    last_error = e
                    backoff = min(2 ** attempt * random.uniform(0.5, 1.5), 30.0)
                    logger.warning(
                        "PDF download failed for %s: %s, retrying in %.1fs (attempt %d/%d)",
                        url, e, backoff, attempt, self._max_retries,
                    )
                    await asyncio.sleep(backoff)

                except asyncio.TimeoutError as e:
                    last_error = e
                    backoff = min(2 ** attempt * random.uniform(1.0, 2.0), 60.0)
                    logger.warning(
                        "PDF download timeout for %s, retrying in %.1fs (attempt %d/%d)",
                        url, backoff, attempt, self._max_retries,
                    )
                    await asyncio.sleep(backoff)

        raise last_error or Exception(f"All {self._max_retries} attempts failed for PDF {url}")

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> "AsyncHTTPClient":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
