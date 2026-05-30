from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Optional

logger = logging.getLogger("rate_limiter")


class RateLimiter:
    def __init__(
        self,
        min_delay: float = 3.0,
        max_delay: float = 6.0,
        max_retries: int = 5,
    ) -> None:
        self._min_delay = min_delay
        self._max_delay = max_delay
        self._max_retries = max_retries
        self._last_hit: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._backoff_factors: dict[str, float] = {}

    def _get_lock(self, domain: str) -> asyncio.Lock:
        if domain not in self._locks:
            self._locks[domain] = asyncio.Lock()
        return self._locks[domain]

    def _compute_delay(self, domain: str) -> float:
        base_delay = random.uniform(self._min_delay, self._max_delay)
        backoff = self._backoff_factors.get(domain, 0.0)
        return base_delay + backoff

    async def acquire(self, domain: str) -> None:
        lock = self._get_lock(domain)
        async with lock:
            now = time.monotonic()
            last = self._last_hit.get(domain, 0.0)
            delay = self._compute_delay(domain)
            elapsed = now - last
            if elapsed < delay:
                wait_time = delay - elapsed
                logger.debug("Rate limiting %s: waiting %.2fs", domain, wait_time)
                await asyncio.sleep(wait_time)
            self._last_hit[domain] = time.monotonic()

    def record_success(self, domain: str) -> None:
        self._backoff_factors[domain] = 0.0

    def record_rate_limit(self, domain: str, retry_after: Optional[float] = None) -> None:
        current_backoff = self._backoff_factors.get(domain, 0.0)
        if retry_after is not None:
            new_backoff = retry_after
        else:
            new_backoff = min(current_backoff * 2 + random.uniform(1.0, 3.0), 120.0)
        self._backoff_factors[domain] = new_backoff
        logger.warning(
            "Rate limit hit for %s, backoff increased to %.1fs",
            domain, new_backoff,
        )

    def record_server_error(self, domain: str) -> None:
        current_backoff = self._backoff_factors.get(domain, 0.0)
        new_backoff = min(current_backoff + random.uniform(2.0, 5.0), 60.0)
        self._backoff_factors[domain] = new_backoff
        logger.warning(
            "Server error for %s, backoff increased to %.1fs",
            domain, new_backoff,
        )

    async def execute_with_retry(self, domain: str, coro):
        last_error: Optional[Exception] = None
        for attempt in range(1, self._max_retries + 1):
            await self.acquire(domain)
            try:
                result = await coro
                self.record_success(domain)
                return result
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                if "429" in error_str or "too many requests" in error_str:
                    self.record_rate_limit(domain)
                elif "503" in error_str or "service unavailable" in error_str:
                    self.record_server_error(domain)
                else:
                    backoff = min(2 ** attempt * random.uniform(0.5, 1.5), 30.0)
                    await asyncio.sleep(backoff)

                logger.warning(
                    "Attempt %d/%d failed for %s: %s",
                    attempt, self._max_retries, domain, e,
                )

        raise last_error or Exception(f"All {self._max_retries} attempts failed for {domain}")

    def reset(self, domain: Optional[str] = None) -> None:
        if domain:
            self._backoff_factors.pop(domain, None)
            self._last_hit.pop(domain, None)
        else:
            self._backoff_factors.clear()
            self._last_hit.clear()
