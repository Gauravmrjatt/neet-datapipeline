"""
Fast parallel downloader for state portal PDFs.
Bypasses pipeline overhead, uses 16 concurrent workers.
"""

import asyncio
import csv
import hashlib
import random
import time
from pathlib import Path

import aiohttp

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/pdf,*/*",
}

SEM = asyncio.Semaphore(16)
RAW_DIR = Path("data/raw")
MANIFEST = Path("data/state_download_manifest.csv")


async def download_one(session: aiohttp.ClientSession, url: str, dest: Path, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            async with SEM:
                timeout = aiohttp.ClientTimeout(total=30)
                async with session.get(url, headers=HEADERS, timeout=timeout, ssl=False) as resp:
                    if resp.status == 200:
                        data = await resp.read()
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.write_bytes(data)
                        sha = hashlib.sha256(data).hexdigest()
                        return {"url": url, "path": str(dest), "size": len(data), "sha256": sha, "status": "ok"}
                    elif resp.status == 404:
                        return {"url": url, "path": "", "size": 0, "sha256": "", "status": "404"}
        except Exception as e:
            if attempt < retries - 1:
                await asyncio.sleep(1 + attempt)
    return {"url": url, "path": "", "size": 0, "sha256": "", "status": "failed"}


def sanitize_filename(title: str, max_len: int = 120) -> str:
    name = "".join(c if c.isalnum() or c in " -_." else "_" for c in title)
    return name[:max_len].strip()


async def main():
    # Read state inventory
    inv = Path("data/state_links_inventory.csv")
    with open(inv) as f:
        reader = csv.DictReader(f)
        entries = [r for r in reader if r.get("file_type") == "pdf"]

    print(f"Total state portal PDFs to download: {len(entries)}")

    # Build download tasks
    tasks = []
    for entry in entries:
        url = entry["url"]
        state = entry.get("state", "unknown").replace(" ", "_")
        year = entry.get("year", "0")
        title = sanitize_filename(entry.get("title", "unnamed"))
        filename = f"{title}.pdf"
        dest = RAW_DIR / "state_portals" / state / year / filename

        if dest.exists():
            continue  # skip already downloaded

        tasks.append((url, dest))

    print(f"New downloads needed: {len(tasks)}")

    connector = aiohttp.TCPConnector(limit=20, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        batch_size = 50
        results = []
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_results = await asyncio.gather(
                *[download_one(session, url, dest) for url, dest in batch]
            )
            results.extend(batch_results)
            ok = sum(1 for r in results if r["status"] == "ok")
            print(f"  [{i + len(batch)}/{len(tasks)}] ok={ok}")

    # Write manifest
    with open(MANIFEST, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "path", "size", "sha256", "status"])
        writer.writeheader()
        writer.writerows(results)

    ok = sum(1 for r in results if r["status"] == "ok")
    failed = sum(1 for r in results if r["status"] == "failed")
    print(f"\nDone: {ok} downloaded, {failed} failed, manifest at {MANIFEST}")


if __name__ == "__main__":
    asyncio.run(main())
