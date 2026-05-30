"""Parallel PDF extraction - splits manifest into chunks and processes in parallel."""
import asyncio
import csv
import json
import sys
import os
import time
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from src.agents.pdf_extractor import PDFExtractionAgent

def extract_chunk(worker_id: int, chunk_files: list[str], output_dir: Path) -> int:
    """Extract a chunk of PDFs synchronously. Returns count of extracted files."""
    extractor = PDFExtractionAgent(output_dir=output_dir)
    count = 0
    for pdf_path_str in chunk_files:
        pdf_path = Path(pdf_path_str)
        if pdf_path.name in extractor._completed_files:
            continue
        try:
            result = extractor._extract_single_pdf(pdf_path)
            source_file = result.get("source_file", "unknown")
            json_path = extractor.extracted_dir / f"{source_file}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            extractor._completed_files.add(source_file)
            count += 1
            if count % 5 == 0:
                print(f"  Worker {worker_id}: {count} files extracted", flush=True)
        except Exception as e:
            print(f"  Worker {worker_id}: ERROR {pdf_path.name}: {e}", flush=True)
    extractor._save_checkpoint()
    return count


def main():
    worker_id = int(sys.argv[1])
    total_workers = int(sys.argv[2])
    manifest_path = Path(sys.argv[3]) if len(sys.argv) > 3 else PROJECT / "data" / "download_manifest.csv"
    output_dir = Path(sys.argv[4]) if len(sys.argv) > 4 else PROJECT / "data"

    # Read manifest
    all_files = []
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("download_status") == "completed" and row.get("local_path"):
                local_path = Path(row["local_path"])
                if local_path.exists():
                    all_files.append(str(local_path))

    # Split into chunks
    chunk_size = len(all_files) // total_workers
    start = worker_id * chunk_size
    end = start + chunk_size if worker_id < total_workers - 1 else len(all_files)
    chunk = all_files[start:end]

    print(f"Worker {worker_id}: Processing {len(chunk)} files (of {len(all_files)} total)")
    t0 = time.time()
    count = extract_chunk(worker_id, chunk, output_dir)
    elapsed = time.time() - t0
    print(f"Worker {worker_id}: Done - {count} files in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
