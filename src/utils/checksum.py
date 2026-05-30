from __future__ import annotations

import hashlib
from pathlib import Path


def compute_sha256(file_path: str | Path) -> str:
    file_path = Path(file_path)
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def compute_md5(file_path: str | Path) -> str:
    file_path = Path(file_path)
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def verify_checksum(
    file_path: str | Path,
    expected_hash: str,
    algorithm: str = "sha256",
) -> bool:
    algorithm = algorithm.lower()
    if algorithm == "sha256":
        actual = compute_sha256(file_path)
    elif algorithm == "md5":
        actual = compute_md5(file_path)
    else:
        raise ValueError(f"Unsupported checksum algorithm: {algorithm}")
    return actual.lower() == expected_hash.lower()
