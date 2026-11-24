"""Generic file helpers that know nothing about surveys or models."""

import hashlib
import shutil
import urllib.request
from pathlib import Path


def sha256_of_file(path: Path) -> str:
    """SHA-256 fingerprint of a file, read in chunks so big files never fill memory."""
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def download_file(url: str, dest: Path, *, timeout: float = 60) -> Path:
    """Download `url` to `dest` and return `dest`.

    The bytes go to a temporary `.part` file that is renamed only once the
    download has finished, so an interrupted run never leaves a truncated file
    that looks complete. A failed download deletes the `.part` file and re-raises.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(f"{dest.name}.part")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response, partial.open("wb") as file:
            shutil.copyfileobj(response, file)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(dest)
    return dest
