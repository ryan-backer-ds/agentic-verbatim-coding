"""Generic file helpers that know nothing about surveys or models."""

import asyncio
import hashlib
import shutil
import time
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


class RateLimiter:
    """Spaces out async calls so at most `per_minute` start in any minute.

    Concurrency limits how many calls run at once; this limits how many start,
    which is what API quotas count. `None` means no limit.
    """

    def __init__(self, per_minute: float | None) -> None:
        self._interval = 60 / per_minute if per_minute else 0.0
        self._next_start = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        if not self._interval:
            return
        async with self._lock:
            now = time.monotonic()
            delay = self._next_start - now
            self._next_start = max(now, self._next_start) + self._interval
        if delay > 0:
            await asyncio.sleep(delay)
