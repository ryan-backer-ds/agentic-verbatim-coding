import asyncio
import hashlib
import shutil
import time

import pytest

from verbatim.utils import RateLimiter, download_file, sha256_of_file


def test_sha256_of_file(tmp_path):
    path = tmp_path / "answers.txt"
    path.write_bytes(b"Staff were lovely")

    assert sha256_of_file(path) == hashlib.sha256(b"Staff were lovely").hexdigest()


def test_download_file_creates_folders_and_leaves_no_partial_file(tmp_path):
    upstream = tmp_path / "upstream.csv"
    upstream.write_bytes(b"id,answer\n1,None\n")
    dest = tmp_path / "raw" / "copy.csv"

    assert download_file(upstream.as_uri(), dest) == dest
    assert dest.read_bytes() == upstream.read_bytes()
    assert not (tmp_path / "raw" / "copy.csv.part").exists()


def test_interrupted_download_leaves_no_partial_file(monkeypatch, tmp_path):
    upstream = tmp_path / "upstream.csv"
    upstream.write_bytes(b"id,answer\n1,None\n")

    def time_out_halfway(response, file):
        file.write(b"id,")
        raise TimeoutError("The read operation timed out")

    monkeypatch.setattr(shutil, "copyfileobj", time_out_halfway)
    dest = tmp_path / "raw" / "copy.csv"

    with pytest.raises(TimeoutError):
        download_file(upstream.as_uri(), dest)

    assert list(dest.parent.iterdir()) == []


def test_rate_limiter_spaces_out_starts():
    limiter = RateLimiter(per_minute=1200)  # one start every 0.05 s

    async def start_three():
        started = []
        for _ in range(3):
            await limiter.wait()
            started.append(time.monotonic())
        return started

    started = asyncio.run(start_three())

    assert started[2] - started[0] >= 0.09


def test_rate_limiter_without_a_limit_never_waits():
    limiter = RateLimiter(per_minute=None)
    started = time.monotonic()
    asyncio.run(limiter.wait())
    assert time.monotonic() - started < 0.05
