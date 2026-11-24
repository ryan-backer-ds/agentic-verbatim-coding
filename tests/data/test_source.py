import pandas as pd
import pytest

from verbatim.data.source import DatasetSource, load_raw
from verbatim.exceptions import DataIntegrityError, DownloadError
from verbatim.utils import sha256_of_file


@pytest.fixture
def upstream(tmp_path):
    """A local file standing in for the remote CSV."""
    path = tmp_path / "upstream.csv"
    path.write_text("Comment ID,FFT answer\nc1,None\n", encoding="utf-8")
    return path


def test_fetch_downloads_once_then_uses_the_cache(upstream, tmp_path):
    source = DatasetSource(upstream.as_uri(), sha256_of_file(upstream), "raw.csv")

    path = source.fetch(tmp_path / "raw")
    upstream.unlink()  # a second fetch must not need the source any more

    assert source.fetch(tmp_path / "raw") == path
    assert path.read_text(encoding="utf-8").startswith("Comment ID")


def test_fetch_rejects_a_file_that_changed(upstream, tmp_path):
    source = DatasetSource(upstream.as_uri(), "0" * 64, "raw.csv")

    with pytest.raises(DataIntegrityError, match="Checksum mismatch"):
        source.fetch(tmp_path / "raw")


def test_fetch_turns_a_failed_download_into_an_expected_error(tmp_path):
    source = DatasetSource((tmp_path / "gone.csv").as_uri(), "0" * 64, "raw.csv")

    with pytest.raises(DownloadError, match="Could not download"):
        source.fetch(tmp_path / "raw")


def test_load_raw_keeps_none_as_an_answer(tmp_path):
    csv = tmp_path / "raw.csv"
    csv.write_text("Comment ID,FFT answer\nc1,None\nc2,\nc3,NA\n", encoding="utf-8")

    answers = load_raw(csv)["FFT answer"]

    assert answers[0] == "None"
    assert pd.isna(answers[1])
    assert answers[2] == "NA"
