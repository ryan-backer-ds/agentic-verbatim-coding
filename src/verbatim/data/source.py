"""Where the raw survey comes from, pinned to one exact version of the file."""

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from verbatim.exceptions import DataIntegrityError, DownloadError
from verbatim.utils import download_file, sha256_of_file

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetSource:
    """A remote file, pinned to one version by its SHA-256 fingerprint."""

    url: str
    sha256: str
    filename: str

    def fetch(self, dest_dir: Path, *, force: bool = False) -> Path:
        """Download the file once, verify it on every run, and return its local path."""
        path = dest_dir / self.filename
        if force or not path.exists():
            logger.info("Downloading %s", self.url)
            try:
                download_file(self.url, path)
            except OSError as exc:  # no network, a timeout, a full disk...
                raise DownloadError(f"Could not download {self.url}: {exc}") from exc
        else:
            logger.info("Using cached %s", path)

        actual = sha256_of_file(path)
        if actual != self.sha256:
            raise DataIntegrityError(
                f"Checksum mismatch for {path}: expected {self.sha256}, got {actual}. "
                "Delete the file and run again."
            )
        return path


# The Strategy Unit's labelled NHS Friends and Family Test export (MIT licence),
# pinned to a commit rather than `main` so every run reads byte-identical input.
_NHS_FFT_COMMIT = "1f3ae7601d1738cbd41df1ed37e84f980da0f415"
NHS_FFT_SOURCE = DatasetSource(
    url=(
        "https://raw.githubusercontent.com/The-Strategy-Unit/pxtextmining/"
        f"{_NHS_FFT_COMMIT}/datasets/v6framework_230831.csv"
    ),
    sha256="6c90cc740f2c486a17ef89be852f13f8ca434da4e2a06254a43288de7dc39567",
    filename="nhs_fft_v6framework.csv",
)


def load_raw(path: Path) -> pd.DataFrame:
    """Read the raw CSV, treating only empty cells as missing.

    By default pandas turns strings like "None", "NA" and "null" into NaN. In a
    survey, "None" in reply to "What could we do better?" is a real answer.
    """
    return pd.read_csv(path, keep_default_na=False, na_values=[""])
