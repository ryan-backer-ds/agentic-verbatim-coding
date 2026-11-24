"""Step 1: download, clean and split the NHS Friends and Family Test (FFT) survey.

The raw file is a real survey export. Each row is one open-ended answer with:
  * the closed FFT rating (1 = very good ... 5 = very poor),
  * the question that was asked ("What could we do better?", "FFT Why?", ...),
  * respondent and service metadata,
  * the themes trained human coders assigned, one-hot encoded (~50 codes).

The human themes and sentiment are kept as `human_*` columns. They are the gold
labels the LLM coder is evaluated against later, so they never go into a prompt.

Build the processed dataset with:

    uv run python -m verbatim.data
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
import urllib.request
import zlib
from pathlib import Path

import pandas as pd

from verbatim.config import get_settings

log = logging.getLogger(__name__)

# Pinned to a commit (not `main`) so every run gets byte-identical input.
SOURCE_COMMIT = "1f3ae7601d1738cbd41df1ed37e84f980da0f415"
SOURCE_URL = (
    "https://raw.githubusercontent.com/The-Strategy-Unit/pxtextmining/"
    f"{SOURCE_COMMIT}/datasets/v6framework_230831.csv"
)
SOURCE_SHA256 = "6c90cc740f2c486a17ef89be852f13f8ca434da4e2a06254a43288de7dc39567"
RAW_FILENAME = "nhs_fft_v6framework.csv"

# Raw column -> clean name. Every other column in the file is a codeframe theme.
METADATA_COLUMNS = {
    "Comment ID": "comment_id",
    "Trust": "trust",
    "Respondent ID": "respondent_id",
    "Date": "date",
    "Service type 1": "service",
    "Service type 2": "sub_service",
    "FFT categorical answer": "fft_score",
    "FFT question": "question",
    "FFT answer": "answer",
    "Person identifiable info?": "pii_flag",
    "Comment sentiment": "human_sentiment",
}

# Coders used this code for blank and "Did not answer" rows. They carry no signal.
NO_ANSWER_THEME = "Labelling not possible"

FFT_SCORE_LABELS = {
    1: "Very good",
    2: "Good",
    3: "Neither good nor poor",
    4: "Poor",
    5: "Very poor",
}
# Same idea as NPS promoters / passives / detractors, on the 5-point FFT scale.
FFT_GROUPS = {1: "positive", 2: "positive", 3: "neutral", 4: "negative", 5: "negative"}

OUTPUT_COLUMNS = [
    "comment_id",
    "respondent_id",
    "trust",
    "service",
    "sub_service",
    "date",
    "question",
    "fft_score",
    "fft_label",
    "fft_group",
    "answer",
    "human_sentiment",
    "human_themes",
]

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_UK_PHONE = re.compile(r"(?:\+44\s?|\b0)\d(?:[\s-]?\d){8,9}\b")
_NHS_NUMBER = re.compile(r"\b\d{3}\s?\d{3}\s?\d{4}\b")
_UK_POSTCODE = re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b", re.IGNORECASE)


def download_raw(dest_dir: Path, *, force: bool = False) -> Path:
    """Download the raw CSV once, then verify it against the pinned checksum."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / RAW_FILENAME

    if path.exists() and not force:
        log.info("Using cached raw file %s", path)
    else:
        log.info("Downloading %s", SOURCE_URL)
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
            path.write_bytes(response.read())

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f"Checksum mismatch for {path} (got {digest}). Delete it and re-run.")
    return path


def normalize_text(text: str) -> str:
    """Unicode-normalise and collapse whitespace (non-breaking spaces, stray line breaks)."""
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def scrub_pii(text: str) -> str:
    """Mask obvious identifiers before any text reaches an LLM or a vector store.

    Regexes only catch structured PII (emails, UK phone numbers, NHS numbers,
    postcodes). Names need an NER-based service such as Google Cloud Sensitive
    Data Protection, which comes in the production-hardening step.
    """
    text = _EMAIL.sub("[EMAIL]", text)
    text = _UK_PHONE.sub("[PHONE]", text)
    text = _NHS_NUMBER.sub("[NHS_NUMBER]", text)
    return _UK_POSTCODE.sub("[POSTCODE]", text)


def theme_columns(raw: pd.DataFrame) -> list[str]:
    """Codeframe themes are all columns that are not respondent metadata."""
    return [col for col in raw.columns if col not in METADATA_COLUMNS]


def parse_survey_dates(values: pd.Series) -> pd.Series:
    """Parse the survey dates, which mix UK and US formats.

    Most rows are zero-padded UK dates (30/09/2021). One NEAS export used US
    dates, recognisable by missing zero padding (9/30/2022) or by a "month"
    above 12 (10/17/2021). Padded dates that fit both formats are read as UK,
    like the rest of the padded rows.
    """
    values = values.astype(str).str.strip()
    uk = pd.to_datetime(values, format="%d/%m/%Y", errors="coerce")
    us = pd.to_datetime(values, format="%m/%d/%Y", errors="coerce")

    is_unpadded = ~values.str.fullmatch(r"\d{2}/\d{2}/\d{4}")
    parsed = uk.mask(is_unpadded | uk.isna(), us)

    if parsed.isna().any():
        examples = values[parsed.isna()].unique()[:5].tolist()
        raise ValueError(f"Unparseable survey dates, e.g. {examples}")
    return parsed


def _clean_category(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).map(normalize_text).replace("", pd.NA)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw export into one tidy row per usable open-ended answer."""
    theme_matrix = raw[theme_columns(raw)].fillna(0).astype(bool)

    df = raw[list(METADATA_COLUMNS)].rename(columns=METADATA_COLUMNS)
    df["human_themes"] = [list(theme_matrix.columns[mask]) for mask in theme_matrix.to_numpy()]
    df["answer"] = df["answer"].fillna("").astype(str).map(normalize_text)

    is_answered = df["answer"].ne("") & df["answer"].str.lower().ne("did not answer")
    is_codable = ~theme_matrix[NO_ANSWER_THEME]
    df = df[is_answered & is_codable].drop_duplicates("comment_id").copy()

    df["answer"] = df["answer"].map(scrub_pii)
    for col in ["trust", "service", "sub_service", "question"]:
        df[col] = _clean_category(df[col])
    df["date"] = parse_survey_dates(df["date"])

    # Codes 6 and 7 sit outside the documented 1-5 scale, so they are treated as missing.
    score = pd.to_numeric(df["fft_score"], errors="coerce")
    df["fft_score"] = score.where(score.between(1, 5)).astype("Int64")
    df["fft_label"] = df["fft_score"].map(FFT_SCORE_LABELS)
    df["fft_group"] = df["fft_score"].map(FFT_GROUPS).fillna("unknown")
    df["human_sentiment"] = pd.to_numeric(df["human_sentiment"], errors="coerce").astype("Int64")

    # `pii_flag` is empty in this release, so it is dropped here.
    return df[OUTPUT_COLUMNS].reset_index(drop=True)


def split_by_respondent(df: pd.DataFrame, test_fraction: float, seed: int) -> pd.Series:
    """Assign every row to "train" or "test" by hashing its respondent ID.

    Splitting by respondent rather than by row keeps all answers from one person
    on the same side, so the test set measures performance on unseen people.
    Hashing instead of shuffling keeps assignments stable when new survey waves
    are appended later.
    """
    buckets = (
        df["respondent_id"]
        .astype(str)
        .map(lambda rid: zlib.crc32(f"{seed}:{rid}".encode()) % 10_000)
    )
    is_test = buckets < test_fraction * 10_000
    return is_test.map({True: "test", False: "train"})


def codeframe_summary(themes: list[str], responses: pd.DataFrame) -> list[dict]:
    """The codeframe in its original order, with how often coders used each theme."""
    counts = responses["human_themes"].explode().value_counts()
    return [
        {"theme": theme, "n_responses": int(counts.get(theme, 0))}
        for theme in themes
        if theme != NO_ANSWER_THEME
    ]


def build_dataset() -> pd.DataFrame:
    settings = get_settings()
    raw = pd.read_csv(download_raw(settings.raw_dir))

    responses = clean(raw)
    responses["split"] = split_by_respondent(responses, settings.test_fraction, settings.split_seed)
    codeframe = codeframe_summary(theme_columns(raw), responses)

    out_dir = settings.processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    responses.to_parquet(out_dir / "responses.parquet", index=False)
    (out_dir / "codeframe.json").write_text(json.dumps(codeframe, indent=2), encoding="utf-8")

    log.info("Raw rows: %d -> usable answers: %d", len(raw), len(responses))
    log.info("Respondents: %d", responses["respondent_id"].nunique())
    log.info("Split: %s", responses["split"].value_counts().to_dict())
    log.info("FFT groups: %s", responses["fft_group"].value_counts().to_dict())
    log.info("Codeframe: %d themes, written to %s", len(codeframe), out_dir)
    return responses


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    build_dataset()
