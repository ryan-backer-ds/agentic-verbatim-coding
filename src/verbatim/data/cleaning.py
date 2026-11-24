"""Pure transformations from the raw survey export to tidy, PII-scrubbed answers.

Nothing in this module reads files or calls services: each function takes data
and returns new data, so every rule can be tested on its own with a tiny table.
"""

import re
import unicodedata

import pandas as pd

from verbatim.data.schema import (
    CATEGORY_COLUMNS,
    CLEAN_COLUMNS,
    FFT_SCORE_GROUPS,
    FFT_SCORE_LABELS,
    NO_ANSWER_THEME,
    RAW_TO_CLEAN_COLUMNS,
)
from verbatim.exceptions import DataIntegrityError

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# +44 (optionally with "(0)") or a leading 0, possibly in brackets, then 9-10 more digits.
_UK_PHONE = re.compile(r"(?:\+44\s?(?:\(0\)\s?)?|\(?\b0)\d(?:[\s()-]{0,2}\d){8,9}\b")
_NHS_NUMBER = re.compile(r"\b\d{3}[\s-]?\d{3}[\s-]?\d{4}\b")
# The inward part of a postcode ("4LP" in "NE1 4LP") never uses C, I, K, M, O or V,
# which keeps times like "A1 2pm" out. Ordinals ("the A1 2nd exit") are ruled out too.
_UK_POSTCODE = re.compile(
    r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d(?!(?:st|nd|rd|th)\b)[ABD-HJLNP-UW-Z]{2}\b", re.IGNORECASE
)


def normalize_text(text: str) -> str:
    """Unicode-normalize and collapse whitespace (non-breaking spaces, stray line breaks)."""
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def scrub_pii(text: str) -> str:
    """Mask obvious identifiers before any text reaches an LLM or a vector store.

    Regexes only catch structured PII (emails, UK phone numbers, NHS numbers,
    postcodes). Names need an NER-based service such as Google Cloud Sensitive
    Data Protection, which comes in the production-hardening step. On the NHS
    data this masks nothing: the publisher had already redacted it.
    """
    text = _EMAIL.sub("[EMAIL]", text)
    text = _UK_PHONE.sub("[PHONE]", text)
    text = _NHS_NUMBER.sub("[NHS_NUMBER]", text)
    return _UK_POSTCODE.sub("[POSTCODE]", text)


def parse_survey_dates(values: pd.Series, exports: pd.Series) -> pd.Series:
    """Parse dates that are UK (DD/MM/YYYY) in some exports and US (MM/DD/YYYY) in others.

    A date like 03/10/2021 reads both ways, so the format is decided once per
    export, from its dates that read only one way: 30/09/2021 can only be UK and
    10/17/2021 only US. An export with no such date is read as UK, the survey's
    home format. One that has both raises, because no single rule fits it.
    `exports` labels the export each row came from.
    """
    values = values.astype(str).str.strip()
    uk = pd.to_datetime(values, format="%d/%m/%Y", errors="coerce")
    us = pd.to_datetime(values, format="%m/%d/%Y", errors="coerce")

    is_us_export = (uk.isna() & us.notna()).groupby(exports).transform("any")
    is_uk_export = (us.isna() & uk.notna()).groupby(exports).transform("any")
    if (is_us_export & is_uk_export).any():
        examples = values[is_us_export & is_uk_export].unique()[:5].tolist()
        raise DataIntegrityError(f"An export mixes UK and US dates, e.g. {examples}")

    parsed = uk.mask(is_us_export, us)
    if parsed.isna().any():
        examples = values[parsed.isna()].unique()[:5].tolist()
        raise DataIntegrityError(f"Unparseable survey dates, e.g. {examples}")
    return parsed


def theme_columns(raw: pd.DataFrame) -> list[str]:
    """Codeframe themes are all raw columns that are not respondent metadata."""
    return [column for column in raw.columns if column not in RAW_TO_CLEAN_COLUMNS]


def clean_responses(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw export into one tidy row per usable open-ended answer."""
    themes = theme_columns(raw)
    _check_raw_shape(raw, themes)
    theme_matrix = raw[themes].fillna(0).astype(bool)

    df = raw[list(RAW_TO_CLEAN_COLUMNS)].rename(columns=RAW_TO_CLEAN_COLUMNS)
    df["human_themes"] = [list(theme_matrix.columns[row]) for row in theme_matrix.to_numpy()]
    # Dates first, while every row of an export is still there as evidence of its format.
    df["date"] = parse_survey_dates(df["date"], exports=_export_runs(df))
    df["answer"] = _normalized_text(df["answer"])

    is_answered = df["answer"].ne("") & df["answer"].str.lower().ne("did not answer")
    is_codable = ~theme_matrix[NO_ANSWER_THEME]
    df = df[is_answered & is_codable].copy()

    df["answer"] = df["answer"].map(scrub_pii)
    for column in CATEGORY_COLUMNS:
        df[column] = _normalized_text(df[column]).replace("", pd.NA)

    # Codes 6 and 7 sit outside the documented 1-5 scale, so they are treated as missing.
    score = pd.to_numeric(df["fft_score"], errors="coerce")
    df["fft_score"] = score.where(score.between(1, 5)).astype("Int64")
    df["fft_label"] = df["fft_score"].map(FFT_SCORE_LABELS)
    df["fft_group"] = df["fft_score"].map(FFT_SCORE_GROUPS).fillna("unknown")
    df["human_sentiment"] = pd.to_numeric(df["human_sentiment"], errors="coerce").astype("Int64")

    # `pii_flag` is empty in this release, so it is not kept.
    return df[CLEAN_COLUMNS].reset_index(drop=True)


def _check_raw_shape(raw: pd.DataFrame, themes: list[str]) -> None:
    """Stop with a clear message if the export does not look like the one this code expects."""
    missing = [col for col in [*RAW_TO_CLEAN_COLUMNS, NO_ANSWER_THEME] if col not in raw.columns]
    if missing:
        raise DataIntegrityError(f"The raw export is missing columns {missing}")

    flags = raw[themes]
    if not (flags.isna() | flags.isin([0, 1])).all().all():
        raise DataIntegrityError("Theme columns must hold only 0/1 flags or blanks")


def _export_runs(df: pd.DataFrame) -> pd.Series:
    """Number the runs of consecutive rows that share a trust and service.

    The raw file stacks several exports one after another, so each run comes from
    one export. A run can be part of a bigger export, which only means less
    evidence per run; `validate_responses` checks that each respondent ends up
    with one date.
    """
    source = df["trust"].fillna("") + "|" + df["service"].fillna("")
    return source.ne(source.shift()).cumsum()


def _normalized_text(values: pd.Series) -> pd.Series:
    """Any column as normalized text, with missing values as empty strings."""
    return values.astype("string").fillna("").map(normalize_text)
