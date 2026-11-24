"""Step 1 end to end: fetch -> load -> clean -> split -> validate -> save.

`build_dataset()` only orchestrates; each step lives in its own module and can
be read and tested on its own.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from verbatim.config import Settings, get_settings
from verbatim.data.cleaning import clean_responses, theme_columns
from verbatim.data.schema import NO_ANSWER_THEME, validate_responses
from verbatim.data.source import NHS_FFT_SOURCE, DatasetSource, load_raw
from verbatim.data.splitting import split_by_respondent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetSummary:
    """The numbers worth checking after a build."""

    raw_rows: int
    responses: int
    respondents: int
    splits: dict[str, int]
    fft_groups: dict[str, int]
    themes: int
    output_dir: Path


def build_dataset(
    settings: Settings | None = None,
    *,
    source: DatasetSource = NHS_FFT_SOURCE,
    force_download: bool = False,
) -> DatasetSummary:
    """Build `responses.parquet` and `codeframe.json` from the raw survey export."""
    settings = settings or get_settings()
    data = settings.data

    raw = load_raw(source.fetch(data.raw_dir, force=force_download))
    responses = clean_responses(raw)
    responses["split"] = split_by_respondent(
        responses,
        test_fraction=data.test_fraction,
        dev_fraction=data.dev_fraction,
        seed=data.split_seed,
    )
    validate_responses(responses)
    codeframe = summarize_codeframe(theme_columns(raw), responses)

    data.processed_dir.mkdir(parents=True, exist_ok=True)
    responses.to_parquet(data.responses_path, index=False)
    data.codeframe_path.write_text(json.dumps(codeframe, indent=2), encoding="utf-8")
    logger.info("Wrote %s and %s", data.responses_path, data.codeframe_path)

    return DatasetSummary(
        raw_rows=len(raw),
        responses=len(responses),
        respondents=responses["respondent_id"].nunique(),
        splits=_value_counts(responses["split"]),
        fft_groups=_value_counts(responses["fft_group"]),
        themes=len(codeframe),
        output_dir=data.processed_dir,
    )


def summarize_codeframe(themes: list[str], responses: pd.DataFrame) -> list[dict[str, str | int]]:
    """The codeframe in its original order, with how often coders used each theme."""
    counts = responses["human_themes"].explode().value_counts()
    return [
        {"theme": theme, "n_responses": int(counts.get(theme, 0))}
        for theme in themes
        if theme != NO_ANSWER_THEME
    ]


def _value_counts(values: pd.Series) -> dict[str, int]:
    return {str(value): int(count) for value, count in values.value_counts().items()}
