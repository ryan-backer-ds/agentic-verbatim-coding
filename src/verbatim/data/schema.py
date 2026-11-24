"""The dataset contract: column names, allowed values and the checks that enforce them.

The raw-to-clean column mapping, the saved column order and the allowed values
are written once, here, and `validate_responses` checks every build against
them before anything is saved.
"""

import pandas as pd

from verbatim.exceptions import DataIntegrityError

# Raw export column -> clean name. Every other raw column is a codeframe theme.
RAW_TO_CLEAN_COLUMNS = {
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

# Free-text category columns that need the same tidying as answers.
CATEGORY_COLUMNS = ["trust", "service", "sub_service", "question"]

# Coders used this theme for blank and "Did not answer" rows. It carries no signal.
NO_ANSWER_THEME = "Labelling not possible"

FFT_SCORE_LABELS = {
    1: "Very good",
    2: "Good",
    3: "Neither good nor poor",
    4: "Poor",
    5: "Very poor",
}
# Same idea as NPS promoters / passives / detractors, on the 5-point FFT scale.
FFT_SCORE_GROUPS = {1: "positive", 2: "positive", 3: "neutral", 4: "negative", 5: "negative"}
FFT_GROUPS = ("positive", "neutral", "negative", "unknown")

# train: few-shot examples and baselines; dev: tuning prompts and models; test: scored once.
SPLITS = ("train", "dev", "test")

# What cleaning produces, in this order. The pipeline then adds the split.
CLEAN_COLUMNS = [
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
RESPONSE_COLUMNS = [*CLEAN_COLUMNS, "split"]


def validate_responses(responses: pd.DataFrame) -> None:
    """Check the processed table before it is saved.

    A broken invariant stops the pipeline here, with every problem listed,
    instead of surfacing as odd numbers in a later step.
    """
    columns = list(responses.columns)
    if columns != RESPONSE_COLUMNS:
        raise DataIntegrityError(f"Columns are {columns}, expected {RESPONSE_COLUMNS}")

    per_respondent = responses.groupby("respondent_id")
    checks = {
        "the table is empty": responses.empty,
        "comment_id is not unique": not responses["comment_id"].is_unique,
        "respondent_id is missing": responses["respondent_id"].isna().any(),
        "some answers are empty": responses["answer"].fillna("").eq("").any(),
        "fft_score outside 1-5": not responses["fft_score"].dropna().between(1, 5).all(),
        "unknown fft_group": not responses["fft_group"].isin(FFT_GROUPS).all(),
        "unknown split": not responses["split"].isin(SPLITS).all(),
        "a respondent is in more than one split": per_respondent["split"].nunique().gt(1).any(),
        # One respondent is one survey submission, so their answers share a date.
        "a respondent has more than one date": per_respondent["date"].nunique().gt(1).any(),
    }
    problems = [problem for problem, failed in checks.items() if failed]
    if problems:
        raise DataIntegrityError(f"Processed responses failed validation: {'; '.join(problems)}")
