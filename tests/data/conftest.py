"""A tiny raw export shared by the data tests."""

import pandas as pd
import pytest

from verbatim.data.schema import NO_ANSWER_THEME, RAW_TO_CLEAN_COLUMNS

STAFF = "Staff manner & personal attributes"
TIMELINESS = "Timeliness of care"


def _raw_row(comment_id, respondent_id, answer, *, score=1, sentiment=None, themes=()):
    """One raw row: every column empty except what the test case needs."""
    row = dict.fromkeys([*RAW_TO_CLEAN_COLUMNS, STAFF, TIMELINESS, NO_ANSWER_THEME])
    row.update(
        {
            "Comment ID": comment_id,
            "Trust": "NEAS",
            "Respondent ID": respondent_id,
            "Date": "30/09/2021",
            "Service type 1": "PTS\N{NO-BREAK SPACE} - Patient Transport",
            "FFT categorical answer": score,
            "FFT question": "FFT\nWhy?",
            "FFT answer": answer,
            "Comment sentiment": sentiment,
        }
    )
    for theme in themes:
        row[theme] = 1
    return row


@pytest.fixture
def raw_export() -> pd.DataFrame:
    """One row per case the cleaning rules must handle."""
    return pd.DataFrame(
        [
            _raw_row(
                "c1",
                "r1",
                "Staff were lovely   but the ambulance was 2 hours late",
                score=2,
                sentiment="3",
                themes=[STAFF, TIMELINESS],
            ),
            _raw_row("c2", "r2", "Did not answer", themes=[NO_ANSWER_THEME]),
            _raw_row("c3", "r3", None, score=7, themes=[NO_ANSWER_THEME]),
            _raw_row("c4", "r4", "Call me on 07700 900123", score=7, themes=[TIMELINESS]),
            _raw_row("c5", "r5", "zzzz", themes=[NO_ANSWER_THEME]),
        ]
    )
