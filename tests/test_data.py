import pandas as pd
import pytest

from verbatim.data import (
    METADATA_COLUMNS,
    NO_ANSWER_THEME,
    clean,
    normalize_text,
    parse_survey_dates,
    scrub_pii,
    split_by_respondent,
)

THEMES = ["Staff manner & personal attributes", "Timeliness of care", NO_ANSWER_THEME]


def make_row(comment_id, respondent_id, answer, score=1, themes=()):
    row = dict.fromkeys(list(METADATA_COLUMNS) + THEMES)
    row.update(
        {
            "Comment ID": comment_id,
            "Trust": "NEAS",
            "Respondent ID": respondent_id,
            "Date": "30/09/2021",
            "Service type 1": "PTS  - Patient Transport",
            "FFT categorical answer": score,
            "FFT question": "FFT\nWhy?",
            "FFT answer": answer,
        }
    )
    for theme in themes:
        row[theme] = 1
    return row


@pytest.fixture
def raw() -> pd.DataFrame:
    return pd.DataFrame(
        [
            make_row(
                "c1",
                "r1",
                "Staff were lovely   but the ambulance was 2 hours late",
                score=2,
                themes=THEMES[:2],
            ),
            make_row("c2", "r2", "Did not answer", themes=[NO_ANSWER_THEME]),
            make_row("c3", "r3", None, score=7, themes=[NO_ANSWER_THEME]),
            make_row("c4", "r4", "Call me on 07700 900123", score=7, themes=[THEMES[1]]),
        ]
    )


def test_clean_drops_non_answers(raw):
    assert clean(raw)["comment_id"].tolist() == ["c1", "c4"]


def test_clean_decodes_human_themes(raw):
    themes = clean(raw).set_index("comment_id")["human_themes"]
    assert themes["c1"] == THEMES[:2]
    assert themes["c4"] == ["Timeliness of care"]


def test_clean_maps_scores_and_groups(raw):
    df = clean(raw).set_index("comment_id")
    assert df.loc["c1", "fft_group"] == "positive"
    assert pd.isna(df.loc["c4", "fft_score"])
    assert df.loc["c4", "fft_group"] == "unknown"


def test_clean_normalizes_text_and_scrubs_pii(raw):
    df = clean(raw).set_index("comment_id")
    assert df.loc["c1", "answer"] == "Staff were lovely but the ambulance was 2 hours late"
    assert df.loc["c1", "service"] == "PTS - Patient Transport"
    assert df.loc["c1", "question"] == "FFT Why?"
    assert df.loc["c4", "answer"] == "Call me on [PHONE]"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("mail jo.bloggs@nhs.net please", "mail [EMAIL] please"),
        ("ring +44 7700 900123", "ring [PHONE]"),
        ("my NHS no is 943 476 5919", "my NHS no is [NHS_NUMBER]"),
        ("I live in NE1 4LP", "I live in [POSTCODE]"),
        ("waited 2 hours for 111", "waited 2 hours for 111"),
    ],
)
def test_scrub_pii(text, expected):
    assert scrub_pii(text) == expected


def test_parse_survey_dates_handles_uk_and_us_formats():
    dates = pd.Series(["30/09/2021", "03/05/2022", "9/30/2022", "10/17/2021", "9/5/2022"])

    parsed = parse_survey_dates(dates).dt.strftime("%Y-%m-%d").tolist()

    assert parsed == ["2021-09-30", "2022-05-03", "2022-09-30", "2021-10-17", "2022-09-05"]


def test_parse_survey_dates_rejects_garbage():
    with pytest.raises(ValueError, match="Unparseable"):
        parse_survey_dates(pd.Series(["30/09/2021", "not a date"]))


def test_normalize_text():
    assert normalize_text("  very good\n\nstaff ") == "very good staff"


def test_split_keeps_respondents_together_and_is_stable():
    df = pd.DataFrame({"respondent_id": [f"r{i}" for i in range(1000) for _ in range(2)]})

    split = split_by_respondent(df, test_fraction=0.2, seed=42)

    assert split.equals(split_by_respondent(df, test_fraction=0.2, seed=42))
    assert df.assign(split=split).groupby("respondent_id")["split"].nunique().eq(1).all()
    assert 0.15 < split.eq("test").mean() < 0.25
