import pandas as pd
import pytest

from verbatim.data.cleaning import clean_responses, normalize_text, parse_survey_dates, scrub_pii
from verbatim.data.schema import CLEAN_COLUMNS
from verbatim.exceptions import DataIntegrityError


def test_clean_returns_the_contract_columns(raw_export):
    assert list(clean_responses(raw_export).columns) == CLEAN_COLUMNS


def test_clean_drops_non_answers(raw_export):
    # c2 "Did not answer", c3 blank, c5 coded "Labelling not possible" by the coders
    assert clean_responses(raw_export)["comment_id"].tolist() == ["c1", "c4"]


def test_clean_decodes_human_themes(raw_export):
    themes = clean_responses(raw_export).set_index("comment_id")["human_themes"]

    assert themes["c1"] == ["Staff manner & personal attributes", "Timeliness of care"]
    assert themes["c4"] == ["Timeliness of care"]


def test_clean_maps_scores_and_groups(raw_export):
    df = clean_responses(raw_export).set_index("comment_id")

    assert df.loc["c1", "fft_group"] == "positive"
    assert pd.isna(df.loc["c4", "fft_score"])
    assert df.loc["c4", "fft_group"] == "unknown"


def test_clean_reads_human_sentiment_as_integers(raw_export):
    sentiment = clean_responses(raw_export).set_index("comment_id")["human_sentiment"]

    assert sentiment.dtype == "Int64"
    assert sentiment["c1"] == 3
    assert pd.isna(sentiment["c4"])


def test_clean_normalizes_text_and_scrubs_pii(raw_export):
    df = clean_responses(raw_export).set_index("comment_id")

    assert df.loc["c1", "answer"] == "Staff were lovely but the ambulance was 2 hours late"
    assert df.loc["c1", "service"] == "PTS - Patient Transport"
    assert df.loc["c1", "question"] == "FFT Why?"
    assert df.loc["c4", "answer"] == "Call me on [PHONE]"


def test_clean_reads_dates_per_export_using_dropped_rows_too(raw_export):
    # Rows c1-c2 are one export (UK), c3-c5 another. c3 is dropped as a blank answer,
    # but its date, readable only as US, still decides how c4 and c5 are read.
    raw_export.loc[2:, "Service type 1"] = "See and Convey"
    raw_export["Date"] = ["03/10/2021", "30/09/2021", "10/17/2021", "10/03/2021", "10/03/2021"]

    dates = clean_responses(raw_export).set_index("comment_id")["date"]

    assert dates["c1"] == dates["c4"] == pd.Timestamp("2021-10-03")


def test_clean_rejects_a_raw_file_without_the_expected_columns(raw_export):
    with pytest.raises(DataIntegrityError, match=r"missing columns \['Trust'\]"):
        clean_responses(raw_export.drop(columns="Trust"))


def test_clean_rejects_theme_columns_that_are_not_flags(raw_export):
    raw_export["Unnamed: 0"] = range(len(raw_export))  # a saved index read back as a column

    with pytest.raises(DataIntegrityError, match="0/1 flags"):
        clean_responses(raw_export)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("mail jo.bloggs@nhs.net please", "mail [EMAIL] please"),
        ("ends with jo@nhs.net.", "ends with [EMAIL]."),
        ("ring +44 7700 900123", "ring [PHONE]"),
        ("ring +44 (0)7700 900123", "ring [PHONE]"),
        ("call (0191) 234 5678", "call [PHONE]"),
        ("my NHS no is 943 476 5919", "my NHS no is [NHS_NUMBER]"),
        ("my NHS no is 943-476-5919", "my NHS no is [NHS_NUMBER]"),
        ("I live in NE1 4LP", "I live in [POSTCODE]"),
        ("I live in ne1 4lp", "I live in [POSTCODE]"),
        # Text that must not be masked
        ("waited 2 hours for 111", "waited 2 hours for 111"),
        ("take the A1 2nd exit", "take the A1 2nd exit"),
        ("Ward A1 2pm visit", "Ward A1 2pm visit"),
        ("came on 30/09/2021", "came on 30/09/2021"),
    ],
)
def test_scrub_pii(text, expected):
    assert scrub_pii(text) == expected


def test_normalize_text():
    assert normalize_text("  very\N{NO-BREAK SPACE}good\n\nstaff ") == "very good staff"


def _parse(dates, exports):
    parsed = parse_survey_dates(pd.Series(dates), exports=pd.Series(exports))
    return parsed.dt.strftime("%Y-%m-%d").tolist()


def test_parse_survey_dates_decides_the_format_per_export():
    # The real case: one respondent's two answers, from a UK and a US export.
    dates = ["03/10/2021", "30/09/2021", "10/03/2021", "9/30/2021"]

    assert _parse(dates, exports=[1, 1, 2, 2]) == [
        "2021-10-03",
        "2021-09-30",
        "2021-10-03",
        "2021-09-30",
    ]


def test_parse_survey_dates_reads_an_undecided_export_as_uk():
    assert _parse(["03/10/2021", "9/5/2022", "13/5/2022"], exports=[1, 1, 1]) == [
        "2021-10-03",
        "2022-05-09",
        "2022-05-13",
    ]


def test_parse_survey_dates_rejects_an_export_with_both_formats():
    with pytest.raises(DataIntegrityError, match="mixes UK and US"):
        _parse(["30/09/2021", "10/17/2021"], exports=[1, 1])


def test_parse_survey_dates_rejects_garbage():
    with pytest.raises(DataIntegrityError, match="Unparseable"):
        _parse(["30/09/2021", "not a date"], exports=[1, 1])
