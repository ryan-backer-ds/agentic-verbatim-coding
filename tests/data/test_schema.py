import pandas as pd
import pytest

from verbatim.data.cleaning import clean_responses
from verbatim.data.schema import validate_responses
from verbatim.exceptions import DataIntegrityError


@pytest.fixture
def responses(raw_export):
    df = clean_responses(raw_export)
    df["split"] = ["train", "test"]
    return df


# One way to break each rule, keyed by the message validation must give.
BROKEN = {
    "the table is empty": lambda df: df.iloc[:0],
    "comment_id is not unique": lambda df: df.assign(comment_id="c1"),
    "respondent_id is missing": lambda df: df.assign(respondent_id=[None, "r4"]),
    "some answers are empty": lambda df: df.assign(answer=["", "Fine"]),
    "fft_score outside 1-5": lambda df: df.assign(fft_score=pd.array([1, 9], dtype="Int64")),
    "unknown fft_group": lambda df: df.assign(fft_group="promoter"),
    "unknown split": lambda df: df.assign(split="validation"),
    "a respondent is in more than one split": lambda df: df.assign(respondent_id="r1"),
    "a respondent has more than one date": lambda df: df.assign(
        respondent_id="r1", split="train", date=pd.to_datetime(["2021-10-03", "2021-03-10"])
    ),
}


def test_valid_responses_pass(responses):
    validate_responses(responses)


def test_wrong_columns_are_rejected(responses):
    with pytest.raises(DataIntegrityError, match="Columns"):
        validate_responses(responses.drop(columns="split"))


@pytest.mark.parametrize("problem", BROKEN)
def test_each_rule_is_enforced(responses, problem):
    with pytest.raises(DataIntegrityError, match=problem):
        validate_responses(BROKEN[problem](responses))


def test_every_broken_rule_is_reported(responses):
    responses.loc[1, "comment_id"] = responses.loc[0, "comment_id"]
    responses.loc[0, "split"] = "validation"

    with pytest.raises(DataIntegrityError) as error:
        validate_responses(responses)

    assert "comment_id is not unique" in str(error.value)
    assert "unknown split" in str(error.value)
