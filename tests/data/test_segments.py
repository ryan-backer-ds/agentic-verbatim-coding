import pytest
from pydantic import ValidationError

from verbatim.data import SegmentFilter


def test_an_empty_filter_keeps_everything(toy_responses):
    assert SegmentFilter().mask(toy_responses).all()
    assert SegmentFilter().describe() == "all answers"


def test_text_fields_match_part_of_the_value_ignoring_case(toy_responses):
    segment = SegmentFilter(service="transport", fft_group="negative")

    kept = toy_responses[segment.mask(toy_responses)]

    assert kept["comment_id"].tolist() == ["c1", "c5"]
    assert segment.describe() == "service=transport, fft_group=negative"


def test_mask_and_matches_agree(toy_responses):
    segment = SegmentFilter(trust="nhft", split="train")

    by_mask = toy_responses[segment.mask(toy_responses)]["comment_id"].tolist()
    by_record = [r["comment_id"] for r in toy_responses.to_dict("records") if segment.matches(r)]

    assert by_mask == by_record == ["c4"]


def test_unknown_groups_and_fields_are_rejected():
    with pytest.raises(ValidationError):
        SegmentFilter(fft_group="promoters")
    with pytest.raises(ValidationError):
        SegmentFilter(region="North East")
