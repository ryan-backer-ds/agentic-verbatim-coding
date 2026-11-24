import pytest

from verbatim.analysis import theme_shares, wilson_interval
from verbatim.data import SegmentFilter


def test_wilson_interval_known_values():
    assert wilson_interval(5, 10) == pytest.approx((0.2366, 0.7634), abs=1e-4)
    assert wilson_interval(0, 10)[0] == 0.0  # never below zero, unlike the normal approximation
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_theme_shares_for_a_segment(toy_responses):
    [group] = theme_shares(toy_responses, segment=SegmentFilter(trust="NEAS"), min_group_size=1)

    shares = {t.theme: t.share for t in group.themes}
    assert group.answers == 6
    assert shares["Positive experience & gratitude"] == pytest.approx(3 / 6)
    assert shares["Timeliness of care"] == pytest.approx(2 / 6)


def test_small_groups_are_suppressed(toy_responses):
    groups = theme_shares(
        toy_responses, segment=SegmentFilter(), group_by="trust", min_group_size=3
    )

    by_name = {g.group: g for g in groups}
    assert not by_name["NEAS"].suppressed
    assert by_name["NHFT"].suppressed and by_name["NHFT"].themes == []


def test_toolkit_returns_plain_dictionaries(toy_toolkit):
    stats = toy_toolkit.theme_stats(SegmentFilter(fft_group="negative"))
    quotes = toy_toolkit.search_answers("late", SegmentFilter(fft_group="negative"), k=50)

    assert stats["segment"] == "fft_group=negative"
    assert stats["groups"][0]["themes"][0]["theme"] == "Timeliness of care"
    assert {q["comment_id"] for q in quotes} == {"c1", "c5"}
    assert set(toy_toolkit.segment_options()["trust"]) == {"NEAS", "NHFT"}
