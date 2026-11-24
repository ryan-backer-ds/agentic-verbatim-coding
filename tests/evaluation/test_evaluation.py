import numpy as np
import pandas as pd
import pytest

from verbatim.evaluation import human_agreement_on_repeats, paired_difference, score_coding
from verbatim.evaluation.metrics import (
    cluster_bootstrap,
    f1,
    jaccard,
    label_counts,
    per_label_scores,
    quadratic_weighted_kappa,
)
from verbatim.exceptions import DataIntegrityError

GOLD = [["a"], ["a", "b"], ["c"]]
PREDICTED = [["a"], ["a"], ["b"]]  # tp=2, fp=1, fn=2


def test_micro_f1_from_label_counts():
    counts = label_counts(GOLD, PREDICTED)

    assert (counts.tp.sum(), counts.fp.sum(), counts.fn.sum()) == (2, 1, 2)
    assert counts.micro_f1() == pytest.approx(2 * 2 / (2 * 2 + 1 + 2))
    assert counts.micro_f1(np.array([0])) == 1.0  # only the first answer


def test_f1_of_nothing_is_zero():
    assert f1(0, 0, 0) == 0.0


def test_per_label_scores_report_support_and_missing_values():
    scores = per_label_scores(GOLD, PREDICTED, ["a", "b", "c", "d"]).set_index("theme")

    assert scores.loc["a", "support"] == 2 and scores.loc["a", "precision"] == 1.0
    assert scores.loc["b", "recall"] == 0.0
    assert np.isnan(scores.loc["d", "f1"])  # never used by anyone: not scored


def test_jaccard_averages_overlap_per_answer():
    assert jaccard(GOLD, PREDICTED) == pytest.approx((1 + 0.5 + 0) / 3)


def test_quadratic_kappa():
    assert quadratic_weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == 1.0
    assert quadratic_weighted_kappa([1, 5, 1, 5], [5, 1, 5, 1]) < 0  # worse than chance
    near = quadratic_weighted_kappa([1, 2, 4, 5], [2, 2, 4, 4])
    far = quadratic_weighted_kappa([1, 2, 4, 5], [5, 4, 2, 1])
    assert near > far


def test_cluster_bootstrap_brackets_the_estimate():
    values = np.arange(100, dtype=float)
    groups = [f"r{i // 2}" for i in range(100)]

    low, high = cluster_bootstrap(lambda rows: values[rows].mean(), groups, n_resamples=300)

    assert low < values.mean() < high


def test_score_coding_counts_failed_answers_as_empty_predictions():
    table = pd.DataFrame(
        {
            "respondent_id": ["r1", "r2", "r3"],
            "human_themes": GOLD,
            "themes": [["a"], ["a", "b"], ["c"]],
            "error": [None, None, "TimeoutError"],
            "sentiment": pd.array([1, 2, None], dtype="Int64"),
            "human_sentiment": pd.array([1, 4, 5], dtype="Int64"),
            "cost_usd": [0.001, 0.001, 0.0],
        }
    )
    meta = {"name": "toy", "model": "m", "few_shot": 0}

    scores = score_coding(table, ["a", "b", "c"], meta)

    assert scores.errors == 1
    assert scores.micro_f1 == pytest.approx(f1(3, 0, 1))  # the failed "c" is a miss
    assert scores.exact_match == pytest.approx(2 / 3)
    assert scores.sentiment_within_one == 0.5
    assert scores.cost_per_1k_answers == pytest.approx(2 / 3)


def test_human_agreement_on_repeated_answers():
    responses = pd.DataFrame(
        {
            "question": ["Q"] * 4,
            "answer": ["Nothing.", "nothing.", "Nothing.", "Late"],
            "human_themes": [["gratitude"], ["gratitude"], ["not assigned"], ["timeliness"]],
        }
    )

    # three codings of "nothing": 1 agreeing pair, 2 disagreeing pairs
    assert human_agreement_on_repeats(responses) == pytest.approx(f1(1, 2, 2))


def _run_table(themes, error=None):
    return pd.DataFrame(
        {
            "comment_id": ["c1", "c2", "c3", "c4"],
            "respondent_id": ["r1", "r2", "r3", "r4"],
            "human_themes": [["a"], ["b"], ["a"], ["b"]],
            "themes": themes,
            "error": error or [None] * 4,
        }
    )


def test_paired_difference_compares_runs_on_the_same_answers():
    better = _run_table([["a"], ["b"], ["a"], ["a"]])
    worse = _run_table([["a"], ["a"], ["b"], ["a"]])

    difference, low, high = paired_difference(better, worse, n_resamples=200)

    assert difference == pytest.approx(0.75 - 0.25)
    assert low <= difference <= high


def test_paired_difference_refuses_different_answers():
    other = _run_table([["a"]] * 4).assign(comment_id=["x1", "x2", "x3", "x4"])

    with pytest.raises(DataIntegrityError, match="same answers"):
        paired_difference(_run_table([["a"]] * 4), other)
