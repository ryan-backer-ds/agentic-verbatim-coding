"""Scoring coding runs against the human coders (step 6).

Answers whose call failed count as empty predictions, so a run can't look
better by failing on hard answers.
"""

import json
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from verbatim.coding import load_coding_run
from verbatim.config import Settings, get_settings
from verbatim.data import load_codeframe
from verbatim.evaluation.metrics import (
    cluster_bootstrap,
    f1,
    jaccard,
    label_counts,
    per_label_scores,
    quadratic_weighted_kappa,
)
from verbatim.exceptions import DataIntegrityError


@dataclass(frozen=True)
class CodingScores:
    """How closely one run agrees with the human coders, and what it cost."""

    run: str
    model: str
    few_shot: int
    answers: int
    errors: int
    micro_f1: float
    micro_f1_ci: tuple[float, float]
    macro_f1: float
    jaccard: float
    exact_match: float
    sentiment_within_one: float
    sentiment_kappa: float
    cost_per_1k_answers: float
    per_theme: pd.DataFrame


def score_coding(table: pd.DataFrame, codeframe: list[str], meta: dict[str, Any]) -> CodingScores:
    """Scores for a coding-run table (one row per answer, `themes` vs `human_themes`)."""
    gold = table["human_themes"].tolist()
    failed = table["error"].notna()
    predicted = _predictions(table)
    counts = label_counts(gold, predicted)
    per_theme = per_label_scores(gold, predicted, codeframe)

    scored = table["sentiment"].notna() & table["human_sentiment"].notna()
    human = table.loc[scored, "human_sentiment"].astype(int).to_numpy()
    model = table.loc[scored, "sentiment"].astype(int).to_numpy()

    return CodingScores(
        run=meta["name"],
        model=meta["model"],
        few_shot=int(meta["few_shot"]),
        answers=len(table),
        errors=int(failed.sum()),
        micro_f1=counts.micro_f1(),
        micro_f1_ci=cluster_bootstrap(counts.micro_f1, table["respondent_id"].tolist()),
        macro_f1=float(per_theme.loc[per_theme["support"] > 0, "f1"].fillna(0).mean()),
        jaccard=jaccard(gold, predicted),
        exact_match=float(
            np.mean([set(g) == set(p) for g, p in zip(gold, predicted, strict=True)])
        ),
        sentiment_within_one=float(np.mean(np.abs(human - model) <= 1)) if len(human) else 0.0,
        sentiment_kappa=quadratic_weighted_kappa(human, model) if len(human) else 0.0,
        cost_per_1k_answers=1000 * float(table["cost_usd"].sum()) / max(len(table), 1),
        per_theme=per_theme,
    )


def evaluate_coding_run(name: str, settings: Settings | None = None) -> CodingScores:
    """Load a saved run by name and score it."""
    settings = settings or get_settings()
    table = load_coding_run(name, settings)
    meta = json.loads((settings.data.coded_dir / f"{name}.json").read_text(encoding="utf-8"))
    return score_coding(table, load_codeframe(settings), meta)


def human_agreement_on_repeats(responses: pd.DataFrame, max_pairs_per_text: int = 2000) -> float:
    """Micro F1 between human codings of the same text to the same question.

    "Nothing." was answered hundreds of times and not always coded alike. How
    far coders agree with each other on identical answers is a rough ceiling
    for any model.
    """
    tp = fp = fn = 0
    groups = responses.groupby([responses["question"], responses["answer"].str.lower()])
    for _, group in groups:
        codings = [set(themes) for themes in group["human_themes"]]
        for a, b in list(combinations(codings, 2))[:max_pairs_per_text]:
            tp, fp, fn = tp + len(a & b), fp + len(b - a), fn + len(a - b)
    return f1(tp, fp, fn)


def paired_difference(
    run: pd.DataFrame, baseline: pd.DataFrame, *, n_resamples: int = 2000
) -> tuple[float, float, float]:
    """Micro F1 of `run` minus `baseline` on the same answers, with a 95% interval.

    Pairing removes the variation that comes from which answers were sampled, so
    it detects smaller real differences than comparing two separate intervals.
    """
    if run["comment_id"].tolist() != baseline["comment_id"].tolist():
        raise DataIntegrityError("Paired comparisons need two runs over the same answers.")
    ours = label_counts(run["human_themes"].tolist(), _predictions(run))
    theirs = label_counts(baseline["human_themes"].tolist(), _predictions(baseline))

    def difference(rows: np.ndarray) -> float:
        return ours.micro_f1(rows) - theirs.micro_f1(rows)

    low, high = cluster_bootstrap(
        difference, run["respondent_id"].tolist(), n_resamples=n_resamples
    )
    return difference(np.arange(len(run))), low, high


def _predictions(table: pd.DataFrame) -> list[list[str]]:
    """Predicted themes, with failed answers as empty predictions."""
    failed = table["error"].notna()
    return [[] if fail else themes for themes, fail in zip(table["themes"], failed, strict=True)]
