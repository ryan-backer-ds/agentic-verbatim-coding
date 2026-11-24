"""Agreement metrics for multi-label theme codes and 1-5 sentiment scores.

Plain numpy, so every formula is visible and tested. Confidence intervals use a
cluster bootstrap over respondents: answers from one person are not independent,
so whole respondents are resampled, the same unit the train/test split uses.
"""

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd


@dataclass(frozen=True)
class LabelCounts:
    """True positives, false positives and false negatives, per answer."""

    tp: np.ndarray
    fp: np.ndarray
    fn: np.ndarray

    def micro_f1(self, rows: np.ndarray | None = None) -> float:
        tp, fp, fn = (
            counts if rows is None else counts[rows] for counts in (self.tp, self.fp, self.fn)
        )
        return f1(tp.sum(), fp.sum(), fn.sum())


def f1(tp: float, fp: float, fn: float) -> float:
    """F1 from counts; 0 when there is nothing to score."""
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator else 0.0


def label_counts(
    gold: Sequence[Collection[str]], predicted: Sequence[Collection[str]]
) -> LabelCounts:
    pairs = [(set(g), set(p)) for g, p in zip(gold, predicted, strict=True)]
    return LabelCounts(
        tp=np.array([len(g & p) for g, p in pairs]),
        fp=np.array([len(p - g) for g, p in pairs]),
        fn=np.array([len(g - p) for g, p in pairs]),
    )


def per_label_scores(
    gold: Sequence[Collection[str]], predicted: Sequence[Collection[str]], labels: Sequence[str]
) -> pd.DataFrame:
    """Precision, recall, F1 and support (gold count) for every label, in codeframe order."""
    rows = []
    for label in labels:
        in_gold = np.array([label in g for g in gold])
        in_pred = np.array([label in p for p in predicted])
        tp, fp, fn = (
            (in_gold & in_pred).sum(),
            (~in_gold & in_pred).sum(),
            (in_gold & ~in_pred).sum(),
        )
        rows.append(
            {
                "theme": label,
                "support": int(in_gold.sum()),
                "predicted": int(in_pred.sum()),
                "precision": tp / (tp + fp) if tp + fp else np.nan,
                "recall": tp / (tp + fn) if tp + fn else np.nan,
                "f1": f1(tp, fp, fn) if in_gold.any() or in_pred.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def jaccard(gold: Sequence[Collection[str]], predicted: Sequence[Collection[str]]) -> float:
    """Mean overlap |gold & predicted| / |gold | predicted| per answer."""
    scores = [
        len(set(g) & set(p)) / len(set(g) | set(p)) if set(g) | set(p) else 1.0
        for g, p in zip(gold, predicted, strict=True)
    ]
    return float(np.mean(scores)) if scores else 0.0


def quadratic_weighted_kappa(
    gold: npt.ArrayLike, predicted: npt.ArrayLike, levels: int = 5
) -> float:
    """Cohen's kappa with quadratic weights for ordinal 1..levels scores (1 = perfect)."""
    g = np.asarray(gold) - 1
    p = np.asarray(predicted) - 1
    observed = np.zeros((levels, levels))
    np.add.at(observed, (g, p), 1)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / max(len(g), 1)
    i, j = np.indices((levels, levels))
    weights = (i - j) ** 2 / (levels - 1) ** 2
    denominator = (weights * expected).sum()
    return float(1 - (weights * observed).sum() / denominator) if denominator else 1.0


def cluster_bootstrap(
    statistic: Callable[[np.ndarray], float],
    groups: Sequence[str],
    *,
    n_resamples: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """95% interval for `statistic(row_indices)`, resampling whole groups with replacement."""
    codes, uniques = pd.factorize(pd.Series(groups))
    rows_by_group = [np.flatnonzero(codes == g) for g in range(len(uniques))]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_resamples):
        picked = rng.integers(0, len(rows_by_group), len(rows_by_group))
        values.append(statistic(np.concatenate([rows_by_group[g] for g in picked])))
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)
