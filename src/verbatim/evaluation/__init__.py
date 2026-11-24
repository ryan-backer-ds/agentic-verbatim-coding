"""Step 6: measuring the agents against the human coders.

- metrics.py: multi-label F1, Jaccard, quadratic-weighted kappa, cluster bootstrap
- coding.py: scores for a coding run, and how far coders agree with each other
- retrieval.py: precision@k and MRR per search setup, with human codes as labels
"""

from verbatim.evaluation.coding import (
    CodingScores,
    evaluate_coding_run,
    human_agreement_on_repeats,
    paired_difference,
    score_coding,
)
from verbatim.evaluation.retrieval import evaluate_retrieval

__all__ = [
    "CodingScores",
    "evaluate_coding_run",
    "evaluate_retrieval",
    "human_agreement_on_repeats",
    "paired_difference",
    "score_coding",
]
