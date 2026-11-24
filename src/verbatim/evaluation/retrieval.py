"""Retrieval quality without hand-made judgements: the human codes are the relevance labels.

Each theme name becomes a query ("Parking", "Timeliness of care"). An answer is
relevant if the coders gave it that theme. Generic themes such as "Positive
experience & gratitude" are hard on purpose: their answers ("Nothing.") don't
mention the theme at all.
"""

import numpy as np
import pandas as pd

from verbatim.retrieval import VerbatimIndex
from verbatim.retrieval.index import SearchMethod

SETUPS: dict[str, tuple[SearchMethod, bool]] = {
    "keyword (BM25)": ("keyword", False),
    "dense (embeddings)": ("dense", False),
    "hybrid (RRF)": ("hybrid", False),
    "hybrid + diversity (MMR)": ("hybrid", True),
}


def evaluate_retrieval(
    index: VerbatimIndex, theme_counts: pd.Series, *, k: int = 10, min_support: int = 20
) -> pd.DataFrame:
    """Precision@k and mean reciprocal rank per search setup, over themes with enough answers.

    `theme_counts` maps theme name to how many answers the coders gave it.
    """
    themes = [theme for theme, count in theme_counts.items() if count >= min_support]
    setups = SETUPS if index.has_dense else {"keyword (BM25)": SETUPS["keyword (BM25)"]}
    rows = []
    for setup, (method, diversify) in setups.items():
        precisions, reciprocal_ranks = [], []
        for theme in themes:
            hits = index.search(str(theme), k=k, method=method, diversify=diversify)
            relevant = [theme in doc.metadata["human_themes"] for doc in hits]
            precisions.append(np.mean(relevant) if relevant else 0.0)
            first = next((rank for rank, is_hit in enumerate(relevant, start=1) if is_hit), None)
            reciprocal_ranks.append(1 / first if first else 0.0)
        rows.append(
            {
                "setup": setup,
                f"precision@{k}": float(np.mean(precisions)),
                "MRR": float(np.mean(reciprocal_ranks)),
                "themes": len(themes),
            }
        )
    return pd.DataFrame(rows)
