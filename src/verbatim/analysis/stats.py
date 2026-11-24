"""Theme statistics with honest uncertainty and small-group suppression.

Shares come with 95% Wilson intervals, which behave well for small groups and
for shares near 0% or 100%. Groups with fewer answers than the minimum are
reported without numbers, the usual disclosure rule for survey results.
"""

import math
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from verbatim.data import SegmentFilter

GroupBy = Literal["fft_group", "service", "trust", "question"]


@dataclass(frozen=True)
class ThemeShare:
    theme: str
    answers: int
    share: float
    low: float
    high: float


@dataclass(frozen=True)
class GroupStats:
    """Top themes for one group of answers (or the whole segment)."""

    group: str
    answers: int
    suppressed: bool
    themes: list[ThemeShare] = field(default_factory=list)


def wilson_interval(count: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion count / total."""
    if total == 0:
        return 0.0, 0.0
    p = count / total
    centre = (p + z**2 / (2 * total)) / (1 + z**2 / total)
    spread = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / (1 + z**2 / total)
    return max(0.0, centre - spread), min(1.0, centre + spread)


def theme_shares(
    responses: pd.DataFrame,
    *,
    segment: SegmentFilter,
    group_by: GroupBy | None = None,
    top_n: int = 8,
    min_group_size: int = 10,
    themes_column: str = "human_themes",
) -> list[GroupStats]:
    """The most common themes in `segment`, overall or per group.

    A share is the fraction of answers in the group that mention the theme; an
    answer can mention several themes, so shares don't add up to 100%.
    """
    selected = responses[segment.mask(responses)]
    if group_by is None:
        groups = [(segment.describe(), selected)]
    else:
        groups = [(str(name), rows) for name, rows in selected.groupby(group_by, dropna=False)]
    return [_group_stats(name, rows, top_n, min_group_size, themes_column) for name, rows in groups]


def _group_stats(
    name: str, rows: pd.DataFrame, top_n: int, min_group_size: int, themes_column: str
) -> GroupStats:
    total = len(rows)
    if total < min_group_size:
        return GroupStats(name, total, suppressed=True)
    counts = rows[themes_column].explode().value_counts().head(top_n)
    themes = [
        ThemeShare(str(theme), int(count), int(count) / total, *wilson_interval(int(count), total))
        for theme, count in counts.items()
    ]
    return GroupStats(name, total, suppressed=False, themes=themes)
