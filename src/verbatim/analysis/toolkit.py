"""The read-only tools every analyst agent gets: theme statistics and evidence search.

The LangGraph workflow, the ADK agent and the MCP server all call these same
methods, so a number or a quote is the same whichever front end asks for it.
Results are plain dictionaries, ready to hand to a model as JSON.

Statistics use the human codes. On new survey waves the coding agent's output
takes their place (`themes_column`).
"""

from dataclasses import asdict
from typing import Any

import pandas as pd

from verbatim.analysis.stats import GroupBy, theme_shares
from verbatim.config import Settings, get_settings
from verbatim.data import SegmentFilter, load_responses
from verbatim.retrieval import VerbatimIndex

MAX_QUOTES = 20


class AnalystToolkit:
    """Theme statistics and quote search over one set of coded answers."""

    def __init__(
        self,
        responses: pd.DataFrame,
        index: VerbatimIndex,
        *,
        min_group_size: int = 10,
        themes_column: str = "human_themes",
    ) -> None:
        self.responses = responses
        self.index = index
        self.min_group_size = min_group_size
        self.themes_column = themes_column

    @classmethod
    def load(cls, settings: Settings | None = None) -> "AnalystToolkit":
        settings = settings or get_settings()
        return cls(
            load_responses(settings),
            VerbatimIndex.load(settings),
            min_group_size=settings.analysis.min_group_size,
        )

    def theme_stats(
        self, segment: SegmentFilter, *, group_by: GroupBy | None = None, top_n: int = 8
    ) -> dict[str, Any]:
        """Top themes and their shares (with 95% intervals) in a segment, optionally per group."""
        groups = theme_shares(
            self.responses,
            segment=segment,
            group_by=group_by,
            top_n=top_n,
            min_group_size=self.min_group_size,
            themes_column=self.themes_column,
        )
        return {
            "segment": segment.describe(),
            "grouped_by": group_by,
            "min_group_size": self.min_group_size,
            "groups": [asdict(group) for group in groups],
        }

    def search_answers(
        self, query: str, segment: SegmentFilter, *, k: int = 8
    ) -> list[dict[str, Any]]:
        """Answers that best match `query` in a segment, as quotable evidence."""
        hits = self.index.search(query, k=min(k, MAX_QUOTES), segment=segment)
        return [
            {
                "comment_id": doc.metadata["comment_id"],
                "answer": doc.page_content,
                "question": doc.metadata["question"],
                "service": doc.metadata["service"],
                "fft_group": doc.metadata["fft_group"],
                "themes": doc.metadata["human_themes"],
            }
            for doc in hits
        ]

    def segment_options(self) -> dict[str, list[str]]:
        """The values a segment can take, to put in front of a planning model."""
        return {
            column: sorted(self.responses[column].dropna().astype(str).unique())
            for column in ("trust", "service", "question", "fft_group")
        }
