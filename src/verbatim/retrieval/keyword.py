"""Keyword search with BM25, as a LangChain retriever.

langchain-community's BM25Retriever is being retired along with that package,
and it cannot filter. This one is a few lines on top of `rank_bm25`, filters by
segment, and scores the whole collection in one vectorised call.
"""

import re

import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, PrivateAttr
from rank_bm25 import BM25Okapi

from verbatim.data.segments import SegmentFilter


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever(BaseRetriever):
    """Okapi BM25 over a fixed list of documents."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    documents: list[Document]
    k: int = 10
    _bm25: BM25Okapi = PrivateAttr()

    def model_post_init(self, context: object, /) -> None:
        self._bm25 = BM25Okapi([tokenize(doc.page_content) for doc in self.documents])

    def search(
        self, query: str, *, k: int | None = None, segment: SegmentFilter | None = None
    ) -> list[Document]:
        """The `k` best matches inside `segment`; documents with no shared word are skipped."""
        scores = np.asarray(self._bm25.get_scores(tokenize(query)))
        if segment is not None:
            inside = np.array([segment.matches(doc.metadata) for doc in self.documents])
            scores = np.where(inside, scores, 0.0)
        best = np.argsort(-scores, kind="stable")[: k or self.k]
        return [self.documents[i] for i in best if scores[i] > 0]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self.search(query)
