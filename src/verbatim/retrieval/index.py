"""Hybrid search over survey answers: meaning, keywords, and no near-duplicate quotes.

1. dense: Gemini embeddings in a LangChain `InMemoryVectorStore` (cosine similarity),
2. keyword: BM25, which catches exact words such as "parking" or "111",
3. fusion: reciprocal rank fusion merges the two ranked lists,
4. diversity: maximal marginal relevance, so ten quotes are not ten "Nothing."s.

Every step respects a `SegmentFilter`. With RETRIEVAL__MODE=keyword the index
has no dense part and every search is a BM25 search, which needs no API calls.

On Google Cloud the same design maps to Vertex AI Vector Search (dense) or
BigQuery VECTOR_SEARCH, with this module as the local, zero-infrastructure version.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.vectorstores.utils import maximal_marginal_relevance

from verbatim.config import Settings, get_settings
from verbatim.data import SegmentFilter, load_responses
from verbatim.exceptions import MissingArtifactError
from verbatim.llm import gemini_client
from verbatim.retrieval.embeddings import CachedEmbeddings, GeminiEmbeddings
from verbatim.retrieval.keyword import BM25Retriever

logger = logging.getLogger(__name__)

SearchMethod = Literal["hybrid", "dense", "keyword"]

_RRF_K = 60  # the usual constant: damps the weight of the very top ranks
_METADATA_COLUMNS = [
    "comment_id",
    "respondent_id",
    "trust",
    "service",
    "question",
    "fft_group",
    "split",
    "human_themes",
    "human_sentiment",
]


@dataclass(frozen=True)
class IndexSummary:
    answers: int
    unique_texts: int
    cached_vectors: int


def to_documents(responses: pd.DataFrame) -> list[Document]:
    """One LangChain document per answer, with the columns filters and agents need."""
    records = responses[["answer", *_METADATA_COLUMNS]].to_dict("records")
    documents = []
    for record in records:
        answer = record.pop("answer")
        metadata = {key: _plain(value) for key, value in record.items()}
        documents.append(Document(page_content=answer, metadata=metadata, id=record["comment_id"]))
    return documents


def build_embeddings(settings: Settings | None = None) -> CachedEmbeddings:
    """Gemini embeddings (AI Studio or Vertex AI, as configured) with the disk cache."""
    settings = settings or get_settings()
    gemini = GeminiEmbeddings(
        gemini_client(settings),
        model=settings.retrieval.embedding_model,
        dimensions=settings.retrieval.dimensions,
        texts_per_minute=settings.retrieval.embed_texts_per_minute,
    )
    return CachedEmbeddings(gemini, settings.data.embeddings_path)


def build_index(settings: Settings | None = None) -> IndexSummary:
    """Embed every answer once (step 3). Re-running only embeds new text."""
    settings = settings or get_settings()
    responses = load_responses(settings)
    embeddings = build_embeddings(settings)
    texts = responses["answer"].drop_duplicates().tolist()
    embeddings.embed_documents(texts)
    return IndexSummary(len(responses), len(texts), len(embeddings))


class VerbatimIndex:
    """Search over survey answers with optional segment filters."""

    def __init__(
        self, documents: list[Document], embeddings: Embeddings | None, *, candidates: int = 40
    ) -> None:
        """Pass `embeddings=None` for a keyword-only index."""
        self._documents = {doc.metadata["comment_id"]: doc for doc in documents}
        self._embeddings = embeddings
        self._candidates = candidates
        self._keyword = BM25Retriever(documents=documents)
        self._dense: InMemoryVectorStore | None = None
        if embeddings is not None:
            self._dense = InMemoryVectorStore(embeddings)
            self._dense.add_documents(documents, ids=list(self._documents))

    @classmethod
    def load(cls, settings: Settings | None = None) -> "VerbatimIndex":
        """The index over all processed answers, using the vectors `build_index` cached."""
        settings = settings or get_settings()
        documents = to_documents(load_responses(settings))
        candidates = settings.retrieval.candidates
        if settings.retrieval.mode == "keyword":
            return cls(documents, None, candidates=candidates)

        embeddings = build_embeddings(settings)
        missing = embeddings.count_missing([doc.page_content for doc in documents])
        if missing:  # fail now, not after spending quota on a half-built index
            raise MissingArtifactError(
                f"{missing:,} answers have no embedding yet. Run `verbatim build-index` "
                "(it resumes where it stopped), or set RETRIEVAL__MODE=keyword."
            )
        return cls(documents, embeddings, candidates=candidates)

    def __len__(self) -> int:
        return len(self._documents)

    @property
    def has_dense(self) -> bool:
        return self._dense is not None

    def search(
        self,
        query: str,
        *,
        k: int = 8,
        segment: SegmentFilter | None = None,
        method: SearchMethod = "hybrid",
        diversify: bool = True,
    ) -> list[Document]:
        """The `k` answers that best match `query` inside `segment`.

        Without dense vectors, every method falls back to keyword search.
        """
        if self._embeddings is None or method == "keyword":
            return self._keyword.search(query, k=k, segment=segment)
        query_vector = self._embeddings.embed_query(query)
        ranked = self._ranked(query, query_vector, segment, method)
        if diversify and len(ranked) > k:
            return self._diversify(self._embeddings, query_vector, ranked, k)
        return ranked[:k]

    def similar_answers(
        self, answer: str, *, k: int, segment: SegmentFilter | None = None
    ) -> list[Document]:
        """Answers that mean the same as `answer` (used for few-shot examples).

        The answer's own cached document vector is the query, so this costs no API
        call. A keyword-only index uses BM25 on the answer's words instead.
        """
        if self._embeddings is None or self._dense is None:
            return self._keyword.search(answer, k=k, segment=segment)
        vector = self._embeddings.embed_documents([answer])[0]
        return self._dense.similarity_search_by_vector(vector, k=k, filter=_as_filter(segment))

    def _ranked(
        self,
        query: str,
        query_vector: list[float],
        segment: SegmentFilter | None,
        method: SearchMethod,
    ) -> list[Document]:
        assert self._dense is not None  # callers check has_dense first
        dense = self._dense.similarity_search_by_vector(
            query_vector, k=self._candidates, filter=_as_filter(segment)
        )
        if method == "dense":
            return dense
        keyword = self._keyword.search(query, k=self._candidates, segment=segment)
        return reciprocal_rank_fusion([dense, keyword])

    @staticmethod
    def _diversify(
        embeddings: Embeddings, query_vector: list[float], ranked: list[Document], k: int
    ) -> list[Document]:
        vectors = embeddings.embed_documents([doc.page_content for doc in ranked])
        chosen = maximal_marginal_relevance(np.array(query_vector), vectors, lambda_mult=0.7, k=k)
        return [ranked[i] for i in chosen]


def reciprocal_rank_fusion(rankings: list[list[Document]]) -> list[Document]:
    """Merge ranked lists: each document scores the sum of 1 / (60 + rank) over the lists."""
    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            key = doc.metadata["comment_id"]
            scores[key] = scores.get(key, 0.0) + 1 / (_RRF_K + rank)
            documents[key] = doc
    return [documents[key] for key in sorted(scores, key=lambda key: -scores[key])]


def _as_filter(segment: SegmentFilter | None) -> Callable[[Document], bool] | None:
    if segment is None:
        return None
    return lambda doc: segment.matches(doc.metadata)


def _plain(value: object) -> object:
    """Metadata as plain Python values: no NaN, numpy arrays or pandas timestamps."""
    if isinstance(value, np.ndarray | list):
        return [str(item) for item in value]
    if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value
