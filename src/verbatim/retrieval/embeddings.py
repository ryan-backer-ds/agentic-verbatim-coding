"""Gemini embeddings behind LangChain's `Embeddings` interface, plus a disk cache.

Answers are embedded once. `CachedEmbeddings` stores each vector under the
SHA-256 of its text, so rebuilding the index, re-running an evaluation or adding
a survey wave only pays for text it has never seen.
"""

import hashlib
import logging
import time
from pathlib import Path

import pandas as pd
from google import genai
from google.genai.types import EmbedContentConfig
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

_BATCH_SIZE = 100  # texts per embed_content request
_SAVE_EVERY = 500  # texts embedded between cache saves


class GeminiEmbeddings(Embeddings):
    """Gemini embeddings with retrieval task types (documents and queries differ)."""

    def __init__(
        self,
        client: genai.Client,
        *,
        model: str,
        dimensions: int,
        texts_per_minute: int | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._dimensions = dimensions
        # Free-tier keys count every text against a per-minute quota. Small, evenly paced
        # batches keep every rolling minute under it, not just the average.
        self._seconds_per_text = 60 / texts_per_minute if texts_per_minute else 0.0
        self.batch_size = max(1, texts_per_minute // 6) if texts_per_minute else _BATCH_SIZE

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            started = time.monotonic()
            vectors += self._embed(batch, "RETRIEVAL_DOCUMENT")
            # Pace every batch, including the last: the next call may start right away.
            budget = len(batch) * self._seconds_per_text
            time.sleep(max(0.0, budget - (time.monotonic() - started)))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        config = EmbedContentConfig(task_type=task_type, output_dimensionality=self._dimensions)
        response = self._client.models.embed_content(
            model=self._model, contents=list(texts), config=config
        )
        return [list(embedding.values or []) for embedding in response.embeddings or []]


class CachedEmbeddings(Embeddings):
    """Wraps another `Embeddings` and keeps document vectors in a parquet file.

    Query vectors are only cached in memory, for the life of the process.
    """

    def __init__(self, inner: Embeddings, path: Path) -> None:
        self._inner = inner
        self._path = path
        self._vectors: dict[str, list[float]] = {}
        self._queries: dict[str, list[float]] = {}
        if path.exists():
            cached = pd.read_parquet(path)
            vectors = cached["vector"].map(list)
            self._vectors = dict(zip(cached["text_sha256"], vectors, strict=True))

    def __len__(self) -> int:
        return len(self._vectors)

    def count_missing(self, texts: list[str]) -> int:
        """How many distinct `texts` have no cached vector yet."""
        return len({_sha256(text) for text in texts} - self._vectors.keys())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        keys = [_sha256(text) for text in texts]
        missing = {key: text for key, text in zip(keys, texts, strict=True) if key not in self}
        if missing:
            logger.info("Embedding %d new texts (%d cached)", len(missing), len(self._vectors))
        pending = list(missing.items())
        # Save after every chunk, so an interrupted run resumes instead of starting over.
        for start in range(0, len(pending), _SAVE_EVERY):
            chunk = dict(pending[start : start + _SAVE_EVERY])
            vectors = self._inner.embed_documents(list(chunk.values()))
            self._vectors.update(zip(chunk, vectors, strict=True))
            self._save()
            logger.info("Embedded %d / %d", min(start + _SAVE_EVERY, len(pending)), len(pending))
        return [self._vectors[key] for key in keys]

    def embed_query(self, text: str) -> list[float]:
        if text not in self._queries:
            self._queries[text] = self._inner.embed_query(text)
        return self._queries[text]

    def __contains__(self, key: object) -> bool:
        return key in self._vectors

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        table = pd.DataFrame({"text_sha256": list(self._vectors), "vector": self._vectors.values()})
        table.to_parquet(self._path, index=False)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
