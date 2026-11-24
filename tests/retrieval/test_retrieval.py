import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from pydantic import Field, SecretStr

from verbatim.data import SegmentFilter
from verbatim.exceptions import MissingArtifactError
from verbatim.retrieval import VerbatimIndex, to_documents
from verbatim.retrieval.embeddings import CachedEmbeddings
from verbatim.retrieval.index import reciprocal_rank_fusion
from verbatim.retrieval.keyword import BM25Retriever


class CountingEmbeddings(DeterministicFakeEmbedding):
    """Fake embeddings that remember which texts reached the "API"."""

    calls: list[str] = Field(default_factory=list)

    def embed_documents(self, texts):
        self.calls.extend(texts)
        return super().embed_documents(texts)


def test_cached_embeddings_only_pay_for_new_text(tmp_path):
    inner = CountingEmbeddings(size=8)
    cache = CachedEmbeddings(inner, tmp_path / "embeddings.parquet")

    first = cache.embed_documents(["late ambulance", "kind staff", "late ambulance"])
    reloaded = CachedEmbeddings(inner, tmp_path / "embeddings.parquet")
    second = reloaded.embed_documents(["kind staff", "parking"])

    assert inner.calls == ["late ambulance", "kind staff", "parking"]
    assert first[0] == first[2] and second[0] == first[1]
    assert len(reloaded) == 3


def test_reciprocal_rank_fusion_rewards_agreement():
    def docs(*ids):
        return [Document(page_content=i, metadata={"comment_id": i}) for i in ids]

    fused = reciprocal_rank_fusion([docs("a", "b", "c"), docs("b", "d", "e")])

    ids = [doc.metadata["comment_id"] for doc in fused]
    assert ids[0] == "b"  # second in one list, first in the other: beats a single first place
    assert sorted(ids) == ["a", "b", "c", "d", "e"]


def test_bm25_finds_keywords_inside_a_segment(toy_index):
    keyword = BM25Retriever(documents=list(toy_index._documents.values()))

    everywhere = keyword.search("staff", k=5)
    in_forensic = keyword.search("staff", k=5, segment=SegmentFilter(service="forensic"))

    assert {d.metadata["comment_id"] for d in everywhere} == {"c3", "c4", "c8"}
    assert {d.metadata["comment_id"] for d in in_forensic} == {"c4", "c8"}
    assert keyword.search("helicopter") == []


def test_search_respects_the_segment_with_every_method(toy_index):
    segment = SegmentFilter(fft_group="negative")

    for method in ("keyword", "dense", "hybrid"):
        hits = toy_index.search("two hours late", k=3, segment=segment, method=method)
        assert hits, method
        assert all(doc.metadata["fft_group"] == "negative" for doc in hits), method


def test_similar_answers_come_from_the_requested_split(toy_index):
    similar = toy_index.similar_answers("Nothing.", k=1, segment=SegmentFilter(split="train"))

    assert similar[0].metadata["comment_id"] == "c2"  # the identical training answer


def test_keyword_mode_needs_no_embeddings(toy_responses):
    index = VerbatimIndex(to_documents(toy_responses), None)

    assert not index.has_dense
    assert index.search("late", method="hybrid")  # falls back to keyword search
    assert index.similar_answers("Nothing.", k=1, segment=SegmentFilter(split="train"))


def test_hybrid_mode_refuses_a_half_built_index(toy_settings):
    settings = toy_settings.model_copy(update={"google_ai_api_key": SecretStr("fake-key")})

    with pytest.raises(MissingArtifactError, match="build-index"):
        VerbatimIndex.load(settings)
