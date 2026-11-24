"""Step 3: finding the right answers to quote, with LangChain building blocks.

- embeddings.py: Gemini embeddings as a LangChain `Embeddings`, cached on disk by text hash
- keyword.py: BM25 keyword search as a LangChain retriever, with segment filters
- index.py: VerbatimIndex, hybrid dense + keyword search with fusion and diversity
"""

from verbatim.retrieval.index import IndexSummary, VerbatimIndex, build_index, to_documents

__all__ = ["IndexSummary", "VerbatimIndex", "build_index", "to_documents"]
