"""
retriever.py
============
Core retrieval logic for the BIS RAG pipeline.

Orchestrates the full retrieve() call:

    1. Query preprocessing  — relevance check + expansion  (query_preprocessor.py)
    2. Dense retrieval      — FAISS inner-product search over BGE embeddings
    3. Sparse retrieval     — BM25Okapi keyword search
    4. Fusion               — Reciprocal Rank Fusion (RRF) combines both lists
    5. Reranking            — CrossEncoder precision filter              (reranker.py)
    6. Deduplication        — one result per standard_id, hydrated from parent_map

Dependencies:
    index_builder.py      — EmbeddingEncoder, IndexStore, tokenize
    query_preprocessor.py — check_relevance_and_expand_query, IrrelevantQueryError
    reranker.py           — Reranker

Public surface used by pipeline.py and api.py:
    BISRetriever          — main class; call load_index() then retrieve()
    get_retriever()       — module-level singleton accessor
    IrrelevantQueryError  — re-exported so callers need only one import
"""

import os
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

from src.index_builder import EmbeddingEncoder, IndexStore, tokenize
from src.query_preprocessor import check_relevance_and_expand_query, IrrelevantQueryError
from src.reranker import Reranker


class BISRetriever:
    """
    Full retrieval pipeline: relevance-check -> dense+sparse -> RRF -> rerank.

    Typical lifecycle:
        retriever = BISRetriever(groq_api_key="...")
        retriever.load_index()                  # once, at startup
        results = retriever.retrieve(query)     # per request
    """

    def __init__(self, groq_api_key: Optional[str] = None):
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY", "")

        # All three sub-components are lazy — nothing heavy loads at construction.
        self.encoder     = EmbeddingEncoder()
        self.index_store = IndexStore(self.encoder)
        self.reranker    = Reranker()

    # ---- Property shim so api.py pre-warm can touch embed_model directly -----

    @property
    def embed_model(self):
        """Expose the underlying SentenceTransformer for external pre-warming."""
        return self.encoder._model

    @embed_model.setter
    def embed_model(self, value):
        self.encoder._model = value

    # ---- Index management ---------------------------------------------------

    def build_index(self, chunks_data: Dict):
        """Build and persist a fresh FAISS + BM25 index. Delegates to IndexStore."""
        self.index_store.build(chunks_data)

    def load_index(self):
        """
        Load pre-built index artefacts from disk into memory.
        Must be called before retrieve(). Raises FileNotFoundError if the index
        hasn't been built yet — run build_index.py first.
        """
        self.index_store.load()

    def _load_models_if_needed(self):
        """
        Lazy-load the reranker model on the first request that needs it.
        Delegates to Reranker.load() which is a no-op for "groq" and "none" modes.
        """
        self.reranker.load()

    # ---- Core retrieval steps -----------------------------------------------

    def _dense_retrieve(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """
        FAISS inner-product search over BGE embeddings.

        Returns (chunk_index, score) pairs in descending score order.
        The BGE query prefix is applied inside encoder.encode(is_query=True).
        """
        query_embed     = self.encoder.encode([query], is_query=True)
        scores, indices = self.index_store.faiss_index.search(query_embed, top_k)
        return list(zip(indices[0].tolist(), scores[0].tolist()))

    def _sparse_retrieve(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """
        BM25 keyword search over the tokenised corpus.

        The query is already expanded by check_relevance_and_expand_query() before
        this is called, so abbreviations and synonyms are naturally included.
        Returns (chunk_index, score) pairs in descending score order.
        """
        tokens      = tokenize(query)
        scores      = self.index_store.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]

    def _rrf_fusion(
        self,
        dense_results:  List[Tuple[int, float]],
        sparse_results: List[Tuple[int, float]],
        k: int = 60,
    ) -> List[Tuple[int, float]]:
        """
        Reciprocal Rank Fusion — merges the dense and sparse ranked lists.

        Each candidate receives a score of 1/(k + rank) from each list it
        appears in. Scores are summed, so a hit near the top of both lists
        outranks one that dominates only a single list.

        k=60 is the standard RRF constant; higher values dampen rank influence
        and make the fusion more conservative.

        Returns candidates sorted by descending RRF score.
        """
        scores: Dict[int, float] = {}
        for rank, (idx, _) in enumerate(dense_results):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
        for rank, (idx, _) in enumerate(sparse_results):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # ---- Public retrieve ----------------------------------------------------

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        End-to-end retrieval: preprocess -> dense+sparse -> RRF -> rerank -> deduplicate.

        Args:
            query  : Raw user query string.
            top_k  : Number of unique standards to return.

        Returns:
            List of dicts with keys: standard_id, title, text, page_number.

        Raises:
            IrrelevantQueryError : query is a greeting or off-topic.
                                   Caught by api.py for a friendly response.
        """
        # Step 1 — Relevance check + query expansion (single Groq call).
        # Previously two sequential calls; merged to save ~300-600 ms per request.
        relevant, expanded_query, msg = check_relevance_and_expand_query(
            query, self.groq_api_key
        )
        if not relevant:
            raise IrrelevantQueryError(msg)

        # Ensure reranker weights are in memory before we start scoring.
        self._load_models_if_needed()

        # Steps 2 & 3 — Independent dense and sparse searches on the expanded query.
        dense_res  = self._dense_retrieve(expanded_query, top_k=20)
        sparse_res = self._sparse_retrieve(expanded_query, top_k=20)

        # Step 4 — Fuse both ranked lists into one via RRF.
        fused = self._rrf_fusion(dense_res, sparse_res)

        # Step 5 — Hydrate chunk metadata from the fused indices.
        candidates   = []
        seen_indices = set()
        for idx, score in fused[:5]:
            if idx < 0 or idx in seen_indices:
                continue
            seen_indices.add(idx)
            chunk = dict(self.index_store.metadata[idx])
            chunk["rrf_score"] = score
            candidates.append(chunk)

        # Step 6 — Rerank: CrossEncoder precision filter over fused candidates.
        reranked = self.reranker.rerank(query, candidates, top_k=top_k * 2)

        # Step 7 — Deduplicate by standard_id; hydrate from parent_map for full text.
        seen_ids = set()
        results  = []
        for chunk in reranked:
            sid = chunk["standard_id"]
            if sid in seen_ids:
                continue
            seen_ids.add(sid)
            parent = self.index_store.parent_map.get(sid, chunk)
            results.append({
                "standard_id": sid,
                "title":       parent.get("title",       chunk.get("title",       "")),
                "text":        parent.get("text",        chunk.get("text",        "")),
                "page_number": parent.get("page_number", chunk.get("page_number", 0)),
            })
            if len(results) >= top_k:
                break

        return results


# ---------------------------------------------------------------------------
# Module-level singleton — api.py and pipeline.py share one retriever instance.
# ---------------------------------------------------------------------------

_retriever_instance: Optional[BISRetriever] = None


def get_retriever() -> BISRetriever:
    """
    Return the shared BISRetriever, creating and loading it on first call.
    Thread-safe for single-worker FastAPI. For multi-worker deployments, load
    the index inside each worker's startup hook instead of using this singleton.
    """
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = BISRetriever()
        _retriever_instance.load_index()
    return _retriever_instance
