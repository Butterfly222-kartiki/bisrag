import os
import time
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

from src.index_builder import EmbeddingEncoder, IndexStore, tokenize
from src.query_preprocessor import check_relevance_and_expand_query, IrrelevantQueryError
from src.reranker import Reranker


class BISRetriever:
    def __init__(self, groq_api_key: Optional[str] = None):
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY", "")

        self.encoder     = EmbeddingEncoder()
        self.index_store = IndexStore(self.encoder)
        self.reranker    = Reranker()
        self.use_reranker = True

    @property
    def embed_model(self):
        return self.encoder._model

    @embed_model.setter
    def embed_model(self, value):
        self.encoder._model = value

    def build_index(self, chunks_data: Dict):
        self.index_store.build(chunks_data)

    def load_index(self):
        self.index_store.load()

    def _load_models_if_needed(self):
        self.reranker.load()

    def _dense_retrieve(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        query_embed     = self.encoder.encode([query], is_query=True)
        scores, indices = self.index_store.faiss_index.search(query_embed, top_k)
        return list(zip(indices[0].tolist(), scores[0].tolist()))

    def _sparse_retrieve(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
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
        scores: Dict[int, float] = {}
        for rank, (idx, _) in enumerate(dense_results):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
        for rank, (idx, _) in enumerate(sparse_results):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        timings = {}
        t_total_start = time.time()

        # Step 1 — Relevance + Expansion
        t0 = time.time()
        relevant, expanded_query, msg = check_relevance_and_expand_query(
            query, self.groq_api_key
        )
        timings["preprocess+expand"] = time.time() - t0

        if not relevant:
            raise IrrelevantQueryError(msg)

        # Step 2 — Load reranker (lazy)
        t1 = time.time()
        self._load_models_if_needed()
        timings["model_load"] = time.time() - t1

        # Step 3 — Dense retrieval
        t2 = time.time()
        dense_res = self._dense_retrieve(expanded_query, top_k=20)
        timings["dense_retrieval"] = time.time() - t2

        # Step 4 — Sparse retrieval
        t3 = time.time()
        sparse_res = self._sparse_retrieve(expanded_query, top_k=20)
        timings["sparse_retrieval"] = time.time() - t3

        # Step 5 — Fusion
        t4 = time.time()
        fused = self._rrf_fusion(dense_res, sparse_res)
        timings["fusion"] = time.time() - t4

        # Step 6 — Candidate building
        t5 = time.time()
        candidates = []
        seen_indices = set()
        for idx, score in fused[:5]:
            if idx < 0 or idx in seen_indices:
                continue
            seen_indices.add(idx)
            chunk = dict(self.index_store.metadata[idx])
            chunk["rrf_score"] = score
            candidates.append(chunk)
        timings["candidate_build"] = time.time() - t5

        # Step 7 — Reranking
        t6 = time.time()
        if self.use_reranker:
            reranked = self.reranker.rerank(query, candidates, top_k=top_k * 2)
        else:
            reranked = candidates 
        timings["rerank"] = time.time() - t6

        # Step 8 — Deduplication
        t7 = time.time()
        seen_ids = set()
        results = []
        for chunk in reranked:
            sid = chunk["standard_id"]
            if sid in seen_ids:
                continue
            seen_ids.add(sid)
            parent = self.index_store.parent_map.get(sid, chunk)
            results.append({
                "standard_id": sid,
                "title": parent.get("title", chunk.get("title", "")),
                "text": parent.get("text", chunk.get("text", "")),
                "page_number": parent.get("page_number", chunk.get("page_number", 0)),
            })
            if len(results) >= top_k:
                break
        timings["dedup"] = time.time() - t7

        timings["total"] = time.time() - t_total_start

        # 🔥 PRINT TIMING BREAKDOWN
        print("\n⏱ TIMING BREAKDOWN:")
        for k, v in timings.items():
            print(f"{k:20s}: {v:.4f} sec")
        print("-" * 40)

        return results


_retriever_instance: Optional[BISRetriever] = None


def get_retriever() -> BISRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = BISRetriever()
        _retriever_instance.load_index()
    return _retriever_instance