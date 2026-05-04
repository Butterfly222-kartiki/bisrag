"""
reranker.py
===========
Takes the fused candidate list produced by RRF and scores each (query, passage)
pair with a CrossEncoder, returning only the top-k results.


"""

from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# Mode configuration — change RERANKER_MODE to switch backend.
# ---------------------------------------------------------------------------

RERANKER_MODE = "local_small"   # "local_small" | "local_large" | "groq" | "none"

# Maps each local mode to its HuggingFace model ID.
RERANKER_MODELS = {
    "local_small": "cross-encoder/ms-marco-MiniLM-L-6-v2",  # ~50 MB, fast, good enough
    "local_large": "BAAI/bge-reranker-large",               # best quality, slow
}


# ---------------------------------------------------------------------------
# Reranker class
# ---------------------------------------------------------------------------

class Reranker:
    """
    Wraps a CrossEncoder model and exposes a single rerank() method.

    The underlying model is loaded lazily on the first call to load(), so
    importing this module has no startup cost.

    Usage:
        reranker = Reranker()
        reranker.load()                          # loads model weights once
        top5 = reranker.rerank(query, candidates, top_k=5)
    """

    def __init__(self):
        # Held as None until load() is called — avoids loading weights at import time.
        self._model = None

    def load(self):
        """
        Load the CrossEncoder model weights into memory.

        Called once during server startup (api.py pre-warm) and again lazily
        inside retriever.py if pre-warm was skipped.  Safe to call multiple times —
        the second call is a no-op.

        Skipped silently when RERANKER_MODE is "groq" or "none" since those
        modes don't use a local model.
        """
        if RERANKER_MODE not in ("local_small", "local_large"):
            # Groq and none modes have no local model to load.
            return

        if self._model is not None:
            return  # Already loaded — nothing to do.

        from sentence_transformers import CrossEncoder
        model_name = RERANKER_MODELS[RERANKER_MODE]
        print(f"[Reranker] Loading model: {model_name}")
        self._model = CrossEncoder(model_name, max_length=512)

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Score each candidate against the query and return the top_k results.

        Args:
            query      : The original (unexpanded) user query string.
            candidates : List of chunk dicts from the RRF fusion step.
                         Each dict must have at least a "text" key.
            top_k      : Number of results to return after reranking.

        Returns:
            The top_k highest-scoring candidates, sorted descending by score.
            If RERANKER_MODE is "none" or the model isn't loaded, returns the
            first top_k candidates unchanged (preserving RRF order).

        Implementation note:
            We score against parent_text (full standard entry) when available,
            because the CrossEncoder benefits from more context than the short
            sliding-window child chunk.  Falls back to the child "text" field
            if parent_text is absent.
        """
        if not candidates:
            return candidates

        # "none" mode or model not loaded — return candidates as-is (RRF order).
        if self._model is None:
            return candidates[:top_k]

        # Build (query, passage) pairs — one per candidate.
        pairs = [(query, c.get("parent_text", c["text"])) for c in candidates]

        # CrossEncoder.predict() returns a score per pair; higher = more relevant.
        scores = self._model.predict(pairs)

        # Sort candidates by descending score and return the top_k.
        scored  = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]
