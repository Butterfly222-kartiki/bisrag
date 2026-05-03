"""
BIS RAG Pipeline — Main orchestrator.
Combines retrieval + LLM rationale generation.
Passes GROQ_API_KEY through to retriever for query expansion (Fix 7).
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.retriever import BISRetriever, get_retriever, IrrelevantQueryError  # noqa: F401
from src.llm import generate_rationales

CHUNKS_PATH = Path("data/chunks.json")
INDEX_DIR = Path("data/index")


class BISRecommender:
    """
    High-level interface for the BIS RAG pipeline.
    Handles index loading and query processing.
    """

    def __init__(self, groq_api_key: Optional[str] = None):
        self.retriever: Optional[BISRetriever] = None
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY", "")
        self._loaded = False

    def _ensure_loaded(self):
        if not self._loaded:
            self.retriever = BISRetriever(groq_api_key=self.groq_api_key)
            self.retriever.load_index()
            self._loaded = True

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieve top-k standard IDs for a query.
        Returns a list of standard ID strings (for eval_script compatibility).
        """
        self._ensure_loaded()
        results = self.retriever.retrieve(query, top_k=top_k)
        return [r["standard_id"] for r in results]

    def recommend(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Full recommendation pipeline: retrieval + rationale generation.
        Returns list of dicts with standard_id, title, rationale, page_number.
        """
        import time as _time
        self._ensure_loaded()
        standards = self.retriever.retrieve(query, top_k=top_k)
        _t = _time.perf_counter()
        standards_with_rationale = generate_rationales(
            query=query,
            standards=standards,
            api_key=self.groq_api_key
        )
        print(f"[TIMER] generate_rationales   : {(_time.perf_counter()-_t)*1000:.1f}ms")
        return standards_with_rationale


# ── CHANGE: New function — builds index from pre-generated chunks JSON ─────────

def build_index_from_chunks(chunks_path: str):
    """
    Build the FAISS + BM25 index from pre-generated chunks JSON.
    Use this instead of build_index_from_pdf when you already have
    bis_all_chunks.json (skips all PDF parsing).

    Steps:
        1. load_from_chunks_json() maps existing fields to parent/child schema
           and generates sliding-window child chunks.
        2. save_chunks() writes data/chunks.json.
        3. BISRetriever.build_index() builds FAISS + BM25 from those chunks.
    """
    from src.parser import load_from_chunks_json, save_chunks

    print(f"[Pipeline] Loading pre-generated chunks from: {chunks_path}")
    chunks_data = load_from_chunks_json(chunks_path)
    save_chunks(chunks_data, str(CHUNKS_PATH))

    print("[Pipeline] Building retrieval index...")
    retriever = BISRetriever()
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks_data = json.load(f)
    retriever.build_index(chunks_data)
    print("[Pipeline] Index build complete!")


# ── Original PDF function (unchanged) ─────────────────────────────────────────

def build_index_from_pdf(pdf_path: str):
    """Build the FAISS + BM25 index from the BIS SP21 PDF. Run this once."""
    from src.parser import parse_pdf, save_chunks

    print(f"[Pipeline] Parsing PDF: {pdf_path}")
    chunks_data = parse_pdf(pdf_path)
    save_chunks(chunks_data, str(CHUNKS_PATH))

    print("[Pipeline] Building retrieval index...")
    retriever = BISRetriever()
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks_data = json.load(f)
    retriever.build_index(chunks_data)
    print("[Pipeline] Index build complete!")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.pipeline <path_to_pdf>")
        sys.exit(1)
    build_index_from_pdf(sys.argv[1])