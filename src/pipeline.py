"""
pipeline.py
===========
High-level orchestrator for the BIS RAG pipeline.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.retriever import BISRetriever, IrrelevantQueryError  # noqa: F401
from src.response_generator import generate_rationales

CHUNKS_PATH = Path("data/chunks.json")
INDEX_DIR   = Path("data/index")


class BISRecommender:
    def __init__(self, groq_api_key: Optional[str] = None, use_reranker: bool = True):
        self.retriever: Optional[BISRetriever] = None
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY", "")
        self._loaded = False

        # ✅ store reranker flag
        self.use_reranker = use_reranker

    def _ensure_loaded(self):
        """Initialise and load the retriever if not already done."""
        if not self._loaded:
            self.retriever = BISRetriever(groq_api_key=self.groq_api_key)

            # ✅ apply reranker flag AFTER retriever creation
            self.retriever.use_reranker = self.use_reranker

            self.retriever.load_index()
            self._loaded = True

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        self._ensure_loaded()
        results = self.retriever.retrieve(query, top_k=top_k)
        return [r["standard_id"] for r in results]

    def recommend(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        standards = self.retriever.retrieve(query, top_k=top_k)

        standards_with_rationale = generate_rationales(
            query=query,
            standards=standards,
            api_key=self.groq_api_key,
        )
        return standards_with_rationale


# ---------------------------------------------------------------------------
# Index build helpers
# ---------------------------------------------------------------------------

def build_index_from_chunks(chunks_path: str):
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


def build_index_from_pdf(pdf_path: str):
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