"""
pipeline.py
===========
High-level orchestrator for the BIS RAG pipeline.

Wires together:
    - BISRetriever (retriever.py) — finds relevant standard chunks
    - generate_rationales (response_generator.py) — explains why each standard matters

BISRecommender is the single object that entry points (api.py, inference.py)
instantiate and interact with.  It is intentionally thin: all domain logic
lives in the modules it delegates to.

Index build helpers (build_index_from_chunks, build_index_from_pdf) are also
here so build_index.py has a single import target.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.retriever import BISRetriever, get_retriever, IrrelevantQueryError  # noqa: F401
from src.response_generator import generate_rationales

CHUNKS_PATH = Path("data/chunks.json")
INDEX_DIR   = Path("data/index")


class BISRecommender:
    """
    Façade over the full recommendation pipeline.

    Keeps the index load lazy (first request triggers it) so the object can be
    constructed cheaply at module import time.  The api.py startup hook calls
    _ensure_loaded() explicitly to pre-warm everything before the first request.
    """

    def __init__(self, groq_api_key: Optional[str] = None):
        self.retriever: Optional[BISRetriever] = None
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY", "")
        self._loaded = False

    def _ensure_loaded(self):
        """Initialise and load the retriever if not already done."""
        if not self._loaded:
            self.retriever = BISRetriever(groq_api_key=self.groq_api_key)
            self.retriever.load_index()
            self._loaded = True

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieve top-k standard IDs for a query.

        Returns a plain list of standard ID strings — the format expected by
        eval_script.py for benchmarking retrieval quality.
        """
        self._ensure_loaded()
        results = self.retriever.retrieve(query, top_k=top_k)
        return [r["standard_id"] for r in results]

    def recommend(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Full recommendation pipeline: retrieval + rationale generation.

        Returns a list of dicts, each with:
            standard_id, title, text, page_number, rationale
        """
        self._ensure_loaded()
        standards = self.retriever.retrieve(query, top_k=top_k)
        standards_with_rationale = generate_rationales(
            query=query,
            standards=standards,
            api_key=self.groq_api_key,
        )
        return standards_with_rationale


# ---------------------------------------------------------------------------
# Index build helpers — called by build_index.py CLI.
# ---------------------------------------------------------------------------

def build_index_from_chunks(chunks_path: str):
    """
    Build the FAISS + BM25 index from a pre-generated chunks JSON file.

    Prefer this over build_index_from_pdf when bis_all_chunks.json already
    exists — it skips all PDF parsing and is significantly faster.

    Steps:
        1. load_from_chunks_json() maps the existing fields to parent/child
           schema and generates sliding-window child chunks.
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


def build_index_from_pdf(pdf_path: str):
    """
    Build the FAISS + BM25 index directly from the BIS SP21 PDF.

    Requires PyMuPDF or pdfplumber to be installed.  Run this once; after that
    use build_index_from_chunks() with the saved chunks.json for much faster
    rebuilds.
    """
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
