"""
api.py
======
FastAPI backend for the BIS Standards Recommendation Engine.

Responsibilities:
    - Serve the HTML frontend at GET /
    - Expose POST /recommend for single-query recommendations
    - Expose POST /batch  for bulk retrieval (eval / testing)
    - Pre-warm the index and all ML models at startup to eliminate cold-start
      latency on the first real request

Irrelevant/casual query detection is handled upstream in query_preprocessor.py
(raises IrrelevantQueryError); this module catches that error and converts it
into a graceful JSON response rather than an HTTP error.
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.pipeline import BISRecommender
from src.retriever import IrrelevantQueryError

app = FastAPI(
    title="BIS Standards Recommendation Engine",
    description="AI-powered BIS standard discovery for MSEs using RAG",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Module-level recommender instance — created once, shared across all requests.
_recommender: Optional[BISRecommender] = None


def get_recommender() -> BISRecommender:
    """Return the shared BISRecommender, creating it on first call."""
    global _recommender
    if _recommender is None:
        _recommender = BISRecommender(
            groq_api_key=os.environ.get("GROQ_API_KEY", "")
        )
    return _recommender


def _prewarm_blocking(recommender: BISRecommender):
    """
    Synchronous helper that loads all heavy artefacts into memory.

    Called inside a thread executor during startup so the async event loop
    is never blocked.  Loads:
        - FAISS index + metadata + BM25 from disk
        - CrossEncoder reranker weights
        - SentenceTransformer embedding model weights

    Without this pre-warm, the first request bears a ~3–10 s cold-start penalty.
    """
    recommender._ensure_loaded()                       # FAISS + metadata + BM25
    recommender.retriever._load_models_if_needed()     # CrossEncoder reranker

    # Also trigger embedding model load (lazy inside encoder.encode).
    if recommender.retriever.embed_model is None:
        from sentence_transformers import SentenceTransformer
        from src.index_builder import EMBED_MODEL_NAME
        recommender.retriever.embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        print(f"[Startup] Embedding model loaded: {EMBED_MODEL_NAME}")


@app.on_event("startup")
async def startup_prewarm():
    """
    FastAPI startup hook — pre-warm index and models before accepting requests.

    Runs blocking I/O and model loading in a thread executor so the async
    event loop stays responsive.  If the index hasn't been built yet
    (FileNotFoundError), we log a warning instead of crashing the server.
    """
    import asyncio
    loop = asyncio.get_event_loop()
    try:
        print("[Startup] Pre-warming index and models...")
        recommender = get_recommender()
        await loop.run_in_executor(None, _prewarm_blocking, recommender)
        print("[Startup] Pre-warm complete — server ready.")
    except Exception as e:
        # Index not built yet — server still starts, just slower on first request.
        print(f"[Startup] Pre-warm skipped ({e}). Run build_index.py first.")


# ---------------------------------------------------------------------------
# Pydantic models — request / response shapes for the /recommend endpoint.
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    groq_api_key: Optional[str] = None   # Optional per-request key override.


class StandardResult(BaseModel):
    standard_id: str
    title: str
    rationale: str
    page_number: Optional[int] = 0


class RecommendResponse(BaseModel):
    query: str
    standards: List[StandardResult]
    latency_seconds: float
    message: Optional[str] = None   # Populated only for irrelevant/casual queries.


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the bundled HTML frontend."""
    frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"
    if frontend_path.exists():
        return HTMLResponse(content=frontend_path.read_text())
    return HTMLResponse(content="<h1>BIS RAG API v2</h1><p>See /docs for API.</p>")


@app.get("/health")
async def health():
    """Simple liveness probe."""
    return {"status": "ok", "service": "BIS RAG Engine v2"}


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: QueryRequest):
    """
    Main recommendation endpoint.

    Accepts a natural-language query describing a product or manufacturing
    requirement and returns the most relevant BIS standards with rationales.

    Returns HTTP 200 even for irrelevant/casual queries — the `message` field
    carries the friendly redirect text and `standards` will be empty.
    Returns HTTP 503 if the index hasn't been built yet.
    Returns HTTP 500 for unexpected errors.
    """
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    start = time.time()
    try:
        recommender = get_recommender()

        # Allow callers to override the Groq key on a per-request basis
        # (useful for multi-tenant deployments where each client has its own key).
        if req.groq_api_key:
            recommender.groq_api_key = req.groq_api_key
            if recommender.retriever:
                recommender.retriever.groq_api_key = req.groq_api_key

        standards = recommender.recommend(query, top_k=req.top_k)

    except IrrelevantQueryError as e:
        # Off-topic or greeting — return friendly message, empty standards list.
        return RecommendResponse(
            query=query,
            standards=[],
            latency_seconds=round(time.time() - start, 3),
            message=e.user_message,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Index not built yet: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency = round(time.time() - start, 3)
    results = [
        StandardResult(
            standard_id=s.get("standard_id", ""),
            title=s.get("title", ""),
            rationale=s.get("rationale", ""),
            page_number=s.get("page_number") or 0,
        )
        for s in standards
    ]
    return RecommendResponse(query=query, standards=results, latency_seconds=latency)


@app.post("/batch")
async def batch_recommend(queries: List[QueryRequest]):
    """
    Bulk retrieval endpoint — retrieve standard IDs for multiple queries.

    Used by inference.py and the eval script.  Returns standard IDs only
    (no rationales) to keep latency low for bulk evaluation.
    """
    results = []
    for req in queries:
        start = time.time()
        try:
            recommender  = get_recommender()
            standard_ids = recommender.retrieve(req.query, top_k=req.top_k)
            results.append({
                "query":               req.query,
                "retrieved_standards": standard_ids,
                "latency_seconds":     round(time.time() - start, 3),
            })
        except IrrelevantQueryError as e:
            results.append({
                "query":               req.query,
                "message":             e.user_message,
                "retrieved_standards": [],
            })
        except Exception as e:
            results.append({"query": req.query, "error": str(e)})
    return results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
