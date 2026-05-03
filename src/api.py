"""
FastAPI Backend for BIS Standards Recommendation Engine.
Serves the HTML frontend and provides the /recommend API endpoint.
Irrelevant/casual query detection lives in src/retriever.py (IrrelevantQueryError).
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
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

_recommender: Optional[BISRecommender] = None


def get_recommender() -> BISRecommender:
    global _recommender
    if _recommender is None:
        _recommender = BISRecommender(
            groq_api_key=os.environ.get("GROQ_API_KEY", "")
        )
    return _recommender


def _prewarm_blocking(recommender: BISRecommender):
    """Blocking helper: loads index + models. Called in a thread executor during startup."""
    recommender._ensure_loaded()                          # loads FAISS index + metadata + BM25
    recommender.retriever._load_models_if_needed()        # loads CrossEncoder reranker
    # Also trigger embedding model load (lazy inside _encode)
    if recommender.retriever.embed_model is None:
        from sentence_transformers import SentenceTransformer
        from src.retriever import EMBED_MODEL_NAME
        recommender.retriever.embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        print(f"[Startup] Embedding model loaded: {EMBED_MODEL_NAME}")


@app.on_event("startup")
async def startup_prewarm():
    """
    Pre-warm the index and all local models at server startup.
    Eliminates cold-start latency on the first real request:
      - FAISS index + metadata loaded from disk into RAM
      - Embedding model (SentenceTransformer) loaded into memory
      - CrossEncoder reranker loaded into memory
    Without this, the first request bears a ~3-10s one-time load penalty.
    """
    import asyncio
    loop = asyncio.get_event_loop()
    try:
        print("[Startup] Pre-warming index and models...")
        recommender = get_recommender()
        # Run blocking I/O + model loads in a thread so we don't block the event loop
        await loop.run_in_executor(None, _prewarm_blocking, recommender)
        print("[Startup] Pre-warm complete — server ready.")
    except Exception as e:
        # Don't crash the server if index isn't built yet; just warn
        print(f"[Startup] Pre-warm skipped ({e}). Run build_index.py first.")


# ── Models ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    groq_api_key: Optional[str] = None


class StandardResult(BaseModel):
    standard_id: str
    title: str
    rationale: str
    page_number: Optional[int] = 0


class RecommendResponse(BaseModel):
    query: str
    standards: List[StandardResult]
    latency_seconds: float
    message: Optional[str] = None   # set when query is irrelevant/casual


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"
    if frontend_path.exists():
        return HTMLResponse(content=frontend_path.read_text())
    return HTMLResponse(content="<h1>BIS RAG API v2</h1><p>See /docs for API.</p>")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "BIS RAG Engine v2"}


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: QueryRequest):
    query = req.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    start = time.time()
    try:
        recommender = get_recommender()

        # Inject per-request Groq key if provided
        if req.groq_api_key:
            recommender.groq_api_key = req.groq_api_key
            if recommender.retriever:
                recommender.retriever.groq_api_key = req.groq_api_key

        standards = recommender.recommend(query, top_k=req.top_k)

    except IrrelevantQueryError as e:
        # Casual/irrelevant query — return friendly message, no standards, no error status
        return RecommendResponse(
            query=query,
            standards=[],
            latency_seconds=round(time.time() - start, 3),
            message=e.user_message
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
            page_number=s.get("page_number") or 0
        )
        for s in standards
    ]
    return RecommendResponse(query=query, standards=results, latency_seconds=latency)


@app.post("/batch")
async def batch_recommend(queries: List[QueryRequest]):
    results = []
    for req in queries:
        start = time.time()
        try:
            recommender = get_recommender()
            standard_ids = recommender.retrieve(req.query, top_k=req.top_k)
            results.append({
                "query": req.query,
                "retrieved_standards": standard_ids,
                "latency_seconds": round(time.time() - start, 3)
            })
        except IrrelevantQueryError as e:
            results.append({"query": req.query, "message": e.user_message, "retrieved_standards": []})
        except Exception as e:
            results.append({"query": req.query, "error": str(e)})
    return results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False, log_level="info")