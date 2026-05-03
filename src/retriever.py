"""
BIS RAG Retriever — Updated for faster indexing + lightweight reranker.

EMBEDDING_MODE options:
    "local_fast"  -> BAAI/bge-small-en-v1.5  (384-dim, ~3x faster, ~1% quality drop)
    "local_large" -> BAAI/bge-large-en-v1.5  (1024-dim, original, slowest)
    "hf_api"      -> HuggingFace Inference API (no local compute, needs HF_API_KEY)

RERANKER_MODE options:
    "local_small" -> cross-encoder/ms-marco-MiniLM-L-6-v2  (fast, ~50MB, good enough)
    "local_large" -> BAAI/bge-reranker-large                (original, slow, best)
    "groq"        -> Groq LLM used as reranker              (no local model at all)
    "none"        -> skip reranking entirely                 (fastest, lower quality)
"""

import os
import json
import pickle
import re
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from rank_bm25 import BM25Okapi
import faiss


# ─────────────────────────────────────────────
# CHANGE THIS TO SWITCH EMBEDDING MODE
# ─────────────────────────────────────────────

EMBEDDING_MODE = "local_fast"   # "local_fast" | "local_large" | "hf_api"

EMBED_MODELS = {
    "local_fast":  "BAAI/bge-small-en-v1.5",   # 384-dim, ~3x faster, ~1% quality drop
    "local_large": "BAAI/bge-large-en-v1.5",   # 1024-dim, original, slowest
    "hf_api":      "BAAI/bge-large-en-v1.5",   # same quality, runs on HF servers
}

EMBED_MODEL_NAME = EMBED_MODELS[EMBEDDING_MODE]

# ── Reranker mode ──────────────────────────────────────────────────────────────
RERANKER_MODE = "local_small"   # "local_small" | "local_large" | "groq" | "none"

RERANKER_MODELS = {
    "local_small": "cross-encoder/ms-marco-MiniLM-L-6-v2",  # ~50MB, fast, good enough
    "local_large": "BAAI/bge-reranker-large",               # original, slow, best
}

# HuggingFace Inference API key - only needed if EMBEDDING_MODE = "hf_api"
# Get free key at https://huggingface.co/settings/tokens
HF_API_KEY = os.environ.get("HF_API_KEY", "")

INDEX_DIR        = Path("data/index")
FAISS_INDEX_FILE = INDEX_DIR / "faiss.index"
METADATA_FILE    = INDEX_DIR / "metadata.pkl"
BM25_FILE        = INDEX_DIR / "bm25.pkl"

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ─────────────────────────────────────────────
# Irrelevant / casual query detection
# ─────────────────────────────────────────────

class IrrelevantQueryError(Exception):
    """Raised when the query is casual, greeting, or unrelated to BIS standards."""
    def __init__(self, message: str):
        super().__init__(message)
        self.user_message = message


# ── Minimal keyword fallback (used only when Groq key is absent) ───────────────

_GREETINGS = {
    "hi", "hello", "hey", "helo", "hii", "hiii", "howdy", "yo", "sup",
    "good morning", "good afternoon", "good evening", "good night",
    "greetings", "salutations", "namaste", "namaskar",
}

_DOMAIN_HINTS = {
    "bis", "is", "standard", "cement", "steel", "pipe", "water", "concrete",
    "quality", "product", "material", "specification", "manufacture", "rubber",
    "textile", "food", "chemical", "paint", "wire", "cable", "brick", "tile",
    "glass", "plastic", "wood", "timber", "copper", "aluminium", "aluminum",
    "iron", "coal", "oil", "gas", "pressure", "safety", "weight", "dimension",
    "mse", "factory", "industrial", "bureau", "indian", "national", "grade",
    "class", "type", "requirement", "compressive", "tensile", "strength",
}

_NOT_RELEVANT_MSG = (
    "I\'m specialized in finding **BIS Standards** for products and manufacturing. 🏭\n\n"
    "Your query doesn\'t seem related to Indian Standards compliance. "
    "Please describe your product or manufacturing requirement.\n\n"
    "Example: *We manufacture OPC cement — which BIS standard applies?*"
)


def _keyword_relevance_check(query: str) -> tuple[bool, str]:
    """
    Fast fallback check used when Groq key is unavailable.
    Only catches obvious greetings and queries with zero domain keywords.
    """
    q = query.strip().lower()
    q = re.sub(r"[!?.,\'\"]+$", "", q).strip()
    if q in _GREETINGS:
        return False, (
            "👋 Hello! I\'m the **BIS Standards Finder**.\n\n"
            "Describe your product or compliance requirement and I\'ll find "
            "the relevant Indian Standards for you.\n\n"
            "Example: *We manufacture OPC cement — which BIS standard applies?*"
        )
    words = q.split()
    if not any(w in _DOMAIN_HINTS for w in words):
        return False, _NOT_RELEVANT_MSG
    return True, ""


def check_relevance_and_expand_query(query: str, groq_api_key: Optional[str] = None) -> tuple[bool, str, str]:
    """
    Single Groq call that BOTH checks relevance AND expands the query.
    Returns (is_relevant, expanded_query_or_empty, user_message_if_irrelevant).

    Previously this was two sequential Groq calls:
        1. check_query_relevance_with_groq()
        2. expand_query_with_groq()
    Merging them saves one full round-trip (~300–600 ms).

    Falls back to keyword relevance check + no expansion if Groq key is missing or call fails.
    """
    api_key = groq_api_key or os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        relevant, msg = _keyword_relevance_check(query)
        return relevant, query, msg

    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        system_prompt = (
            "You are an assistant for a BIS (Bureau of Indian Standards) search engine "
            "that helps Indian businesses find relevant IS standards for their products.\n\n"
            "Given a user query, do TWO things in ONE response:\n"
            "1. Decide if the query is relevant to BIS/IS standards, product compliance, "
            "manufacturing requirements, or material specifications.\n"
            "2. If relevant, expand the query with: relevant IS standard numbers, technical "
            "synonyms, abbreviations (OPC, PPC, PSC, HAC, AAC, TMT, etc.), and related "
            "material/process terms.\n\n"
            "Reply with ONLY valid JSON, no markdown, no extra text:\n"
            "{\n"
            '  "relevant": true,\n'
            '  "expanded_query": "<original query> <space-separated expansion keywords>",\n'
            '  "message": ""\n'
            "}\n"
            "OR if not relevant:\n"
            "{\n"
            '  "relevant": false,\n'
            '  "expanded_query": "",\n'
            '  "message": "<one short friendly sentence explaining you only handle BIS standards>"\n'
            "}"
        )

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            max_tokens=160,
            temperature=0.0
        )

        import json as _json
        text = response.choices[0].message.content.strip()
        # Strip accidental markdown fences
        text = re.sub(r'^```json\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        data = _json.loads(text)

        if data.get("relevant", True):
            expanded = data.get("expanded_query", "") or query
            print(f"[Retriever] Combined check+expand: relevant=True, expanded={expanded[:100]}...")
            return True, expanded, ""
        else:
            msg_raw = data.get("message", "")
            user_msg = (
                f"🏭 {msg_raw}\n\n"
                "Try something like: *We manufacture steel pipes — which BIS standard applies?*"
            ) if msg_raw else _NOT_RELEVANT_MSG
            print("[Retriever] Combined check+expand: relevant=False")
            return False, "", user_msg

    except Exception as e:
        print(f"[Retriever] Combined Groq call failed ({e}). Using keyword fallback + no expansion.")
        relevant, msg = _keyword_relevance_check(query)
        return relevant, query, msg


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    return re.findall(r'\b\w+\b', text.lower())


# ─────────────────────────────────────────────
# HF Inference API encoder
# ─────────────────────────────────────────────

def encode_with_hf_api(
    texts: List[str],
    model_id: str = "BAAI/bge-large-en-v1.5",
    api_key: str = "",
    batch_size: int = 64,
) -> np.ndarray:
    """
    Encode texts using HuggingFace Inference API in batches.
    Free tier: ~1000 requests/day. Each request = one batch of 64 texts.
    At 5944 child chunks / 64 per batch = ~93 API calls total.
    """
    import requests

    api_key = api_key or HF_API_KEY
    if not api_key:
        raise ValueError(
            "HF_API_KEY not set. Get a free key at https://huggingface.co/settings/tokens "
            "and set env var HF_API_KEY, or switch EMBEDDING_MODE to 'local_fast'."
        )

    url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    all_embeddings = []
    total = len(texts)
    total_batches = (total + batch_size - 1) // batch_size
    print(f"[Retriever] HF API encoding {total} texts ({total_batches} batches)...")

    for i in range(0, total, batch_size):
        batch = texts[i: i + batch_size]
        for attempt in range(3):
            try:
                resp = requests.post(
                    url,
                    headers=headers,
                    json={"inputs": batch, "options": {"wait_for_model": True}},
                    timeout=60,
                )
                if resp.status_code == 200:
                    batch_emb = np.array(resp.json(), dtype=np.float32)
                    if batch_emb.ndim == 3:
                        batch_emb = batch_emb[:, 0, :]  # CLS token pooling
                    all_embeddings.append(batch_emb)
                    print(f"  Batch {i // batch_size + 1}/{total_batches} done")
                    break
                elif resp.status_code == 503:
                    wait = int(resp.json().get("estimated_time", 20))
                    print(f"  Model loading on HF, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"HF API {resp.status_code}: {resp.text[:200]}")
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"  Attempt {attempt+1} failed ({e}), retrying in 5s...")
                time.sleep(5)

    embeddings = np.vstack(all_embeddings)
    # L2 normalize for cosine / inner product FAISS search
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-9, None)
    return embeddings


# ─────────────────────────────────────────────
# BISRetriever
# ─────────────────────────────────────────────

class BISRetriever:
    def __init__(self, groq_api_key: Optional[str] = None):
        self.embed_model = None
        self.reranker    = None
        self.faiss_index = None
        self.bm25: Optional[BM25Okapi] = None
        self.metadata: List[Dict]      = []
        self.parent_map: Dict[str, Dict] = {}
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY", "")

    # ── Encoding router ────────────────────────────────────────────────────────

    def _encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """
        Route to local SentenceTransformer or HF Inference API.
        is_query=True prepends BGE query prefix (query time only, not indexing).
        """
        if is_query:
            texts = [BGE_QUERY_PREFIX + t for t in texts]

        if EMBEDDING_MODE == "hf_api":
            return encode_with_hf_api(texts, model_id=EMBED_MODEL_NAME)

        # Local modes
        if self.embed_model is None:
            from sentence_transformers import SentenceTransformer
            print(f"[Retriever] Loading embedding model: {EMBED_MODEL_NAME}")
            self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)

        batch_size = 64 if EMBEDDING_MODE == "local_fast" else 32
        embeddings = self.embed_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return np.array(embeddings, dtype=np.float32)

    # ── Build Index ────────────────────────────────────────────────────────────

    def build_index(self, chunks_data: Dict):
        """Build FAISS + BM25 index from parsed chunks."""
        parent_chunks: List[Dict] = chunks_data["parent_chunks"]
        child_chunks:  List[Dict] = chunks_data["child_chunks"]

        self.parent_map = {c["standard_id"]: c for c in parent_chunks}
        index_chunks    = child_chunks if child_chunks else parent_chunks
        self.metadata   = index_chunks

        print(f"[Retriever] Indexing {len(index_chunks)} chunks "
              f"(mode={EMBEDDING_MODE}, model={EMBED_MODEL_NAME})")

        corpus_texts     = [c["text"] for c in index_chunks]
        tokenized_corpus = [_tokenize(t) for t in corpus_texts]
        self.bm25        = BM25Okapi(tokenized_corpus)

        print(f"[Retriever] Encoding {len(corpus_texts)} texts...")
        embeddings = self._encode(corpus_texts, is_query=False)

        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(embeddings)

        print(f"[Retriever] FAISS index built: {self.faiss_index.ntotal} vectors, dim={dim}")
        self._save_index()

    def _save_index(self):
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.faiss_index, str(FAISS_INDEX_FILE))
        with open(METADATA_FILE, "wb") as f:
            pickle.dump((self.metadata, self.parent_map), f)
        with open(BM25_FILE, "wb") as f:
            pickle.dump(self.bm25, f)
        print(f"[Retriever] Index saved to {INDEX_DIR}")

    def load_index(self):
        if not FAISS_INDEX_FILE.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {FAISS_INDEX_FILE}. Run build_index.py first."
            )
        print("[Retriever] Loading index from disk...")
        self.faiss_index = faiss.read_index(str(FAISS_INDEX_FILE))
        with open(METADATA_FILE, "rb") as f:
            self.metadata, self.parent_map = pickle.load(f)
        with open(BM25_FILE, "rb") as f:
            self.bm25 = pickle.load(f)
        print(f"[Retriever] Loaded {self.faiss_index.ntotal} vectors, "
              f"{len(self.metadata)} metadata entries.")

    def _load_models_if_needed(self):
        if RERANKER_MODE in ("local_small", "local_large") and self.reranker is None:
            from sentence_transformers import CrossEncoder
            model_name = RERANKER_MODELS[RERANKER_MODE]
            print(f"[Retriever] Loading reranker: {model_name}")
            self.reranker = CrossEncoder(model_name, max_length=512)

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def _dense_retrieve(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        query_embed = self._encode([query], is_query=True)
        scores, indices = self.faiss_index.search(query_embed, top_k)
        return list(zip(indices[0].tolist(), scores[0].tolist()))

    def _sparse_retrieve(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        # query is already expanded by check_relevance_and_expand_query() upstream
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]

    def _rrf_fusion(
        self,
        dense_results:  List[Tuple[int, float]],
        sparse_results: List[Tuple[int, float]],
        k: int = 60
    ) -> List[Tuple[int, float]]:
        scores: Dict[int, float] = {}
        for rank, (idx, _) in enumerate(dense_results):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
        for rank, (idx, _) in enumerate(sparse_results):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def _rerank(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        if not candidates:
            return candidates
        pairs = [(query, c.get("parent_text", c["text"])) for c in candidates]
        rerank_scores = self.reranker.predict(pairs)
        scored = sorted(zip(rerank_scores, candidates), key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        import time as _time
        _t0 = _time.perf_counter()

        # Single merged Groq call: relevance check + query expansion together.
        relevant, expanded_query, msg = check_relevance_and_expand_query(query, self.groq_api_key)
        print(f"[TIMER] groq_relevance_expand : {(_time.perf_counter()-_t0)*1000:.1f}ms")
        if not relevant:
            raise IrrelevantQueryError(msg)

        self._load_models_if_needed()

        _t = _time.perf_counter()
        dense_res  = self._dense_retrieve(expanded_query, top_k=20)
        print(f"[TIMER] dense_retrieve        : {(_time.perf_counter()-_t)*1000:.1f}ms")

        _t = _time.perf_counter()
        sparse_res = self._sparse_retrieve(expanded_query, top_k=20)
        print(f"[TIMER] sparse_retrieve       : {(_time.perf_counter()-_t)*1000:.1f}ms")

        _t = _time.perf_counter()
        fused = self._rrf_fusion(dense_res, sparse_res)
        print(f"[TIMER] rrf_fusion            : {(_time.perf_counter()-_t)*1000:.1f}ms")

        candidates   = []
        seen_indices = set()
        for idx, score in fused[:5]:
            if idx < 0 or idx in seen_indices:
                continue
            seen_indices.add(idx)
            chunk = dict(self.metadata[idx])
            chunk["rrf_score"] = score
            candidates.append(chunk)

        _t = _time.perf_counter()
        reranked = self._rerank(query, candidates, top_k=top_k * 2)
        print(f"[TIMER] crossencoder_rerank   : {(_time.perf_counter()-_t)*1000:.1f}ms")
        print(f"[TIMER] retrieve() TOTAL      : {(_time.perf_counter()-_t0)*1000:.1f}ms")

        seen_ids = set()
        results  = []
        for chunk in reranked:
            sid = chunk["standard_id"]
            if sid in seen_ids:
                continue
            seen_ids.add(sid)
            parent = self.parent_map.get(sid, chunk)
            results.append({
                "standard_id":  sid,
                "title":        parent.get("title", chunk.get("title", "")),
                "text":         parent.get("text",  chunk.get("text",  "")),
                "page_number":  parent.get("page_number", chunk.get("page_number", 0)),
            })
            if len(results) >= top_k:
                break

        return results


# ── Singleton ──────────────────────────────────────────────────────────────────

_retriever_instance: Optional[BISRetriever] = None

def get_retriever() -> BISRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = BISRetriever()
        _retriever_instance.load_index()
    return _retriever_instance