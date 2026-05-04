"""
index_builder.py
================
Owns everything related to building, persisting, and loading the two-part
retrieval index used by BISRetriever:

    Dense index  — FAISS IndexFlatIP over BGE sentence embeddings.
    Sparse index — BM25Okapi from the rank_bm25 library.

This module is NOT in the hot path for query serving.  It runs once (or
whenever the corpus changes) via build_index.py / pipeline.py, then the
serialised files are loaded at server startup.

Public surface used by retriever.py:
    EMBEDDING_MODE   — current embedding backend
    EMBED_MODEL_NAME — resolved model ID string
    EmbeddingEncoder — wraps local ST model or HF API
    IndexStore       — build / save / load the FAISS + BM25 + metadata bundle
    tokenize         — shared BM25 tokeniser helper
"""

import os
import pickle
import re
import time
import numpy as np
from pathlib import Path
from typing import List, Dict

from rank_bm25 import BM25Okapi
import faiss


# ---------------------------------------------------------------------------
# Embedding mode configuration — change EMBEDDING_MODE to switch backend.
# ---------------------------------------------------------------------------

EMBEDDING_MODE = "local_fast"   # "local_fast" | "local_large" | "hf_api"

# Maps each mode to its HuggingFace model ID.
EMBED_MODELS = {
    "local_fast":  "all-MiniLM-L6-v2",   
    "local_large": "BAAI/bge-large-en-v1.5", 
    "hf_api":      "BAAI/bge-large-en-v1.5",   
}

EMBED_MODEL_NAME = EMBED_MODELS[EMBEDDING_MODE]

# HF API key — only required when EMBEDDING_MODE = "hf_api".
HF_API_KEY = os.environ.get("HF_API_KEY", "")

# BGE models improve recall when query strings carry this prefix.
# Applied at retrieval time only — NOT during index building.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Filesystem paths where index artefacts are written / read from.
INDEX_DIR        = Path("data/index")
FAISS_INDEX_FILE = INDEX_DIR / "faiss.index"
METADATA_FILE    = INDEX_DIR / "metadata.pkl"
BM25_FILE        = INDEX_DIR / "bm25.pkl"


# ---------------------------------------------------------------------------
# Tokeniser helper — shared between BM25 indexing and query tokenisation.
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """
    Simple word-level tokeniser for BM25.
    Lowercases and extracts alphanumeric tokens; ignores punctuation.
    """
    return re.findall(r'\b\w+\b', text.lower())


# ---------------------------------------------------------------------------
# HuggingFace Inference API encoder
# Used only when EMBEDDING_MODE = "hf_api" (no local GPU available).
# ---------------------------------------------------------------------------

def encode_with_hf_api(
    texts: List[str],
    model_id: str = "BAAI/bge-large-en-v1.5",
    api_key: str = "",
    batch_size: int = 64,
) -> np.ndarray:
    """
    Send texts to the HuggingFace Inference API and return L2-normalised
    embeddings as a float32 numpy array of shape (N, dim).

    Free tier allows ~1000 requests/day.  At 64 texts per request and
    ~5944 child chunks, the full index build costs ~93 API calls.

    Retries each batch up to 3 times to handle transient 503 (model loading)
    responses that the HF API sometimes returns.
    """
    import requests

    api_key = api_key or HF_API_KEY
    if not api_key:
        raise ValueError(
            "HF_API_KEY not set. Get a free key at https://huggingface.co/settings/tokens "
            "and set env var HF_API_KEY, or switch EMBEDDING_MODE to 'local_fast'."
        )

    url     = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
    headers = {"Authorization": f"Bearer {api_key}"}

    all_embeddings = []
    total          = len(texts)
    total_batches  = (total + batch_size - 1) // batch_size
    print(f"[IndexBuilder] HF API encoding {total} texts ({total_batches} batches)...")

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
                        batch_emb = batch_emb[:, 0, :]
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
                print(f"  Attempt {attempt + 1} failed ({e}), retrying in 5s...")
                time.sleep(5)

    embeddings = np.vstack(all_embeddings)

    # L2 normalise so inner-product search equals cosine similarity.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, 1e-9, None)


# ---------------------------------------------------------------------------
# EmbeddingEncoder
# ---------------------------------------------------------------------------

class EmbeddingEncoder:
    """
    Lazy-loading embedding model wrapper.

    The actual SentenceTransformer (or HF API client) is only initialised on
    the first call to encode(), so importing this module has zero overhead.
    """

    def __init__(self):
        # Held as None until first use; avoids loading the model at import time.
        self._model = None

    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """
        Encode a list of texts into L2-normalised float32 embeddings.

        Args:
            texts    : Strings to embed.
            is_query : If True, prepends the BGE query prefix.  Must be True
                       at retrieval time and False during index building.

        Returns:
            numpy array of shape (len(texts), embedding_dim).
        """
        if is_query:
            texts = [BGE_QUERY_PREFIX + t for t in texts]

        if EMBEDDING_MODE == "hf_api":
            return encode_with_hf_api(texts, model_id=EMBED_MODEL_NAME)

        # Local path — load the SentenceTransformer model once and reuse it.
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"[IndexBuilder] Loading embedding model: {EMBED_MODEL_NAME}")
            self._model = SentenceTransformer(EMBED_MODEL_NAME)

        # Smaller batch size for the larger model to avoid OOM on CPU.
        batch_size = 64 if EMBEDDING_MODE == "local_fast" else 32
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,   # Returns L2-normalised vectors.
        )
        return np.array(embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# IndexStore — builds, saves, and loads the FAISS + BM25 + metadata bundle.
# ---------------------------------------------------------------------------

class IndexStore:
    """
    Manages the three artefacts that make up the retrieval index:

        faiss.index   — FAISS IndexFlatIP for dense nearest-neighbour search.
        metadata.pkl  — Chunk metadata list + parent_map dict.
        bm25.pkl      — Serialised BM25Okapi object for sparse search.

    Usage (build once):
        store = IndexStore(encoder)
        store.build(chunks_data)   # writes files to INDEX_DIR

    Usage (load at startup):
        store = IndexStore(encoder)
        store.load()               # reads files from INDEX_DIR
    """

    def __init__(self, encoder: EmbeddingEncoder):
        self.encoder     = encoder
        self.faiss_index = None
        self.bm25        = None
        self.metadata: List[Dict]        = []
        self.parent_map: Dict[str, Dict] = {}

    def build(self, chunks_data: Dict):
        """
        Build and persist the FAISS + BM25 index from parsed chunk data.

        chunks_data must contain:
            "parent_chunks": list of full-text standard entries.
            "child_chunks" : list of sliding-window sub-entries used for indexing.

        The FAISS index is built over child chunks (finer granularity for dense
        retrieval) while parent chunks are stored in parent_map for result
        hydration after reranking.
        """
        parent_chunks: List[Dict] = chunks_data["parent_chunks"]
        child_chunks:  List[Dict] = chunks_data["child_chunks"]

        # Keep parent chunks indexed by standard_id for fast result hydration.
        self.parent_map = {c["standard_id"]: c for c in parent_chunks}

        # Prefer child chunks for indexing; fall back to parents if none exist.
        index_chunks  = child_chunks if child_chunks else parent_chunks
        self.metadata = index_chunks

        print(
            f"[IndexBuilder] Indexing {len(index_chunks)} chunks "
            f"(mode={EMBEDDING_MODE}, model={EMBED_MODEL_NAME})"
        )

        # ── BM25 index ──────────────────────────────────────────────────────
        corpus_texts     = [c["text"] for c in index_chunks]
        tokenized_corpus = [tokenize(t) for t in corpus_texts]
        self.bm25        = BM25Okapi(tokenized_corpus)

        # ── FAISS index ─────────────────────────────────────────────────────
        print(f"[IndexBuilder] Encoding {len(corpus_texts)} texts...")
        embeddings = self.encoder.encode(corpus_texts, is_query=False)

        dim              = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)   # Inner product = cosine on L2-normed vecs.
        self.faiss_index.add(embeddings)

        print(
            f"[IndexBuilder] FAISS index built: "
            f"{self.faiss_index.ntotal} vectors, dim={dim}"
        )
        self._save()

    def _save(self):
        """Persist all three index artefacts to INDEX_DIR."""
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.faiss_index, str(FAISS_INDEX_FILE))
        with open(METADATA_FILE, "wb") as f:
            pickle.dump((self.metadata, self.parent_map), f)
        with open(BM25_FILE, "wb") as f:
            pickle.dump(self.bm25, f)
        print(f"[IndexBuilder] Index saved to {INDEX_DIR}")

    def load(self):
        """
        Load pre-built index artefacts from disk into memory.

        Raises FileNotFoundError if the index has not been built yet.
        Called once at server startup; all subsequent requests read from RAM.
        """
        if not FAISS_INDEX_FILE.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {FAISS_INDEX_FILE}. Run build_index.py first."
            )

        print("[IndexBuilder] Loading index from disk...")
        self.faiss_index = faiss.read_index(str(FAISS_INDEX_FILE))
        with open(METADATA_FILE, "rb") as f:
            self.metadata, self.parent_map = pickle.load(f)
        with open(BM25_FILE, "rb") as f:
            self.bm25 = pickle.load(f)

        print(
            f"[IndexBuilder] Loaded {self.faiss_index.ntotal} vectors, "
            f"{len(self.metadata)} metadata entries."
        )
