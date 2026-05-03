# BIS Standards RAG Engine — v2 (Fixed)

AI-powered retrieval system for BIS SP 21 standards. Recommends relevant IS standards
for any product/material query using hybrid dense + sparse retrieval with cross-encoder reranking.

## What Was Fixed (v2)

This version addresses 7 root causes that caused ~57% accuracy in v1:

| Fix | Root Cause | Change | Expected Gain |
|-----|-----------|--------|---------------|
| 1 | Dedup kept TOC stub (first), discarding real content | Keep **longest** chunk per standard_id | ~15-20% |
| 2 | TOC content in title field poisoned all embeddings | Detect & strip TOC-contaminated titles (>200 chars or inline IS-number lists) | Embedding quality |
| 3 | Child chunks were single decontextualized sentences | 4-sentence sliding window + prepend `Standard: IS XXX — Title.` prefix to every child | Embedding precision |
| 4 | `bge-base-en-v1.5` (768-dim) used for embeddings | Upgraded to `bge-large-en-v1.5` (1024-dim) | +4-6% |
| 5 | `ms-marco-MiniLM-L-6-v2` reranker (web-trained, 6-layer) | Upgraded to `BAAI/bge-reranker-large` | +3-5% |
| 6 | Reranker scored child sentence fragment (30 words) | Reranker now scores full **parent text** for reliable relevance judgment | Score reliability |
| 7 | BM25 used raw query tokens — vocabulary mismatch | Groq LLM expands query with synonyms, IS numbers, abbreviations before BM25 | +5-8% BM25 recall |

---

## Project Structure

```
bis_rag_project/
├── src/
│   ├── parser.py       # PDF parsing + chunking (Fix 1, 2, 3)
│   ├── retriever.py    # Hybrid retrieval + reranking (Fix 4, 5, 6, 7)
│   ├── pipeline.py     # Orchestrator
│   ├── llm.py          # Gemini rationale generation
│   └── api.py          # FastAPI server
├── frontend/
│   └── index.html      # Web UI
├── data/
│   ├── public_test_set.json
│   └── sample_output.json
├── build_index.py      # Run once to build index
├── inference.py        # Hackathon evaluation entry point
├── run_server.py       # Start web server
├── eval_script.py      # Local evaluation
└── requirements.txt
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `bge-large-en-v1.5` and `bge-reranker-large` are larger than the v1 models.
> Ensure at least **8 GB RAM** for indexing, **4 GB** for inference.

### 2. Set API keys

```bash
# Required for rationale generation
export GEMINI_API_KEY="your-gemini-api-key"

# Required for query expansion (Fix 7). Free tier is sufficient.
# Get key at: https://console.groq.com
export GROQ_API_KEY="your-groq-api-key"
```

> **Without `GROQ_API_KEY`:** Query expansion is silently skipped and the raw query
> is used for BM25. The system still works — Fix 7 is just inactive.

### 3. Build the index (run once)

```bash
python build_index.py --pdf path/to/dataset.pdf
```

This creates:
- `data/chunks.json` — parsed standards chunks
- `data/index/faiss.index` — FAISS dense index (bge-large, 1024-dim)
- `data/index/metadata.pkl` — chunk metadata + parent map
- `data/index/bm25.pkl` — BM25 sparse index

**Expected output:**
```
[Parser] Using PyMuPDF...
[Parser] Dedup (keep-longest): 440 raw → 380 unique → 352 valid (dropped 28 stubs)
[Parser] Final: 352 parent chunks, 2800+ child chunks.
[Retriever] Encoding ... with BAAI/bge-large-en-v1.5...
[Retriever] FAISS index built: 2800+ vectors, dim=1024
✅ Index built successfully!
```

---

## Running Inference (Hackathon Evaluation)

```bash
python inference.py --input data/public_test_set.json --output data/results.json
```

The output JSON format:
```json
[
  {
    "id": "Q-001",
    "retrieved_standards": ["IS 269: 1989", "IS 455: 1989", "IS 1489: 1991"],
    "latency_seconds": 0.842
  }
]
```

---

## Running the Web Server

```bash
python run_server.py
# or with custom port:
python run_server.py --port 8080
```

Open **http://localhost:8000** for the frontend.  
API docs: **http://localhost:8000/docs**

### API Usage

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Portland slag cement chemical requirements",
    "top_k": 5
  }'
```

You can also pass API keys per-request (useful for demos):
```json
{
  "query": "Portland slag cement",
  "top_k": 5,
  "gemini_api_key": "...",
  "groq_api_key": "..."
}
```

---

## Local Evaluation

```bash
python eval_script.py \
  --predictions data/results.json \
  --ground_truth data/public_test_set.json
```

---

## Key Changes vs v1 (Developer Notes)

### parser.py
- `_dedup_keep_longest()` replaces the old `seen = set()` dedup block
- `is_toc_title()` detects contaminated titles (>200 chars or `1.11 IS 455` patterns)
- `create_child_chunks()` uses 4-sentence sliding window with step=2, and prepends
  `"Standard: IS XXX — Title. "` to every child before embedding

### retriever.py
- `EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"` (was `bge-base`)
- `RERANKER_MODEL_NAME = "BAAI/bge-reranker-large"` (was `ms-marco-MiniLM-L-6-v2`)
- `_sparse_retrieve()` calls `expand_query_with_groq()` before tokenisation
- `_rerank()` uses `c.get("parent_text", c["text"])` instead of `c["text"]`

### requirements.txt
- Added `groq>=0.9.0`

---

## Environment Variables Reference

| Variable | Required | Purpose |
|----------|----------|---------|
| `GEMINI_API_KEY` | Yes (for rationales) | Gemini 2.5 Flash rationale generation |
| `GROQ_API_KEY` | Recommended | BM25 query expansion via Llama3 (Fix 7) |

---

## Troubleshooting

**`Index not found` error on inference:**
```bash
python build_index.py --pdf dataset.pdf
```

**OOM during indexing (bge-large is bigger):**
Reduce batch size in `retriever.py`: change `batch_size=32` to `batch_size=16`.

**Groq rate limit during batch inference:**
Query expansion adds ~1 API call per query. Groq free tier allows 30 req/min.
For large batches, set `GROQ_API_KEY=""` to skip expansion and use raw queries.

**Standards still missing from results:**
Check `data/chunks.json` for the standard_id. If it's absent, the PDF parsing
didn't extract it — the standard may be in a scanned/image-only page that
PyMuPDF can't text-extract.
