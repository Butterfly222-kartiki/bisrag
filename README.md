# BIS Standards Finder — AI-Powered Compliance Retrieval

An AI-powered system that maps natural language queries to relevant Bureau of Indian Standards (BIS) codes. Eliminates manual search through large documents using a hybrid retrieval pipeline combining semantic understanding with keyword-based matching.

---

## Quick Start

### Step 1 — Environment Setup

Run the setup script once before anything else. It creates the virtual environment, installs all dependencies, and then runs inference automatically.

```bash
chmod +x start.sh && ./start.sh
```

The script handles:

- Python 3.11 version check
- Virtual environment creation (`venv/`)
- Dependency installation from `requirements.txt`
- Inference execution

### Step 2 — Required Command (Hackathon Evaluation)

If the environment is already set up and you only need to re-run inference:

```bash
python inference.py --input hidden_private_dataset.json --output team_results.json
```

This command will:

- Load the prebuilt index
- Process all queries from the input file
- Retrieve the most relevant BIS standards
- Save results to the specified output file

---

## Environment Setup

### 1. Python Version

```
Python 3.11
```

### 2. Create Virtual Environment

```bash
python3.11 -m venv venv
```

Activate on Windows:

```bash
venv\Scripts\activate
```

Activate on Mac/Linux:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Web Interface (Optional)

A hosted interface is available for interactive testing:

```
https://mse-hackathon.web.app/
```

> If the FAISS index is present, the system works directly. If the index is not found, enter a valid API key in the UI input field.

---

## Core Features

- Natural language query to BIS standard mapping
- Hybrid retrieval pipeline:
  - Dense retrieval using embeddings (FAISS)
  - Sparse retrieval using BM25
  - Reciprocal Rank Fusion (RRF)
- Part-number aware filtering (e.g., Part 1, Part 2)
- Batch inference support
- Optimized for low-latency retrieval

---

## System Pipeline

```
1. Query preprocessing and validation
2. Dense retrieval using embeddings
3. Sparse retrieval using BM25
4. Fusion using Reciprocal Rank Fusion (RRF)
5. Part-number prioritization
6. Final result selection
```

---

## Project Structure

```
bisrag/
|
+-- data/
|   +-- index/
|       +-- bm25.pkl
|       +-- faiss.index
|       +-- metadata.pkl
|
+-- src/
|   +-- pipeline.py
|   +-- retriever.py
|   +-- index_builder.py
|   +-- query_preprocessor.py
|   +-- ...
|
+-- inference.py
+-- eval_script.py
+-- build_index.py
+-- requirements.txt
+-- start.sh
+-- README.md
```

---

## Index Setup

Ensure the FAISS index exists at:

```
data/index/faiss.index
```

If missing, rebuild it:

```bash
python build_index.py --chunks data/bis_all_chunks.json
```

---

## Output Format

```json
[
  {
    "id": "Q-001",
    "query": "...",
    "expected_standards": [...],
    "retrieved_standards": [...],
    "latency_seconds": 0.45
  }
]
```

---

## Evaluation

```bash
python eval_script.py --results team_results.json
```

### Metrics

| Metric | Description |
|--------|-------------|
| Hit@K | Fraction of queries where the correct standard appears in the top K results |
| MRR | Mean Reciprocal Rank — average of reciprocal ranks of the first correct result |
| Latency | Average retrieval time per query in seconds |

---

## Notes

- Optimized for steady-state latency; cold start is excluded from benchmarks
- Fully local retrieval — no API required unless explicitly configured
- Designed for robustness on unseen queries