import json
import argparse
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import BISRecommender
from src.retriever import IrrelevantQueryError


def main():
    parser = argparse.ArgumentParser(description="BIS Standards RAG Inference Script")
    parser.add_argument("--input", required=True, help="Path to input JSON file with queries")
    parser.add_argument("--output", required=True, help="Path to output JSON file for results")
    parser.add_argument("--top_k", default=5, type=int, help="Number of standards to retrieve")
    parser.add_argument("--rerank", action="store_true", help="Enable reranker")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)

    with open(input_path, encoding="utf-8") as f:
        queries = json.load(f)

    print(f"[Inference] Loaded {len(queries)} queries from {args.input}")
    print("[Inference] Initializing BIS Recommender...")

    # ✅ pass rerank flag correctly
    recommender = BISRecommender(use_reranker=args.rerank)

    # 🔥 PROPER WARMUP (fixes first-query latency issue)
    print("[Warmup] Loading index + models...")
    warmup_start = time.time()

    try:
        recommender._ensure_loaded()
        retriever = recommender.retriever

        # 🔥 Force embedding model load + forward pass
        retriever.encoder.encode(["warmup query 1", "warmup query 2"], is_query=True)

        # 🔥 Force FAISS search
        emb = retriever.encoder.encode(["warmup query"], is_query=True)
        retriever.index_store.faiss_index.search(emb, 5)

        # 🔥 Force BM25
        retriever._sparse_retrieve("warmup query")

        # 🔥 Force reranker (if enabled)
        if retriever.use_reranker:
            retriever.reranker.rerank(
                "warmup",
                [{"text": "dummy text"}],
                top_k=1
            )

    except Exception as e:
        print("[Warmup error]", e)

    warmup_time = round(time.time() - warmup_start, 3)
    print(f"[Warmup] Done in {warmup_time}s\n")

    results = []
    latencies = []

    for i, item in enumerate(queries):
        query_id = item.get("id", f"Q-{i:03d}")
        query_text = item.get("query", "")
        expected = item.get("expected_standards", [])

        print(f"\n==============================")
        print(f"[QUERY {i+1}/{len(queries)}] {query_text}")

        start = time.time()

        try:
            raw = recommender.retrieve(query_text, top_k=args.top_k)

            standard_ids = [
                r["standard_id"] if isinstance(r, dict) else r
                for r in raw
            ]

        except IrrelevantQueryError:
            standard_ids = []

        except Exception as e:
            print(f"[ERROR] {e}")
            standard_ids = []

        latency = round(time.time() - start, 3)
        latencies.append(latency)

        print(f"[RESULT] {standard_ids}")
        print(f"[LATENCY] {latency}s")
        print(f"==============================")

        results.append({
            "id": query_id,
            "query": query_text,
            "expected_standards": expected,
            "retrieved_standards": standard_ids,
            "latency_seconds": latency
        })

    # ✅ FIXED average latency (exclude first query if needed)
    if len(latencies) > 1:
        avg_latency = sum(latencies[1:]) / (len(latencies) - 1)
    else:
        avg_latency = latencies[0] if latencies else 0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[Inference] Done! {len(results)} results saved to {args.output}")
    print(f"[Inference] Average latency (steady-state): {avg_latency:.3f}s")
    print(f"[Inference] Warmup time (not counted): {warmup_time}s")
    print(f"[Inference] Run eval: python eval_script.py --results {args.output}")


if __name__ == "__main__":
    main()