"""
inference.py — Mandatory entry point for BIS Hackathon judges.

Usage:
    python inference.py --input public_test_set.json --output results.json

Input JSON schema (with or without expected_standards):
    [{"id": "...", "query": "...", "expected_standards": [...]}]
    or
    [{"id": "...", "query": "..."}]

Output JSON schema:
    [{"id": "...", "query": "...", "expected_standards": [...], "retrieved_standards": [...], "latency_seconds": ...}]
"""

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
    parser.add_argument("--input",  required=True, help="Path to input JSON file with queries")
    parser.add_argument("--output", required=True, help="Path to output JSON file for results")
    parser.add_argument("--top_k",  default=5, type=int, help="Number of standards to retrieve")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)

    with open(input_path, encoding="utf-8") as f:
        queries = json.load(f)

    print(f"[Inference] Loaded {len(queries)} queries from {args.input}")
    print("[Inference] Initializing BIS Recommender...")
    recommender = BISRecommender()

    results = []
    for i, item in enumerate(queries):
        query_id   = item.get("id", f"Q-{i:03d}")
        query_text = item.get("query", "")
        expected   = item.get("expected_standards", [])   # empty list if not in input

        print(f"[Inference] ({i+1}/{len(queries)}) {query_id}: {query_text[:70]}...")

        start = time.time()
        try:
            raw = recommender.retrieve(query_text, top_k=args.top_k)
            # retrieve() returns list of dicts — extract just the standard_id strings
            standard_ids = [
                r["standard_id"] if isinstance(r, dict) else r
                for r in raw
            ]
        except IrrelevantQueryError:
            standard_ids = []
        except Exception as e:
            print(f"  [ERROR] {e}")
            standard_ids = []

        latency = round(time.time() - start, 3)

        results.append({
            "id":                   query_id,
            "query":                query_text,
            "expected_standards":   expected,
            "retrieved_standards":  standard_ids,
            "latency_seconds":      latency
        })

        print(f"  → {standard_ids} ({latency}s)")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    avg_latency = sum(r["latency_seconds"] for r in results) / len(results) if results else 0
    print(f"\n[Inference] Done! {len(results)} results saved to {args.output}")
    print(f"[Inference] Average latency: {avg_latency:.3f}s")
    print(f"[Inference] Run eval: python eval_script.py --results {args.output}")


if __name__ == "__main__":
    main()