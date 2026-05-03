"""
inference.py — Mandatory entry point for BIS Hackathon judges.

Usage:
    python inference.py --input hidden_private_dataset.json --output team_results.json

The input JSON must have this schema:
    [{"id": "...", "query": "..."}, ...]

The output JSON will have:
    [{"id": "...", "retrieved_standards": [...], "latency_seconds": ...}, ...]
"""

import json
import argparse
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import BISRecommender


def main():
    parser = argparse.ArgumentParser(
        description="BIS Standards RAG Inference Script"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input JSON file with queries"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output JSON file for results"
    )
    args = parser.parse_args()

    # Load input queries
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)

    with open(input_path) as f:
        queries = json.load(f)

    print(f"[Inference] Loaded {len(queries)} queries from {args.input}")

    # Initialize recommender (loads FAISS index from disk)
    print("[Inference] Initializing BIS Recommender...")
    recommender = BISRecommender()

    # Run inference
    results = []
    for i, item in enumerate(queries):
        query_id = item.get("id", f"Q-{i:03d}")
        query_text = item.get("query", "")

        print(f"[Inference] ({i+1}/{len(queries)}) Processing: {query_id}")

        start = time.time()
        try:
            standard_ids = recommender.retrieve(query_text, top_k=5)
        except Exception as e:
            print(f"[Inference] Error on {query_id}: {e}")
            standard_ids = []

        latency = round(time.time() - start, 3)

        results.append({
            "id": query_id,
            "retrieved_standards": standard_ids,
            "latency_seconds": latency
        })

        print(f"  → Retrieved: {standard_ids[:3]}... ({latency}s)")

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Inference] Done! Results saved to {args.output}")

    # Print quick summary
    avg_latency = sum(r["latency_seconds"] for r in results) / len(results) if results else 0
    print(f"[Inference] Average latency: {avg_latency:.3f}s")


if __name__ == "__main__":
    main()
