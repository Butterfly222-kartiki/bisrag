"""
build_index.py — Run this ONCE to build the FAISS + BM25 index.

Usage (pre-generated chunks — recommended):
    python build_index.py --chunks data/bis_all_chunks.json

Usage (original PDF flow — requires PyMuPDF/pdfplumber):
    python build_index.py --pdf path/to/bis_sp21.pdf

This creates:
    data/chunks.json         — parsed standard chunks (parent + child)
    data/index/faiss.index   — FAISS dense index
    data/index/metadata.pkl  — chunk metadata
    data/index/bm25.pkl      — BM25 sparse index
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(description="Build BIS RAG index")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--chunks",
        help="Path to pre-generated chunks JSON (e.g., data/bis_all_chunks.json)"
    )
    group.add_argument(
        "--pdf",
        help="Path to BIS SP21 PDF file (e.g., dataset.pdf)"
    )
    args = parser.parse_args()

    if args.chunks:
        chunks_path = Path(args.chunks)
        if not chunks_path.exists():
            print(f"[ERROR] Chunks file not found: {chunks_path}")
            sys.exit(1)
        from src.pipeline import build_index_from_chunks
        build_index_from_chunks(str(chunks_path))

    else:
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            print(f"[ERROR] PDF not found: {pdf_path}")
            sys.exit(1)
        from src.pipeline import build_index_from_pdf
        build_index_from_pdf(str(pdf_path))

    print("\n✅ Index built successfully!")
    print("   You can now run the API server with:")
    print("   python run_server.py")
    print("\n   Or run inference with:")
    print("   python inference.py --input data/public_test_set.json --output data/results.json")


if __name__ == "__main__":
    main()