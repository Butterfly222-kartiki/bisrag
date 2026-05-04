"""
run_server.py -- Start the FastAPI web server.

Usage:
    python run_server.py [--port 8000] [--host 0.0.0.0]

Serves:
    http://localhost:8000/          -- HTML frontend
    http://localhost:8000/docs      -- Swagger API docs
    http://localhost:8000/recommend -- POST endpoint
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(description="Run BIS RAG API server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode only)")
    args = parser.parse_args()

    # If the FAISS index is missing the server will still start, but every
    # request will fail at retrieval time. Better to surface this early so the
    # user knows to run build_index.py before sending any traffic.
    index_path = Path("data/index/faiss.index")
    if not index_path.exists():
        print("WARNING: FAISS index not found. Build it first before starting the server:")
        print("   python build_index.py --pdf dataset.pdf")
        print()

    import uvicorn
    print(f"Starting BIS RAG server at http://{args.host}:{args.port}")
    print(f"   Frontend : http://localhost:{args.port}/")
    print(f"   API Docs : http://localhost:{args.port}/docs")
    print()

    uvicorn.run(
        "src.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()