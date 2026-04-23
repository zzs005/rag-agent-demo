from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from dotenv import load_dotenv

from app.qa.answerer import StructuredAnswerer
from app.retrieval.search_service import SearchService


load_dotenv()

VECTORS_PATH = ROOT / "data" / "index" / "chunks_vectors"
METADATA_PATH = ROOT / "data" / "index" / "chunks_metadata.json"


def print_results(title: str, results) -> None:
    print(f"\n[{title}]")
    print("-" * 100)
    for i, item in enumerate(results, start=1):
        preview = item.text[:220].replace("\n", " ")
        print(f"Top {i}")
        print(f"score       : {item.score:.4f}")
        print(f"source      : {item.source}")
        print(f"page        : {item.page_start}-{item.page_end}")
        print(f"section     : {item.section_title}")
        print(f"chunk_id    : {item.chunk_id}")
        print(f"text preview: {preview}")
        print("-" * 100)


def main() -> None:
    metadata_exists = METADATA_PATH.exists()
    vectors_exists = VECTORS_PATH.with_suffix(".npy").exists()

    if not metadata_exists or not vectors_exists:
        print("Index files not found. Please run build_index.py first.")
        return

    top_k = int(os.getenv("TOP_K", "5"))

    search_service = SearchService(
        vectors_path=str(VECTORS_PATH),
        metadata_path=str(METADATA_PATH),
    )
    answerer = StructuredAnswerer()

    print("Enter your question. Type 'exit' to quit.\n")

    while True:
        query = input("Query> ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        results = search_service.search(query=query, top_k=top_k)
        response = answerer.answer(query=query, chunks=results)

        print("\n" + "=" * 100)
        print(f"Query: {query}")
        print("=" * 100)

        print("\n[Structured Response]")
        print(json.dumps(response.to_dict(), ensure_ascii=False, indent=2))

        print_results("Retrieved Chunks", results)
        print()


if __name__ == "__main__":
    main()