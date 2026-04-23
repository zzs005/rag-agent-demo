from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from app.retrieval.load_chunks import load_all_chunks
from app.retrieval.keyword_store import KeywordStore


PROCESSED_DIR = ROOT / "data" / "processed"
INDEX_DIR = ROOT / "data" / "index"
METADATA_PATH = INDEX_DIR / "chunks_metadata.json"


def main() -> None:
    chunks = load_all_chunks(str(PROCESSED_DIR))
    if not chunks:
        print("No chunks found. Please run ingestion first.")
        return

    store = KeywordStore()
    store.build(metadata=chunks)
    store.save(metadata_path=str(METADATA_PATH))

    print(f"Keyword metadata saved to: {METADATA_PATH}")


if __name__ == "__main__":
    main()