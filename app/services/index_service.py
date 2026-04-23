from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

from app.retrieval.load_chunks import load_all_chunks
from app.retrieval.embedder import LocalEmbedder
from app.retrieval.vector_store import NumpyVectorStore


ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
INDEX_DIR = ROOT / "data" / "index"
VECTORS_PATH = INDEX_DIR / "chunks_vectors"
METADATA_PATH = INDEX_DIR / "chunks_metadata.json"


def rebuild_index() -> Dict[str, Any]:
    chunks = load_all_chunks(str(PROCESSED_DIR))
    if not chunks:
        raise ValueError("No chunks found. Please ingest documents first.")

    texts = [item["text"] for item in chunks]

    embedder = LocalEmbedder()
    vectors = embedder.embed_texts(texts)

    store = NumpyVectorStore()
    store.build(vectors=vectors, metadata=chunks)
    store.save(vectors_path=str(VECTORS_PATH), metadata_path=str(METADATA_PATH))

    return {
        "status": "success",
        "chunks_count": len(chunks),
        "vectors_path": str(VECTORS_PATH) + ".npy",
        "metadata_path": str(METADATA_PATH),
    }