from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

from app.retrieval.schemas import RetrievedChunk


class NumpyVectorStore:
    def __init__(self):
        self.vectors = None
        self.metadata: List[Dict[str, Any]] = []

    @staticmethod
    def _to_numpy(vectors: List[List[float]]) -> np.ndarray:
        arr = np.array(vectors, dtype="float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return arr / norms

    def build(self, vectors: List[List[float]], metadata: List[Dict[str, Any]]) -> None:
        if not vectors:
            raise ValueError("No vectors provided.")
        if len(vectors) != len(metadata):
            raise ValueError("vectors and metadata length mismatch.")

        self.vectors = self._to_numpy(vectors)
        self.metadata = metadata

    def search(self, query_vector: List[float], top_k: int = 5) -> List[RetrievedChunk]:
        if self.vectors is None:
            raise ValueError("Index not built.")

        q = np.array([query_vector], dtype="float32")
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)

        scores = (self.vectors @ q.T).reshape(-1)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[RetrievedChunk] = []
        for idx in top_indices:
            item = self.metadata[int(idx)]
            results.append(
                RetrievedChunk(
                    chunk_id=item["chunk_id"],
                    document_id=item["document_id"],
                    document_name=item["document_name"],
                    source=item["source"],
                    chunk_index=item["chunk_index"],
                    page_start=item.get("page_start"),
                    page_end=item.get("page_end"),
                    section_title=item.get("section_title"),
                    text=item["text"],
                    char_count=item["char_count"],
                    score=float(scores[int(idx)]),
                )
            )
        return results

    def save(self, vectors_path: str, metadata_path: str) -> None:
        if self.vectors is None:
            raise ValueError("Index not built.")

        Path(vectors_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)

        np.save(vectors_path, self.vectors)

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def load(self, vectors_path: str, metadata_path: str) -> None:
        npy_path = vectors_path if vectors_path.endswith(".npy") else f"{vectors_path}.npy"
        self.vectors = np.load(npy_path)

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)