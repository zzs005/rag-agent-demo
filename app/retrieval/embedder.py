from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


load_dotenv()


class LocalEmbedder:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
        self.model = SentenceTransformer(self.model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return vectors.tolist()

    def embed_query(self, query: str) -> List[float]:
        vector = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector[0].tolist()