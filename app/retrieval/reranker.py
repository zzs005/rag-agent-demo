from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

from app.retrieval.schemas import RetrievedChunk


load_dotenv()


class LocalReranker:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv(
            "RERANK_MODEL",
            "BAAI/bge-reranker-base"
        )
        self.model = CrossEncoder(self.model_name)

    def rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_k: int = 5,
    ) -> List[RetrievedChunk]:
        if not candidates:
            return []

        pairs = [(query, c.text) for c in candidates]
        scores = self.model.predict(pairs)

        reranked = []
        for chunk, score in zip(candidates, scores):
            reranked.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    document_name=chunk.document_name,
                    source=chunk.source,
                    chunk_index=chunk.chunk_index,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section_title=chunk.section_title,
                    text=chunk.text,
                    char_count=chunk.char_count,
                    score=float(score),
                )
            )

        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_k]