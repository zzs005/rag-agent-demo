from __future__ import annotations

from typing import List

from app.retrieval.schemas import RetrievedChunk


def build_context(chunks: List[RetrievedChunk]) -> str:
    parts = []

    for i, chunk in enumerate(chunks, start=1):
        section = chunk.section_title or "N/A"
        parts.append(
            f"[Chunk {i}]\n"
            f"source: {chunk.source}\n"
            f"page: {chunk.page_start}-{chunk.page_end}\n"
            f"section: {section}\n"
            f"chunk_id: {chunk.chunk_id}\n"
            f"text:\n{chunk.text}\n"
        )

    return "\n".join(parts)