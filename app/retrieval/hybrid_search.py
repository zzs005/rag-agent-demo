from __future__ import annotations

from typing import List, Dict

from app.retrieval.schemas import RetrievedChunk


def merge_hybrid_results(
    vector_results: List[RetrievedChunk],
    keyword_results: List[RetrievedChunk],
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3,
    top_k: int = 5,
) -> List[RetrievedChunk]:
    merged: Dict[str, RetrievedChunk] = {}
    score_map: Dict[str, float] = {}

    for item in vector_results:
        merged[item.chunk_id] = item
        score_map[item.chunk_id] = vector_weight * item.score

    for item in keyword_results:
        if item.chunk_id not in merged:
            merged[item.chunk_id] = item
            score_map[item.chunk_id] = 0.0
        score_map[item.chunk_id] += keyword_weight * item.score

    final_items: List[RetrievedChunk] = []
    for chunk_id, item in merged.items():
        final_items.append(
            RetrievedChunk(
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                document_name=item.document_name,
                source=item.source,
                chunk_index=item.chunk_index,
                page_start=item.page_start,
                page_end=item.page_end,
                section_title=item.section_title,
                text=item.text,
                char_count=item.char_count,
                score=float(score_map[chunk_id]),
            )
        )

    final_items.sort(key=lambda x: x.score, reverse=True)
    return final_items[:top_k]