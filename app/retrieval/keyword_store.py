from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Dict, Any

from app.retrieval.schemas import RetrievedChunk


def simple_tokenize(text: str) -> List[str]:
    text = text.lower().strip()
    # 中英文混合的非常简化版本
    parts = re.split(r"[\s,，。！？；：:\.\(\)（）【】\[\]\-_/]+", text)
    return [p for p in parts if p]


class KeywordStore:
    def __init__(self):
        self.metadata: List[Dict[str, Any]] = []

    def build(self, metadata: List[Dict[str, Any]]) -> None:
        self.metadata = metadata

    def save(self, metadata_path: str) -> None:
        Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def load(self, metadata_path: str) -> None:
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def search(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        query_terms = simple_tokenize(query)
        if not query_terms:
            return []

        scored: List[tuple[float, Dict[str, Any]]] = []

        for item in self.metadata:
            text = item["text"].lower()
            score = 0.0

            for term in query_terms:
                if not term:
                    continue
                # 出现一次记 1 分；你也可以改成 count(term)
                if term in text:
                    score += 1.0

            if score > 0:
                # 简单归一化
                norm_score = score / max(len(query_terms), 1)
                scored.append((norm_score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_items = scored[:top_k]

        results: List[RetrievedChunk] = []
        for score, item in top_items:
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
                    score=float(score),
                )
            )
        return results