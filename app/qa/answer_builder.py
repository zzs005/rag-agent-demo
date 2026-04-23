from __future__ import annotations

from typing import List

from app.retrieval.schemas import RetrievedChunk


class TemplateAnswerBuilder:
    def __init__(self, preview_chars: int = 180):
        self.preview_chars = preview_chars

    def build_answer(self, query: str, chunks: List[RetrievedChunk]) -> str:
        if not chunks:
            return "No grounded evidence was found in the knowledge base yet."

        top1 = chunks[0]
        summary1 = self._clean_preview(top1.text, self.preview_chars)

        lines = [
            f"The most relevant evidence for '{query}' comes from '{top1.source}'",
            f"(page {top1.page_start or '?'}).",
            f"Key evidence: {summary1}",
        ]

        if len(chunks) > 1:
            top2 = chunks[1]
            summary2 = self._clean_preview(top2.text, min(120, self.preview_chars))
            lines.append(
                f"Additional evidence from '{top2.source}' (page {top2.page_start or '?'}): {summary2}"
            )

        lines.append("This answer is grounded only in the currently retrieved knowledge base content.")
        return "\n".join(lines)

    @staticmethod
    def _clean_preview(text: str, max_chars: int) -> str:
        text = text.replace("\n", " ").strip()
        text = " ".join(text.split())
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "..."
