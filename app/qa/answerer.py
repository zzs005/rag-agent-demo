from __future__ import annotations

from typing import List, Optional

from app.qa.answer_builder import TemplateAnswerBuilder
from app.qa.refusal import RuleBasedRefusalJudge
from app.qa.response_formatter import make_grounded_response, make_refusal_response
from app.qa.schemas import Citation, QAResponse
from app.retrieval.schemas import RetrievedChunk


class StructuredAnswerer:
    def __init__(
        self,
        min_top1_score: float = 0.58,
        min_avg_top3_score: float = 0.52,
        min_score_gap: float = 0.03,
        preview_chars: int = 180,
    ):
        self.refusal_judge = RuleBasedRefusalJudge(
            min_top1_score=min_top1_score,
            min_avg_top3_score=min_avg_top3_score,
            min_score_gap=min_score_gap,
        )
        self.answer_builder = TemplateAnswerBuilder(preview_chars=preview_chars)

    def answer(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        session_context: Optional[str] = None,
        agent_trace: Optional[List[str]] = None,
        conversation_id: Optional[str] = None,
    ) -> QAResponse:
        traces = list(agent_trace or [])

        if not chunks:
            traces.append("No usable evidence was retrieved, returning refusal.")
            return make_refusal_response(
                query=query,
                answer="The knowledge base did not return enough evidence to answer this question reliably.",
                refusal_reason="no_chunks_retrieved",
                citations=[],
                used_chunks=[],
                conversation_id=conversation_id,
                agent_trace=traces,
            )

        decision = self.refusal_judge.judge(chunks)

        if decision.should_refuse:
            citations = self.build_citations(chunks[:1])
            used_chunks = [chunks[0].chunk_id]
            traces.append(f"Evidence strength is insufficient, refusal reason: {decision.reason}.")
            return make_refusal_response(
                query=query,
                answer="The current knowledge base does not contain enough reliable evidence for this question.",
                refusal_reason=decision.reason,
                citations=citations,
                used_chunks=used_chunks,
                conversation_id=conversation_id,
                agent_trace=traces,
            )

        answer_text = self.answer_builder.build_answer(query=query, chunks=chunks[:2])
        if session_context:
            traces.append("Recent conversation context was used while drafting the answer.")

        citations = self.build_citations(chunks[:2])
        used_chunks = [chunk.chunk_id for chunk in chunks[:2]]
        confidence = self._estimate_confidence(chunks)
        traces.append(f"Built grounded answer from {len(used_chunks)} evidence chunks.")

        return make_grounded_response(
            query=query,
            answer=answer_text,
            confidence=confidence,
            citations=citations,
            used_chunks=used_chunks,
            conversation_id=conversation_id,
            agent_trace=traces,
        )

    def build_citations(self, chunks: List[RetrievedChunk]) -> List[Citation]:
        citations: List[Citation] = []
        for chunk in chunks:
            citations.append(
                Citation(
                    source=chunk.source,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    chunk_id=chunk.chunk_id,
                    section_title=chunk.section_title,
                )
            )
        return citations

    def _estimate_confidence(self, chunks: List[RetrievedChunk]) -> str:
        if not chunks:
            return "low"

        top1 = chunks[0].score
        top2 = chunks[1].score if len(chunks) > 1 else 0.0

        if top1 >= 1.5:
            return "high"

        if top1 >= 1.0 and (top1 - top2) >= 0.08:
            return "high"

        if top1 >= 0.75:
            return "medium"

        return "low"
