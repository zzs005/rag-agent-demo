from __future__ import annotations

from typing import List, Optional

from app.qa.schemas import Citation, QAResponse


def make_refusal_response(
    query: str,
    answer: str,
    refusal_reason: Optional[str],
    citations: List[Citation] | None = None,
    used_chunks: List[str] | None = None,
    conversation_id: Optional[str] = None,
    agent_trace: List[str] | None = None,
) -> QAResponse:
    return QAResponse(
        query=query,
        answer=answer,
        answer_type="refusal",
        confidence="low",
        citations=citations or [],
        used_chunks=used_chunks or [],
        refused=True,
        refusal_reason=refusal_reason,
        conversation_id=conversation_id,
        needs_clarification=False,
        clarification_question=None,
        agent_trace=agent_trace or [],
    )


def make_grounded_response(
    query: str,
    answer: str,
    confidence: str,
    citations: List[Citation],
    used_chunks: List[str],
    conversation_id: Optional[str] = None,
    agent_trace: List[str] | None = None,
) -> QAResponse:
    return QAResponse(
        query=query,
        answer=answer,
        answer_type="grounded_answer",
        confidence=confidence,
        citations=citations,
        used_chunks=used_chunks,
        refused=False,
        refusal_reason=None,
        conversation_id=conversation_id,
        needs_clarification=False,
        clarification_question=None,
        agent_trace=agent_trace or [],
    )


def make_clarification_response(
    query: str,
    clarification_question: str,
    answer: Optional[str] = None,
    conversation_id: Optional[str] = None,
    citations: List[Citation] | None = None,
    used_chunks: List[str] | None = None,
    agent_trace: List[str] | None = None,
) -> QAResponse:
    prompt = answer or (
        "To answer your question more accurately, I need to clarify one thing first: "
        f"{clarification_question}"
    )
    return QAResponse(
        query=query,
        answer=prompt,
        answer_type="clarification",
        confidence="low",
        citations=citations or [],
        used_chunks=used_chunks or [],
        refused=False,
        refusal_reason=None,
        conversation_id=conversation_id,
        needs_clarification=True,
        clarification_question=clarification_question,
        agent_trace=agent_trace or [],
    )
