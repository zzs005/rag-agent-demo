from __future__ import annotations

from fastapi import APIRouter, Depends

from app.agent.orchestrator import KnowledgeAgentOrchestrator
from app.api.deps import get_agent_orchestrator
from app.api.schemas import ChatRequest, ChatResponse, CitationResponse


router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    orchestrator: KnowledgeAgentOrchestrator = Depends(get_agent_orchestrator),
) -> ChatResponse:
    response = orchestrator.chat(
        query=request.query,
        top_k=request.top_k,
        conversation_id=request.conversation_id,
    )

    citations = [
        CitationResponse(
            source=c.source,
            page_start=c.page_start,
            page_end=c.page_end,
            chunk_id=c.chunk_id,
            section_title=c.section_title,
        )
        for c in response.citations
    ]

    return ChatResponse(
        query=response.query,
        answer=response.answer,
        answer_type=response.answer_type,
        confidence=response.confidence,
        citations=citations,
        used_chunks=response.used_chunks,
        refused=response.refused,
        refusal_reason=response.refusal_reason,
        conversation_id=response.conversation_id or "",
        needs_clarification=response.needs_clarification,
        clarification_question=response.clarification_question,
        agent_trace=response.agent_trace,
    )
