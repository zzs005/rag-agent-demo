from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Literal, Dict, Any


ConfidenceLevel = Literal["high", "medium", "low"]
AnswerType = Literal["grounded_answer", "clarification", "refusal"]


@dataclass
class Citation:
    source: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    chunk_id: str = ""
    section_title: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QAResponse:
    query: str
    answer: str
    answer_type: AnswerType
    confidence: ConfidenceLevel
    citations: List[Citation] = field(default_factory=list)
    used_chunks: List[str] = field(default_factory=list)
    refused: bool = False
    refusal_reason: Optional[str] = None
    conversation_id: Optional[str] = None
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    agent_trace: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "answer_type": self.answer_type,
            "confidence": self.confidence,
            "citations": [c.to_dict() for c in self.citations],
            "used_chunks": self.used_chunks,
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "conversation_id": self.conversation_id,
            "needs_clarification": self.needs_clarification,
            "clarification_question": self.clarification_question,
            "agent_trace": self.agent_trace,
        }
