from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

from app.retrieval.schemas import RetrievedChunk


DecisionType = Literal["answer", "clarify", "refuse"]


@dataclass
class ConversationTurn:
    role: Literal["user", "assistant"]
    content: str


@dataclass
class RetrievalPlan:
    queries: List[str]
    strategy: Literal["single", "rewrite", "follow_up"]
    rationale: str


@dataclass
class AgentDecision:
    action: DecisionType
    reason: str
    clarification_question: Optional[str] = None


@dataclass
class EvidenceBundle:
    query: str
    chunks: List[RetrievedChunk] = field(default_factory=list)


@dataclass
class AgentChatResult:
    conversation_id: str
    rewritten_query: str
    traces: List[str]
