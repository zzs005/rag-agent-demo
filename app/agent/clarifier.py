from __future__ import annotations

import re
from typing import List, Optional

from app.agent.schemas import AgentDecision, ConversationTurn
from app.qa.refusal import RefusalDecision


FOLLOW_UP_HINTS = (
    "this",
    "that",
    "it",
    "then",
    "what about",
    "\u8fd9\u4e2a",
    "\u90a3\u4e2a",
    "\u90a3",
    "\u5b83",
    "\u4e0a\u8ff0",
    "\u4e0a\u9762",
    "\u524d\u8005",
    "\u540e\u8005",
)

SHORT_AMBIGUOUS_PATTERNS = (
    "\u600e\u4e48\u529e",
    "\u600e\u4e48\u505a",
    "\u6d41\u7a0b\u5462",
    "\u65f6\u95f4\u5462",
    "\u8981\u6c42\u5462",
    "\u6761\u4ef6\u5462",
)


class ClarificationEngine:
    def decide_before_search(self, query: str, turns: List[ConversationTurn]) -> Optional[AgentDecision]:
        normalized = query.strip()
        if not normalized:
            return AgentDecision(
                action="clarify",
                reason="empty_query",
                clarification_question="Please add the topic, object, or question you want to ask.",
            )

        if len(normalized) <= 4 and not turns:
            return AgentDecision(
                action="clarify",
                reason="query_too_short",
                clarification_question="Your question is still too short. Please add the topic, object, or scenario.",
            )

        if self._looks_like_follow_up(normalized) and not turns:
            return AgentDecision(
                action="clarify",
                reason="missing_context_for_follow_up",
                clarification_question="This looks like a follow-up question. Please specify which rule, process, or document topic you mean.",
            )

        if any(pattern == normalized for pattern in SHORT_AMBIGUOUS_PATTERNS):
            return AgentDecision(
                action="clarify",
                reason="missing_subject",
                clarification_question="Please add the specific subject, object, or rule you want to ask about.",
            )

        return None

    def decide_after_search(self, query: str, refusal: RefusalDecision) -> Optional[AgentDecision]:
        if refusal.reason == "top_results_ambiguous":
            return AgentDecision(
                action="clarify",
                reason="retrieval_ambiguous",
                clarification_question=(
                    f"I found several close matches for '{query}', but they are still ambiguous. "
                    "Please add the target group, time range, or concrete object."
                ),
            )
        return None

    def _looks_like_follow_up(self, query: str) -> bool:
        if any(query.lower().startswith(prefix) for prefix in FOLLOW_UP_HINTS):
            return True
        return bool(re.match(r"^(\u90a3|\u8fd9\u4e2a|\u5b83|\u4e0a\u8ff0)", query))
