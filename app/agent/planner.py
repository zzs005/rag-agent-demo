from __future__ import annotations

from typing import List

from app.agent.schemas import ConversationTurn, RetrievalPlan


class QueryPlanner:
    def build_plan(self, query: str, turns: List[ConversationTurn]) -> RetrievalPlan:
        rewritten = self._rewrite_query(query=query, turns=turns)
        queries = [rewritten]
        strategy = "single"
        rationale = "Search the user's query directly."

        if rewritten != query:
            queries.append(query)
            strategy = "follow_up"
            rationale = "Rewrite the follow-up question with conversation context and keep the raw query as fallback."
        elif self._needs_second_pass(query):
            queries.append(self._expand_query(query))
            strategy = "rewrite"
            rationale = "Run a second retrieval pass with a rule-based expanded query."

        return RetrievalPlan(queries=queries, strategy=strategy, rationale=rationale)

    def _rewrite_query(self, query: str, turns: List[ConversationTurn]) -> str:
        stripped = query.strip()
        if not turns:
            return stripped

        last_user_turn = next((turn.content for turn in reversed(turns) if turn.role == "user"), "")
        if not last_user_turn:
            return stripped

        follow_up_prefixes = ("that", "this", "it", "\u90a3", "\u8fd9\u4e2a", "\u5b83", "\u4e0a\u8ff0")
        if any(stripped.lower().startswith(prefix) for prefix in follow_up_prefixes):
            return f"{last_user_turn}. Follow-up question: {stripped}"

        follow_up_keywords = (
            "time",
            "requirement",
            "condition",
            "material",
            "process",
            "\u65f6\u95f4",
            "\u8981\u6c42",
            "\u6761\u4ef6",
            "\u6750\u6599",
            "\u6d41\u7a0b",
        )
        if len(stripped) <= 12 and any(keyword in stripped for keyword in follow_up_keywords):
            return f"{last_user_turn}. Continued user question: {stripped}"

        return stripped

    def _needs_second_pass(self, query: str) -> bool:
        markers = ("and", "difference", "respectively", "together", "\u4ee5\u53ca", "\u5e76\u4e14", "\u533a\u522b", "\u5206\u522b", "\u540c\u65f6")
        return len(query) >= 16 and any(token in query.lower() for token in markers)

    def _expand_query(self, query: str) -> str:
        return f"{query} relevant rules conditions process scope"
