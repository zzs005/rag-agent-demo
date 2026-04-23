from __future__ import annotations

from typing import List

from app.agent.clarifier import ClarificationEngine
from app.agent.memory import InMemoryConversationStore
from app.agent.planner import QueryPlanner
from app.qa.answerer import StructuredAnswerer
from app.qa.response_formatter import make_clarification_response
from app.qa.schemas import QAResponse
from app.retrieval.schemas import RetrievedChunk
from app.retrieval.search_service import SearchService


class KnowledgeAgentOrchestrator:
    def __init__(
        self,
        search_service: SearchService,
        answerer: StructuredAnswerer,
        memory: InMemoryConversationStore,
    ):
        self.search_service = search_service
        self.answerer = answerer
        self.memory = memory
        self.planner = QueryPlanner()
        self.clarifier = ClarificationEngine()

    def chat(self, query: str, top_k: int = 5, conversation_id: str | None = None) -> QAResponse:
        cid = self.memory.ensure_conversation(conversation_id)
        turns = self.memory.get_turns(cid)
        traces: List[str] = []

        pre_search_decision = self.clarifier.decide_before_search(query=query, turns=turns)
        if pre_search_decision is not None:
            traces.append(f"Clarification required before retrieval: {pre_search_decision.reason}.")
            response = make_clarification_response(
                query=query,
                clarification_question=pre_search_decision.clarification_question or "Please add more detail.",
                conversation_id=cid,
                agent_trace=traces,
            )
            self._record_turns(cid, query, response.answer)
            return response

        plan = self.planner.build_plan(query=query, turns=turns)
        traces.append(f"Built retrieval plan: {plan.rationale}")

        all_results: List[RetrievedChunk] = []
        seen_chunk_ids: set[str] = set()
        for planned_query in plan.queries:
            results = self.search_service.search(query=planned_query, top_k=top_k)
            traces.append(f"Executed search for '{planned_query}' and received {len(results)} results.")
            for item in results:
                if item.chunk_id not in seen_chunk_ids:
                    all_results.append(item)
                    seen_chunk_ids.add(item.chunk_id)

            if all_results and all_results[0].score >= 0.9:
                traces.append("Top evidence is already strong enough; stop additional retrieval passes.")
                break

        all_results.sort(key=lambda item: item.score, reverse=True)
        selected = all_results[:top_k]

        refusal_decision = self.answerer.refusal_judge.judge(selected)
        post_search_decision = self.clarifier.decide_after_search(query=query, refusal=refusal_decision)
        if post_search_decision is not None:
            traces.append(f"Retrieved evidence is ambiguous: {post_search_decision.reason}.")
            response = make_clarification_response(
                query=query,
                clarification_question=post_search_decision.clarification_question or "Please clarify the target of the question.",
                conversation_id=cid,
                citations=self.answerer.build_citations(selected[:1]) if selected else [],
                used_chunks=[selected[0].chunk_id] if selected else [],
                agent_trace=traces,
            )
            self._record_turns(cid, query, response.answer)
            return response

        context = self._build_session_context(turns)
        response = self.answerer.answer(
            query=query,
            chunks=selected,
            session_context=context,
            agent_trace=traces,
            conversation_id=cid,
        )
        self._record_turns(cid, query, response.answer)
        return response

    def _build_session_context(self, turns) -> str | None:
        recent_user_turns = [turn.content for turn in turns if turn.role == "user"][-2:]
        if not recent_user_turns:
            return None
        return " | ".join(recent_user_turns)

    def _record_turns(self, conversation_id: str, user_query: str, assistant_answer: str) -> None:
        self.memory.append(conversation_id, "user", user_query)
        self.memory.append(conversation_id, "assistant", assistant_answer)
