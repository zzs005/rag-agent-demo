from __future__ import annotations

import unittest

from app.agent.memory import InMemoryConversationStore
from app.agent.orchestrator import KnowledgeAgentOrchestrator
from app.qa.answerer import StructuredAnswerer
from app.retrieval.schemas import RetrievedChunk


class FakeSearchService:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        self.calls.append(query)

        if "补充追问" in query:
            return [
                RetrievedChunk(
                    chunk_id="c2",
                    document_id="d1",
                    document_name="学生手册",
                    source="学生手册",
                    chunk_index=1,
                    page_start=3,
                    page_end=3,
                    section_title="请假管理",
                    text="请假申请通常需要提前三天提交，并附上相关说明。",
                    char_count=24,
                    score=1.2,
                )
            ][:top_k]

        if "模糊问题" in query:
            return [
                RetrievedChunk(
                    chunk_id="c3",
                    document_id="d1",
                    document_name="学生手册",
                    source="学生手册",
                    chunk_index=2,
                    page_start=5,
                    page_end=5,
                    section_title="评优办法",
                    text="评优办法适用于不同学生群体。",
                    char_count=16,
                    score=0.62,
                ),
                RetrievedChunk(
                    chunk_id="c4",
                    document_id="d2",
                    document_name="本科生学习指南",
                    source="本科生学习指南",
                    chunk_index=4,
                    page_start=8,
                    page_end=8,
                    section_title="申请流程",
                    text="申请流程根据学院要求执行。",
                    char_count=15,
                    score=0.61,
                ),
            ][:top_k]

        return [
            RetrievedChunk(
                chunk_id="c1",
                document_id="d1",
                document_name="学生手册",
                source="学生手册",
                chunk_index=0,
                page_start=2,
                page_end=2,
                section_title="请假管理",
                text="请假申请应至少提前三天提交，特殊情况需要补充说明。",
                char_count=26,
                score=1.1,
            )
        ][:top_k]


class AgentOrchestratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.search_service = FakeSearchService()
        self.orchestrator = KnowledgeAgentOrchestrator(
            search_service=self.search_service,
            answerer=StructuredAnswerer(),
            memory=InMemoryConversationStore(),
        )

    def test_chat_creates_conversation_id(self) -> None:
        response = self.orchestrator.chat("请假申请需要提前多久提交？", top_k=3)
        self.assertTrue(response.conversation_id)
        self.assertEqual(response.answer_type, "grounded_answer")
        self.assertFalse(response.needs_clarification)

    def test_chat_returns_clarification_for_missing_context(self) -> None:
        response = self.orchestrator.chat("那时间呢", top_k=3)
        self.assertEqual(response.answer_type, "clarification")
        self.assertTrue(response.needs_clarification)

    def test_follow_up_query_uses_memory(self) -> None:
        first = self.orchestrator.chat("请假申请需要提前多久提交？", top_k=3)
        second = self.orchestrator.chat("那时间呢", top_k=3, conversation_id=first.conversation_id)
        self.assertEqual(second.answer_type, "grounded_answer")
        self.assertTrue(any("Follow-up question" in query for query in self.search_service.calls))

    def test_ambiguous_results_trigger_clarification(self) -> None:
        response = self.orchestrator.chat("模糊问题", top_k=2)
        self.assertEqual(response.answer_type, "clarification")
        self.assertTrue(response.needs_clarification)


if __name__ == "__main__":
    unittest.main()
