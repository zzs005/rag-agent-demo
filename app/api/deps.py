from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from app.agent.memory import InMemoryConversationStore
from app.agent.orchestrator import KnowledgeAgentOrchestrator
from app.qa.answerer import StructuredAnswerer
from app.retrieval.search_service import SearchService


ROOT = Path(__file__).resolve().parent.parent.parent
INDEX_DIR = ROOT / "data" / "index"

VECTORS_PATH = INDEX_DIR / "chunks_vectors"
METADATA_PATH = INDEX_DIR / "chunks_metadata.json"


@lru_cache(maxsize=1)
def get_search_service() -> SearchService:
    return SearchService(
        vectors_path=str(VECTORS_PATH),
        metadata_path=str(METADATA_PATH),
    )


@lru_cache(maxsize=1)
def get_answerer() -> StructuredAnswerer:
    return StructuredAnswerer()


@lru_cache(maxsize=1)
def get_conversation_store() -> InMemoryConversationStore:
    return InMemoryConversationStore()


def get_agent_orchestrator() -> KnowledgeAgentOrchestrator:
    return KnowledgeAgentOrchestrator(
        search_service=get_search_service(),
        answerer=get_answerer(),
        memory=get_conversation_store(),
    )
