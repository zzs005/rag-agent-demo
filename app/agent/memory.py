from __future__ import annotations

from collections import defaultdict
from threading import Lock
from typing import Dict, List
from uuid import uuid4

from app.agent.schemas import ConversationTurn


class InMemoryConversationStore:
    def __init__(self, max_turns: int = 8):
        self.max_turns = max_turns
        self._data: Dict[str, List[ConversationTurn]] = defaultdict(list)
        self._lock = Lock()

    def ensure_conversation(self, conversation_id: str | None = None) -> str:
        cid = conversation_id or uuid4().hex
        with self._lock:
            self._data.setdefault(cid, [])
        return cid

    def get_turns(self, conversation_id: str) -> List[ConversationTurn]:
        with self._lock:
            return list(self._data.get(conversation_id, []))

    def append(self, conversation_id: str, role: str, content: str) -> None:
        with self._lock:
            turns = self._data.setdefault(conversation_id, [])
            turns.append(ConversationTurn(role=role, content=content))
            if len(turns) > self.max_turns:
                self._data[conversation_id] = turns[-self.max_turns :]
