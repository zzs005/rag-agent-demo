from __future__ import annotations

from typing import Optional
from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    source: str
    chunk_index: int
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section_title: Optional[str] = None
    text: str
    char_count: int
    score: float